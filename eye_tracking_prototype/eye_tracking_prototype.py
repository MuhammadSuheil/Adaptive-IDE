import os
import cv2
import csv
import json
import uuid
import time
import math
import numpy as np
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from modules import Config, WebcamStream, GazeFilter, MetricsEngine, GazeMapper

class EyeTrackerApp:
    def __init__(self, config_path):
        self.cfg = Config(config_path)
        
        os.makedirs(self.cfg.session_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}.csv")
        self.json_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}_summary.json")
        
        if self.cfg.async_capture:
            print("[EyeTrack] Initializing Async Multithreaded Camera Stream...")
            self.stream = WebcamStream(
                self.cfg.webcam_idx, self.cfg.webcam_w, self.cfg.webcam_h, 
                self.cfg.webcam_fps, self.cfg.flip_horizontal
            ).start()
        else:
            self.stream = None
            self.cap = cv2.VideoCapture(self.cfg.webcam_idx)
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.webcam_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.webcam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.webcam_h)
        
        base_options = python.BaseOptions(model_asset_path=self.cfg.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.cfg.data['mediapipe']['num_faces'],
            min_face_detection_confidence=self.cfg.data['mediapipe']['min_face_detection_confidence'],
            min_face_presence_confidence=self.cfg.data['mediapipe']['min_face_presence_confidence'],
            min_tracking_confidence=self.cfg.data['mediapipe']['min_tracking_confidence'],
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.gaze_filter = GazeFilter(self.cfg)
        self.metrics = MetricsEngine(self.cfg)
        self.mapper = GazeMapper(method=self.cfg.calib_mapping_method)
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ms", "frame_index", 
            "gaze_x_raw", "gaze_y_raw", "gaze_x_smooth", "gaze_y_smooth",
            "grid_row", "grid_col", "section", "confidence",
            "dwell_time_ms", "nrevisit_count", "transition_rate",
            "iris_size_delta", "fps_actual", "capture_fps", "inference_ms",
            "face_detected", "gaze_status", "calibration_quality"
        ])
        
        self.calibration_quality = 0.0
        self.frame_count = 0
        self.total_face_frames = 0
        self.start_time = time.time()
        self.is_paused = False
        self.show_grid = True
        self.debug_mode = False
        self._last_ts_ms = 0
        self.tracking_start_time = None
        self.last_frame_time = None
        self.fps_ema = 0.0
        self.exit_requested = False
        self.calibration_diagnostics = {}
        
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        except Exception:
            self.screen_w = 1920
            self.screen_h = 1080
        print(f"[EyeTrack] Detected screen resolution: {self.screen_w}x{self.screen_h}")

    def read_frame(self):
        if self.stream is not None:
            return self.stream.read()
        else:
            ret, frame = self.cap.read()
            if ret and self.cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)
            return ret, frame

    def camera_diagnostics(self):
        if self.stream is not None:
            return self.stream.diagnostics()
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "capture_fps_observed": None,
            "backend": self.cap.getBackendName() if self.cap.isOpened() else "closed",
        }

    def make_mediapipe_image(self, frame):
        inference_frame = frame
        if frame.shape[1] != self.cfg.inference_w or frame.shape[0] != self.cfg.inference_h:
            inference_frame = cv2.resize(
                frame, (self.cfg.inference_w, self.cfg.inference_h), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def observed_capture_fps(self):
        return self.stream.observed_fps() if self.stream is not None else None

    def calculate_ear(self, landmarks):
        l_top = landmarks[self.cfg.left_eyelid[0]]
        l_bot = landmarks[self.cfg.left_eyelid[1]]
        l_in = landmarks[self.cfg.left_eyelid[2]]
        l_out = landmarks[self.cfg.left_eyelid[3]]

        r_top = landmarks[self.cfg.right_eyelid[0]]
        r_bot = landmarks[self.cfg.right_eyelid[1]]
        r_out = landmarks[self.cfg.right_eyelid[2]]
        r_in = landmarks[self.cfg.right_eyelid[3]]

        l_ear = math.hypot(l_top.x - l_bot.x, l_top.y - l_bot.y) / (math.hypot(l_in.x - l_out.x, l_in.y - l_out.y) + 1e-6)
        r_ear = math.hypot(r_top.x - r_bot.x, r_top.y - r_bot.y) / (math.hypot(r_in.x - r_out.x, r_in.y - r_out.y) + 1e-6)

        return (l_ear + r_ear) / 2.0

    def get_normalized_eye_vector(self, landmarks):
        left_corner_outer = landmarks[self.cfg.left_eye_corners[0]]
        left_corner_inner = landmarks[self.cfg.left_eye_corners[1]]
        left_iris_pts = [landmarks[i] for i in self.cfg.left_iris_indices if i < len(landmarks)]
        
        right_corner_inner = landmarks[self.cfg.right_eye_corners[0]]
        right_corner_outer = landmarks[self.cfg.right_eye_corners[1]]
        right_iris_pts = [landmarks[i] for i in self.cfg.right_iris_indices if i < len(landmarks)]

        if not left_iris_pts or not right_iris_pts:
            return (0.5, 0.5, 0.2), 0.0

        l_iris_x = sum(p.x for p in left_iris_pts) / len(left_iris_pts)
        l_iris_y = sum(p.y for p in left_iris_pts) / len(left_iris_pts)
        
        r_iris_x = sum(p.x for p in right_iris_pts) / len(right_iris_pts)
        r_iris_y = sum(p.y for p in right_iris_pts) / len(right_iris_pts)

        def eye_local(iris_x, iris_y, corner_a, corner_b):
            # Always orient the local X axis toward image-right, then project Y onto
            # a perpendicular axis that points image-down. This compensates head roll.
            left, right = sorted((corner_a, corner_b), key=lambda p: p.x)
            ax, ay = right.x - left.x, right.y - left.y
            length_sq = ax * ax + ay * ay
            if length_sq < 1e-9:
                return 0.5, 0.0
            vx, vy = iris_x - left.x, iris_y - left.y
            local_x = (vx * ax + vy * ay) / length_sq
            nx, ny = -ay, ax
            if ny < 0:
                nx, ny = -nx, -ny
            local_y = (vx * nx + vy * ny) / length_sq
            return local_x, local_y

        l_norm_x, l_norm_y = eye_local(
            l_iris_x, l_iris_y, left_corner_outer, left_corner_inner
        )
        r_norm_x, r_norm_y = eye_local(
            r_iris_x, r_iris_y, right_corner_inner, right_corner_outer
        )

        norm_x = (l_norm_x + r_norm_x) / 2.0
        norm_y = (l_norm_y + r_norm_y) / 2.0

        ear = self.calculate_ear(landmarks)

        l_size = math.hypot(left_iris_pts[0].x - left_iris_pts[2].x, left_iris_pts[0].y - left_iris_pts[2].y) if len(left_iris_pts) >= 3 else 0.01
        r_size = math.hypot(right_iris_pts[0].x - right_iris_pts[2].x, right_iris_pts[0].y - right_iris_pts[2].y) if len(right_iris_pts) >= 3 else 0.01
        iris_size = (l_size + r_size) / 2.0

        return (norm_x, norm_y, ear), iris_size

    def wait_for_calibration_retry(self):
        cv2.namedWindow("Calibration Required", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration Required", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        panel = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        cv2.putText(panel, "CALIBRATION WAS REJECTED", (80, self.screen_h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 255), 3)
        cv2.putText(panel, "Press R to retry, or Q to quit.", (80, self.screen_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        cv2.putText(panel, "Keep your head still and stare at the center of each dot.",
                    (80, self.screen_h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        while True:
            cv2.imshow("Calibration Required", panel)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                cv2.destroyWindow("Calibration Required")
                return True
            if key == ord('q'):
                cv2.destroyWindow("Calibration Required")
                self.exit_requested = True
                return False

    def calibrate_until_ready(self):
        while not self.exit_requested:
            if self.run_calibration():
                return True
            if self.exit_requested or not self.wait_for_calibration_retry():
                return False
        return False

    def _next_ts(self):
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def run_calibration(self):
        print("\n[EyeTrack] Starting Enhanced Calibration Flow...")
        self.gaze_filter.reset()
        
        grid_r = self.cfg.grid_rows
        grid_c = self.cfg.grid_cols
        
        eye_features = []
        screen_pts = []
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        boundary_clearance = 0.0
        if self.cfg.gaze_boundary_enabled:
            boundary_clearance = max(
                self.cfg.gaze_boundary_pad_x, self.cfg.gaze_boundary_pad_y
            ) + 0.02
        margin = min(max(self.cfg.calib_target_margin, boundary_clearance, 0.05), 0.30)
        target_xs = np.linspace(self.screen_w * margin, self.screen_w * (1.0 - margin), grid_c)
        target_ys = np.linspace(self.screen_h * margin, self.screen_h * (1.0 - margin), grid_r)
        
        pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x) if self.cfg.gaze_boundary_enabled else 0
        pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y) if self.cfg.gaze_boundary_enabled else 0

        MOVE_TIME = self.cfg.calib_move_delay_sec
        
        for r in range(grid_r):
            for c in range(grid_c):
                dot_x = int(target_xs[c])
                dot_y = int(target_ys[r])

                dot_num = r * grid_c + c + 1
                total_dots = grid_r * grid_c
                
                move_start = time.time()
                while time.time() - move_start < MOVE_TIME:
                    elapsed = time.time() - move_start
                    remaining = MOVE_TIME - elapsed
                    bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    if self.cfg.gaze_boundary_enabled:
                        bg[:] = (100, 0, 100)
                        cv2.rectangle(bg, (pad_x, pad_y), (self.screen_w - pad_x, self.screen_h - pad_y), (0, 0, 0), -1)
                        cv2.rectangle(bg, (pad_x, pad_y), (self.screen_w - pad_x, self.screen_h - pad_y), (255, 255, 255), 2)
                        cv2.putText(bg, "off screen", (self.screen_w // 2 - 45, self.screen_h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    cv2.circle(bg, (dot_x, dot_y), self.cfg.calib_radius, self.cfg.calib_color, -1)
                    cv2.putText(bg, f"LOOK AT THE CENTER OF THE RED DOT ({dot_num}/{total_dots})",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(bg, f"Keep your head still; move only your eyes. Recording in {remaining:.1f}s...",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                    cv2.imshow("Calibration", bg)
                    cv2.waitKey(30)
                
                samples_collected = 0
                collected_feats = []
                stable_streak = 0
                prev_ix, prev_iy = None, None
                point_started = time.time()
                
                while samples_collected < self.cfg.calib_samples:
                    if time.time() - point_started > self.cfg.calib_point_timeout_sec:
                        break
                    ret, frame = self.read_frame()
                    if not ret or frame is None: continue
                    
                    mp_img = self.make_mediapipe_image(frame)
                    ts_ms = self._next_ts()
                    
                    try:
                        res = self.landmarker.detect_for_video(mp_img, ts_ms)
                        if not res.face_landmarks:
                            stable_streak = 0
                            cv2.waitKey(1)
                            continue
                        
                        (ix, iy, ear), _ = self.get_normalized_eye_vector(res.face_landmarks[0])
                        if not (self.cfg.min_ear <= ear <= self.cfg.max_ear):
                            stable_streak = 0
                            cv2.waitKey(1)
                            continue
                        
                        if prev_ix is not None:
                            drift = math.hypot(ix - prev_ix, iy - prev_iy)
                            if drift < self.cfg.calib_stability_thresh:
                                stable_streak += 1
                            else:
                                stable_streak = 0
                        prev_ix, prev_iy = ix, iy
                        
                        if stable_streak >= self.cfg.calib_stability_frames:
                            collected_feats.append([ix, iy, ear])
                            samples_collected += 1
                        
                        copy_bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                        if self.cfg.gaze_boundary_enabled:
                            copy_bg[:] = (100, 0, 100)
                            cv2.rectangle(copy_bg, (pad_x, pad_y), (self.screen_w - pad_x, self.screen_h - pad_y), (0, 0, 0), -1)
                            cv2.rectangle(copy_bg, (pad_x, pad_y), (self.screen_w - pad_x, self.screen_h - pad_y), (255, 255, 255), 2)
                            cv2.putText(copy_bg, "off screen", (self.screen_w // 2 - 45, self.screen_h - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                        dot_color = (0, 255, 0) if stable_streak >= self.cfg.calib_stability_frames else self.cfg.calib_color
                        cv2.circle(copy_bg, (dot_x, dot_y), self.cfg.calib_radius, dot_color, -1)
                        ring_r = self.cfg.calib_radius + 20 - int(20 * samples_collected / self.cfg.calib_samples)
                        cv2.circle(copy_bg, (dot_x, dot_y), ring_r, (0, 200, 0), 2)
                        cv2.putText(copy_bg, f"RECORDING GAZE — {samples_collected}/{self.cfg.calib_samples} ({dot_num}/{total_dots})",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        if stable_streak < self.cfg.calib_stability_frames:
                            cv2.putText(copy_bg, "KEEP EYES FIXED ON THE DOT...",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                        cv2.imshow("Calibration", copy_bg)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.exit_requested = True
                            cv2.destroyWindow("Calibration")
                            return False
                        
                    except Exception as e:
                        stable_streak = 0
                
                if len(collected_feats) < self.cfg.calib_samples:
                    print(f"[EyeTrack] Calibration point {dot_num} timed out; restarting calibration.")
                    cv2.destroyWindow("Calibration")
                    return False

                samples = np.asarray(collected_feats, dtype=np.float64)
                median = np.median(samples, axis=0)
                mad = np.median(np.abs(samples - median), axis=0) + 1e-6
                robust_z = np.max(np.abs(samples - median) / (1.4826 * mad), axis=1)
                kept = samples[robust_z <= self.cfg.calib_outlier_mad_scale]
                if len(kept) < max(10, self.cfg.calib_samples // 2):
                    kept = samples
                point_std = float(np.max(np.std(kept[:, :2], axis=0)))
                if point_std > self.cfg.calib_max_sample_std:
                    print(f"[EyeTrack] Calibration point {dot_num} was noisy (std={point_std:.4f}); restarting.")
                    cv2.destroyWindow("Calibration")
                    return False

                avg_feat = np.median(kept, axis=0).tolist()
                eye_features.append(avg_feat)
                screen_pts.append([dot_x, dot_y])
                
        cv2.destroyWindow("Calibration")
        
        feature_array = np.asarray(eye_features)
        span_x = float(np.ptp(feature_array[:, 0]))
        span_y = float(np.ptp(feature_array[:, 1]))
        if span_x < self.cfg.calib_min_feature_span_x or span_y < self.cfg.calib_min_feature_span_y:
            self.calibration_quality = 0.0
            self.calibration_diagnostics = {"feature_span_x": span_x, "feature_span_y": span_y}
            print(f"[EyeTrack] Calibration rejected: insufficient feature separation X={span_x:.4f}, Y={span_y:.4f}")
            return False

        try:
            self.calibration_quality = self.mapper.fit(
                eye_features, screen_pts, screen_size=(self.screen_w, self.screen_h)
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.calibration_quality = 0.0
            self.calibration_diagnostics = {"fit_error": str(exc)}
            print(f"[EyeTrack] Calibration model failed: {exc}")
            return False
        self.calibration_diagnostics = {
            **self.mapper.diagnostics, "feature_span_x": span_x, "feature_span_y": span_y
        }
        print(f"[EyeTrack] Calibration complete! Quality ({self.mapper.method}): {self.calibration_quality:.2f}")
        print(f"[EyeTrack] Validation: {self.mapper.diagnostics}")
        if self.calibration_quality < self.cfg.calib_min_quality:
            print(f"[EyeTrack] REJECTED: quality is below {self.cfg.calib_min_quality:.2f}.")
            return False
        return True

    def is_gaze_on_screen(self, screen_x, screen_y):
        if not self.cfg.gaze_boundary_enabled:
            return True
        pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x)
        pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y)
        return (pad_x <= screen_x <= self.screen_w - pad_x) and (pad_y <= screen_y <= self.screen_h - pad_y)

    def get_grid_cell(self, screen_x, screen_y):
        if self.cfg.gaze_boundary_enabled and not self.is_gaze_on_screen(screen_x, screen_y):
            return -1, -1, self.cfg.off_screen_label, 0.0

        if self.cfg.gaze_boundary_enabled:
            pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x)
            pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y)
            active_w = max(1, self.screen_w - 2 * pad_x)
            active_h = max(1, self.screen_h - 2 * pad_y)
            col = int((screen_x - pad_x) / (active_w / self.cfg.grid_cols))
            row = int((screen_y - pad_y) / (active_h / self.cfg.grid_rows))
        else:
            col = int(screen_x / (self.screen_w / self.cfg.grid_cols))
            row = int(screen_y / (self.screen_h / self.cfg.grid_rows))
        
        col = max(0, min(col, self.cfg.grid_cols - 1))
        row = max(0, min(row, self.cfg.grid_rows - 1))
        
        section = self.cfg.section_map[row][col]
        
        if self.cfg.gaze_boundary_enabled:
            pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x)
            pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y)
            active_w = max(1, self.screen_w - 2 * pad_x)
            active_h = max(1, self.screen_h - 2 * pad_y)
            cx = pad_x + (col + 0.5) * (active_w / self.cfg.grid_cols)
            cy = pad_y + (row + 0.5) * (active_h / self.cfg.grid_rows)
            max_dist = math.hypot(active_w / self.cfg.grid_cols / 2, active_h / self.cfg.grid_rows / 2)
        else:
            cx = (col + 0.5) * (self.screen_w / self.cfg.grid_cols)
            cy = (row + 0.5) * (self.screen_h / self.cfg.grid_rows)
            max_dist = math.hypot(self.screen_w / self.cfg.grid_cols / 2, self.screen_h / self.cfg.grid_rows / 2)
            
        dist = math.hypot(screen_x - cx, screen_y - cy)
        conf = max(0.0, 1.0 - (dist / max_dist))
        
        return row, col, section, float(conf)

    def draw_grid_overlay(self, sx, sy, row, col, section):
        grid_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        is_off_screen = (section == self.cfg.off_screen_label)

        if self.cfg.gaze_boundary_enabled:
            pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x)
            pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y)
            active_w = self.screen_w - 2 * pad_x
            active_h = self.screen_h - 2 * pad_y

            # Subdued red background tint if off-screen, purple padding if on-screen
            pad_color = (25, 25, 100) if is_off_screen else (128, 0, 128)
            grid_img[:] = pad_color

            # Draw central active screen area
            cv2.rectangle(grid_img, (pad_x, pad_y), (pad_x + active_w, pad_y + active_h), (0, 0, 0), -1)

            cell_w = int(active_w / self.cfg.grid_cols)
            cell_h = int(active_h / self.cfg.grid_rows)

            if not is_off_screen and section is not None and row >= 0 and col >= 0:
                color = self.cfg.section_colors.get(section, [128, 128, 128])
                tl = (pad_x + col * cell_w, pad_y + row * cell_h)
                br = (pad_x + (col + 1) * cell_w, pad_y + (row + 1) * cell_h)
                cv2.rectangle(grid_img, tl, br, color, -1)

            for r in range(self.cfg.grid_rows):
                for c in range(self.cfg.grid_cols):
                    tl = (pad_x + c * cell_w, pad_y + r * cell_h)
                    br = (pad_x + (c + 1) * cell_w, pad_y + (r + 1) * cell_h)
                    cv2.rectangle(grid_img, tl, br, (255, 255, 255), 1)

                    sec_lbl = self.cfg.section_map[r][c]
                    if sec_lbl:
                        cv2.putText(grid_img, sec_lbl, (tl[0] + 10, tl[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            border_color = (40, 40, 200) if is_off_screen else (0, 255, 255)
            cv2.rectangle(grid_img, (pad_x, pad_y), (pad_x + active_w, pad_y + active_h), border_color, 2)
            cv2.putText(grid_img, "off screen", (self.screen_w // 2 - 45, self.screen_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cell_w = int(self.screen_w / self.cfg.grid_cols)
            cell_h = int(self.screen_h / self.cfg.grid_rows)

            if section is not None:
                color = self.cfg.section_colors.get(section, [128, 128, 128])
                tl = (col * cell_w, row * cell_h)
                br = ((col + 1) * cell_w, (row + 1) * cell_h)
                cv2.rectangle(grid_img, tl, br, color, -1)

            for r in range(self.cfg.grid_rows):
                for c in range(self.cfg.grid_cols):
                    tl = (c * cell_w, r * cell_h)
                    br = ((c + 1) * cell_w, (r + 1) * cell_h)
                    cv2.rectangle(grid_img, tl, br, (255, 255, 255), 1)

                    sec_lbl = self.cfg.section_map[r][c]
                    if sec_lbl:
                        cv2.putText(grid_img, sec_lbl, (tl[0] + 10, tl[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if is_off_screen:
            cx, cy = int(np.clip(sx, 10, self.screen_w - 10)), int(np.clip(sy, 10, self.screen_h - 10))
            cv2.circle(grid_img, (cx, cy), 12, (30, 30, 160), -1)
            cv2.circle(grid_img, (cx, cy), 16, (40, 40, 220), 2)
            cv2.putText(grid_img, "GAZE: OFF SCREEN", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 220), 2)
        else:
            cv2.circle(grid_img, (int(sx), int(sy)), 15, (0, 0, 255), -1)

        cv2.putText(grid_img, "[Q] Quit  [C] Recalibrate  [S] Snapshot  [P] Pause  [G] Toggle Grid",
                    (20, self.screen_h - 20 if not self.cfg.gaze_boundary_enabled else 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Eye Tracking - Gaze Grid", grid_img)


    def run(self):
        if not self.calibrate_until_ready():
            self.cleanup()
            return

        self.tracking_start_time = time.time()
        self.last_frame_time = None
        print(f"[EyeTrack] Camera diagnostics: {self.camera_diagnostics()}")
        
        cv2.namedWindow("Eye Tracking - Gaze Grid", cv2.WINDOW_NORMAL)
        print("[EyeTrack] Starting tracking loop...")
        
        while True:
            iter_start = time.time()
            ret, frame = self.read_frame()
            if not ret or frame is None: continue
            
            if self.is_paused:
                cv2.putText(frame, "PAUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Eye Tracking - Camera Feed", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p'): self.is_paused = False
                elif key == ord('q'): break
                continue
                
            self.frame_count += 1
            ts_ms = int(iter_start * 1000)
            if self.last_frame_time is not None:
                instantaneous_fps = 1.0 / max(iter_start - self.last_frame_time, 1e-6)
                self.fps_ema = instantaneous_fps if self.fps_ema == 0 else 0.9 * self.fps_ema + 0.1 * instantaneous_fps
            self.last_frame_time = iter_start
            
            mp_img = self.make_mediapipe_image(frame)
            
            face_detected = False
            raw_x, raw_y = 0.0, 0.0
            sm_x, sm_y = 0.0, 0.0
            grid_r, grid_c = -1, -1
            section = None
            conf = 0.0
            iris_size = 0.0
            gaze_status = "face_missing"
            
            res = None
            inference_started = time.perf_counter()
            try:
                res = self.landmarker.detect_for_video(mp_img, self._next_ts())
            except Exception as e:
                print(f"[Warn] MediaPipe error: {e}")
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
                
            if res and res.face_landmarks:
                face_detected = True
                self.total_face_frames += 1
                
                lm = res.face_landmarks[0]
                (norm_x, norm_y, ear), iris_size = self.get_normalized_eye_vector(lm)
                
                if self.debug_mode:
                    for p in lm:
                        px = int(p.x * frame.shape[1])
                        py = int(p.y * frame.shape[0])
                        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
                
                if not (self.cfg.min_ear <= ear <= self.cfg.max_ear):
                    gaze_status = "eyes_invalid_or_blink"
                else:
                    raw_x, raw_y = self.mapper.predict(norm_x, norm_y, ear)
                    sm_x, sm_y = self.gaze_filter.update(raw_x, raw_y)
                    grid_r, grid_c, section, conf = self.get_grid_cell(sm_x, sm_y)
                    gaze_status = "gaze_outside_screen" if section == self.cfg.off_screen_label else "on_screen"
                
            self.metrics.update(ts_ms, section, iris_size)
            
            fps = self.fps_ema
            capture_fps = self.observed_capture_fps()
            
            row = [
                ts_ms, self.frame_count,
                raw_x, raw_y, sm_x, sm_y,
                grid_r, grid_c, section if section else "", conf,
                self.metrics.dwell_time_ms,
                self.metrics.get_nrevisit(section),
                self.metrics.get_transition_rate(),
                self.metrics.iris_delta, fps, capture_fps if capture_fps is not None else "",
                inference_ms, face_detected, gaze_status, self.calibration_quality
            ]
            self.csv_writer.writerow(row)
            
            if self.frame_count % self.cfg.csv_buffer_size == 0:
                self.csv_file.flush()
                
            hud_y = 30
            is_off_screen = (section == self.cfg.off_screen_label)
            hud_color = (40, 40, 200) if is_off_screen else (0, 255, 0)
            status_text = f"Status: {gaze_status}" if gaze_status != "on_screen" else f"Section: {section}"
            lines = [
                f"Processing FPS: {fps:.1f}",
                f"Inference: {inference_ms:.1f}ms",
                status_text,
                f"Dwell: {self.metrics.dwell_time_ms / 1000.0:.1f}s",
                f"NRevisit: {self.metrics.get_nrevisit(section)}",
                f"Trans Rate: {self.metrics.get_transition_rate():.2f}/s",
                f"Iris D: {self.metrics.iris_delta:.4f}"
            ]
            for l in lines:
                cv2.putText(frame, l, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
                hud_y += 25

                
            cv2.imshow("Eye Tracking - Camera Feed", frame)
            
            if self.show_grid:
                self.draw_grid_overlay(sm_x, sm_y, grid_r, grid_c, section)
            else:
                try: cv2.destroyWindow("Eye Tracking - Gaze Grid")
                except: pass
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if not self.calibrate_until_ready():
                    break
                self.gaze_filter.reset()
                self.last_frame_time = None
            elif key == ord('p'):
                self.is_paused = True
            elif key == ord('g'):
                self.show_grid = not self.show_grid
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
            elif key == ord('s'):
                cv2.imwrite(f"snapshot_{ts_ms}.png", frame)
                print(f"[EyeTrack] Saved snapshot_{ts_ms}.png")

        self.cleanup()

    def cleanup(self):
        print("\n[EyeTrack] Cleaning up...")
        camera_stats = self.camera_diagnostics()
        if self.stream is not None:
            self.stream.stop()
        else:
            self.cap.release()
            
        cv2.destroyAllWindows()
        self.csv_file.flush()
        self.csv_file.close()
        self.metrics.finalize()
        
        duration = time.time() - self.start_time
        tracking_duration = (time.time() - self.tracking_start_time) if self.tracking_start_time else 0.0
        summary = {
            "session_id": self.session_id,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time_iso": datetime.now().isoformat(),
            "duration_seconds": duration,
            "tracking_duration_seconds": tracking_duration,
            "config": {
                "grid_rows": self.cfg.grid_rows,
                "grid_cols": self.cfg.grid_cols,
                "filter_type": self.cfg.filter_type,
                "mapping_method": self.cfg.calib_mapping_method,
                "async_capture": self.cfg.async_capture
            },
            "summary": {
                "total_frames": self.frame_count,
                "frames_with_face": self.total_face_frames,
                "face_detection_rate": self.total_face_frames / self.frame_count if self.frame_count > 0 else 0,
                "avg_fps": self.frame_count / tracking_duration if tracking_duration > 0 else 0,
                "processing_fps_ema_final": self.fps_ema,
                "camera": camera_stats,
                "sections": self.metrics.session_sections_summary,
                "calibration_quality": self.calibration_quality,
                "calibration_diagnostics": self.calibration_diagnostics
            },
            "csv_path": self.csv_path
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"[EyeTrack] Session saved to {self.csv_path} and {self.json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive IDE - Eye Tracking Prototype")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    app = EyeTrackerApp(args.config)
    app.run()
