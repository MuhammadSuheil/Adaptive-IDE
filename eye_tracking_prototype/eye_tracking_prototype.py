import os
import cv2
import csv
import json
import uuid
import time
import yaml
import math
import argparse
import numpy as np
from datetime import datetime
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# -------------------------------------------------------------------------
# Core Config
# -------------------------------------------------------------------------
class Config:
    def __init__(self, path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(path):
            path = os.path.join(script_dir, path)
            
        with open(path, 'r') as f:
            self.data = yaml.safe_load(f)
        
        self.webcam_idx = self.data['webcam']['device_index']
        self.webcam_fps = self.data['webcam']['fps_target']
        self.webcam_w = self.data['webcam']['width']
        self.webcam_h = self.data['webcam']['height']
        
        self.grid_rows = self.data['grid']['rows']
        self.grid_cols = self.data['grid']['cols']
        self.section_map = self.data['grid']['section_map']
        self.section_colors = self.data['grid']['section_colors']
        
        self.filter_type = self.data['filter']['type']
        self.ema_alpha = self.data['filter']['ema_alpha']
        self.median_window = self.data['filter']['median_window']
        self.kalman_p_noise = self.data['filter'].get('kalman_process_noise', 0.1)
        self.kalman_m_noise = self.data['filter'].get('kalman_measurement_noise', 4.0)
        
        self.dwell_threshold = self.data['dwell']['threshold_ms']
        
        self.calib_samples = self.data['calibration']['n_samples_per_point']
        self.calib_delay_ms = self.data['calibration']['dot_display_time_ms']
        self.calib_radius = self.data['calibration']['dot_radius']
        self.calib_color = self.data['calibration']['dot_color_bgr']
        
        self.iris_baseline_frames = self.data['iris']['baseline_frames']
        
        self.session_dir = self.data['output']['session_dir']
        if not os.path.isabs(self.session_dir):
            self.session_dir = os.path.join(script_dir, self.session_dir)
            
        self.csv_buffer_size = self.data['output']['csv_buffer_size']
        self.save_video = self.data['output']['save_video']
        
        self.model_path = self.data['mediapipe']['model_path']
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(script_dir, self.model_path)

# -------------------------------------------------------------------------
# Gaze Filter
# -------------------------------------------------------------------------
class GazeFilter:
    def __init__(self, cfg):
        self.type = cfg.filter_type
        if self.type == "kalman" and not KALMAN_AVAILABLE:
            print("[WARN] filterpy not found. Falling back to EMA.")
            self.type = "ema"
            
        self.ema_alpha = cfg.ema_alpha
        self.median_window = cfg.median_window
        
        self.last_val = None
        self.history = deque(maxlen=self.median_window)
        
        if self.type == "kalman":
            self.kf = KalmanFilter(dim_x=4, dim_z=2) # state: [x, y, dx, dy]
            self.kf.x = np.array([0., 0., 0., 0.])
            self.kf.F = np.array([[1., 0., 1., 0.],
                                  [0., 1., 0., 1.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
            self.kf.H = np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.]])
            self.kf.P *= 1000.
            self.kf.R = np.eye(2) * cfg.kalman_m_noise
            self.kf.Q = np.eye(4) * cfg.kalman_p_noise

    def update(self, x, y):
        if self.type == "none":
            return x, y
            
        elif self.type == "ema":
            if self.last_val is None:
                self.last_val = (x, y)
            else:
                lx, ly = self.last_val
                a = self.ema_alpha
                self.last_val = (lx * (1-a) + x * a, ly * (1-a) + y * a)
            return self.last_val
            
        elif self.type == "median":
            self.history.append((x, y))
            xs = [v[0] for v in self.history]
            ys = [v[1] for v in self.history]
            return np.median(xs), np.median(ys)
            
        elif self.type == "kalman":
            if self.last_val is None:
                self.kf.x = np.array([x, y, 0., 0.])
                self.last_val = (x, y)
                return x, y
            
            self.kf.predict()
            self.kf.update(np.array([x, y]))
            self.last_val = (self.kf.x[0], self.kf.x[1])
            return self.last_val
            
        return x, y

    def reset(self):
        self.last_val = None
        self.history.clear()

# -------------------------------------------------------------------------
# Metrics Engine
# -------------------------------------------------------------------------
class MetricsEngine:
    def __init__(self, cfg):
        self.dwell_threshold = cfg.dwell_threshold / 1000.0 # to seconds
        self.iris_baseline_frames = cfg.iris_baseline_frames
        
        self.current_section = None
        self.section_enter_time = 0
        self.dwell_time_ms = 0
        
        self.nrevisit_counts = {}
        self.transition_history = deque() # tuples of (timestamp, section_name)
        
        self.iris_history = []
        self.iris_baseline = None
        self.iris_delta = 0.0
        
        self.session_sections_summary = {}

    def update(self, timestamp_ms, section, iris_size):
        ts_sec = timestamp_ms / 1000.0
        
        # 1. Update Iris Baseline & Delta
        if self.iris_baseline is None:
            self.iris_history.append(iris_size)
            if len(self.iris_history) >= self.iris_baseline_frames:
                self.iris_baseline = np.mean(self.iris_history)
                self.iris_delta = 0.0
        else:
            self.iris_delta = iris_size - self.iris_baseline
            
        # 2. Section Dwell & Transition
        if section != self.current_section:
            # We transitioned
            if self.current_section is not None:
                # Log stats for old section
                if self.current_section not in self.session_sections_summary:
                    self.session_sections_summary[self.current_section] = {
                        "total_dwell_ms": 0, "visit_count": 0, "nrevisit_count": 0, "max_continuous_dwell_ms": 0
                    }
                ss = self.session_sections_summary[self.current_section]
                ss["total_dwell_ms"] += self.dwell_time_ms
                ss["visit_count"] += 1
                if self.dwell_time_ms > ss["max_continuous_dwell_ms"]:
                    ss["max_continuous_dwell_ms"] = self.dwell_time_ms
            
            # Record transition
            if section is not None:
                self.transition_history.append((ts_sec, section))
                # Update NRevisit
                if section in self.nrevisit_counts:
                    self.nrevisit_counts[section] += 1
                else:
                    self.nrevisit_counts[section] = 0
                
            self.current_section = section
            self.section_enter_time = ts_sec
            self.dwell_time_ms = 0
        else:
            if section is not None:
                self.dwell_time_ms = int((ts_sec - self.section_enter_time) * 1000)
                
        # Clean up transition history (rolling 5s window)
        while self.transition_history and (ts_sec - self.transition_history[0][0]) > 5.0:
            self.transition_history.popleft()
            
    def get_transition_rate(self):
        # transitions per second over the last 5 seconds
        return len(self.transition_history) / 5.0

    def get_nrevisit(self, section):
        return self.nrevisit_counts.get(section, 0)

# -------------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------------
class EyeTrackerApp:
    def __init__(self, config_path):
        self.cfg = Config(config_path)
        
        # Setup paths
        os.makedirs(self.cfg.session_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}.csv")
        self.json_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}_summary.json")
        
        # Initialize OpenCV
        self.cap = cv2.VideoCapture(self.cfg.webcam_idx)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.webcam_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.webcam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.webcam_h)
        
        # Initialize MediaPipe
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
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ms", "frame_index", 
            "gaze_x_raw", "gaze_y_raw", "gaze_x_smooth", "gaze_y_smooth",
            "grid_row", "grid_col", "section", "confidence",
            "dwell_time_ms", "nrevisit_count", "transition_rate",
            "iris_size_delta", "fps_actual", "face_detected", "calibration_confidence"
        ])
        
        self.transform_matrix = None
        self.calibration_quality = 0.0
        
        self.frame_count = 0
        self.total_face_frames = 0
        self.start_time = time.time()
        self.is_paused = False
        self.show_grid = True
        self.debug_mode = False
        self._last_ts_ms = 0   # for monotonic timestamp tracking
        
        # Detect actual screen resolution (Windows)
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

    def get_iris_data(self, landmarks):
        # L: 468,469,470,471 | Center: 473
        # R: 473,474,475,476 | Center: 477 (Indices from MediaPipe docs, but let's be careful. 
        # Actually Left Iris center is 468, Right is 473 in Python tasks sometimes, let's just average all iris points to find center)
        
        # Left iris indices: 468-471 (ring), 473 (center) - wait, if 468 is ring...
        # Using 468 to 471 for left iris, 473 to 476 for right iris.
        # Let's average the ring to find center.
        left_iris_pts = [landmarks[i] for i in range(468, 472)]
        right_iris_pts = [landmarks[i] for i in range(473, 477)]
        
        lx = sum(p.x for p in left_iris_pts) / 4.0
        ly = sum(p.y for p in left_iris_pts) / 4.0
        
        rx = sum(p.x for p in right_iris_pts) / 4.0
        ry = sum(p.y for p in right_iris_pts) / 4.0
        
        # Iris size approx
        l_size = math.hypot(landmarks[468].x - landmarks[470].x, landmarks[468].y - landmarks[470].y)
        r_size = math.hypot(landmarks[473].x - landmarks[475].x, landmarks[473].y - landmarks[475].y)
        iris_size = (l_size + r_size) / 2.0
        
        # Average both eyes for gaze
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        
        return (cx, cy), iris_size

    def _next_ts(self):
        """Return a monotonically increasing timestamp in milliseconds."""
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def run_calibration(self):
        print("\n[EyeTrack] Starting Calibration Flow...")
        self.gaze_filter.reset()
        
        grid_r = self.cfg.grid_rows
        grid_c = self.cfg.grid_cols
        
        iris_pts = []
        screen_pts = []
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cell_w = self.screen_w / grid_c
        cell_h = self.screen_h / grid_r

        # Stability settings: require iris to stay within this normalised radius
        STABILITY_RADIUS = 0.008   # ~0.8% of frame width
        STABILITY_FRAMES = 8       # must be stable for 8 consecutive frames before counting
        
        for r in range(grid_r):
            for c in range(grid_c):
                dot_x = int((c + 0.5) * cell_w)
                dot_y = int((r + 0.5) * cell_h)
                dot_num = r * grid_c + c + 1
                total_dots = grid_r * grid_c
                
                # ── Phase 1: show dot + countdown ──────────────────────────────
                # Give user 2 seconds to move their gaze to the new dot
                MOVE_TIME = 2.0   # seconds
                move_start = time.time()
                while time.time() - move_start < MOVE_TIME:
                    elapsed = time.time() - move_start
                    remaining = MOVE_TIME - elapsed
                    bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    cv2.circle(bg, (dot_x, dot_y), self.cfg.calib_radius, self.cfg.calib_color, -1)
                    cv2.putText(bg, f"Move your eyes to the dot ({dot_num}/{total_dots})",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(bg, f"Starting in {remaining:.1f}s...",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                    cv2.imshow("Calibration", bg)
                    cv2.waitKey(30)
                
                # ── Phase 2: wait for stable gaze, then collect ────────────────
                samples_collected = 0
                collected_x, collected_y = [], []
                stable_streak = 0
                prev_ix, prev_iy = None, None
                
                while samples_collected < self.cfg.calib_samples:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = self._next_ts()
                    
                    try:
                        res = self.landmarker.detect_for_video(mp_img, ts_ms)
                        if not res.face_landmarks:
                            stable_streak = 0
                            cv2.waitKey(1)
                            continue
                        
                        (ix, iy), _ = self.get_iris_data(res.face_landmarks[0])
                        
                        # Stability gate
                        if prev_ix is not None:
                            drift = math.hypot(ix - prev_ix, iy - prev_iy)
                            if drift < STABILITY_RADIUS:
                                stable_streak += 1
                            else:
                                stable_streak = 0
                        prev_ix, prev_iy = ix, iy
                        
                        # Only collect if eye has been stable for enough frames
                        if stable_streak >= STABILITY_FRAMES:
                            collected_x.append(ix)
                            collected_y.append(iy)
                            samples_collected += 1
                        
                        # Progress feedback
                        copy_bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                        dot_color = (0, 255, 0) if stable_streak >= STABILITY_FRAMES else self.cfg.calib_color
                        cv2.circle(copy_bg, (dot_x, dot_y), self.cfg.calib_radius, dot_color, -1)
                        # Outer ring shrinks as samples are collected
                        ring_r = self.cfg.calib_radius + 20 - int(20 * samples_collected / self.cfg.calib_samples)
                        cv2.circle(copy_bg, (dot_x, dot_y), ring_r, (0, 200, 0), 2)
                        cv2.putText(copy_bg, f"HOLD STILL — {samples_collected}/{self.cfg.calib_samples} ({dot_num}/{total_dots})",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        if stable_streak < STABILITY_FRAMES:
                            cv2.putText(copy_bg, "Waiting for stable gaze...",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                        cv2.imshow("Calibration", copy_bg)
                        cv2.waitKey(1)
                        
                    except Exception as e:
                        stable_streak = 0
                
                avg_ix = np.mean(collected_x)
                avg_iy = np.mean(collected_y)
                iris_pts.append([avg_ix, avg_iy])
                screen_pts.append([dot_x, dot_y])
                
        cv2.destroyWindow("Calibration")
        
        # Compute affine transform
        iris_pts_arr = np.array(iris_pts, dtype=np.float32)
        screen_pts_arr = np.array(screen_pts, dtype=np.float32)
        
        M, inliers = cv2.estimateAffinePartial2D(iris_pts_arr, screen_pts_arr)
        if M is not None:
            self.transform_matrix = M
            self.calibration_quality = float(np.sum(inliers)) / len(inliers)
            print(f"[EyeTrack] Calibration complete! Quality: {self.calibration_quality:.2f}")
            if self.calibration_quality < 0.6:
                print("[EyeTrack] WARNING: Low quality (<0.6). Consider recalibrating with 'C'.")
        else:
            print("[EyeTrack] Calibration failed. Press 'C' to retry.")

    def apply_transform(self, ix, iy):
        if self.transform_matrix is None:
            return 0.0, 0.0
        src = np.array([[[ix, iy]]], dtype=np.float32)
        dst = cv2.transform(src, self.transform_matrix)
        return dst[0][0][0], dst[0][0][1]

    def get_grid_cell(self, screen_x, screen_y):
        col = int(screen_x / (self.screen_w / self.cfg.grid_cols))
        row = int(screen_y / (self.screen_h / self.cfg.grid_rows))
        
        col = max(0, min(col, self.cfg.grid_cols - 1))
        row = max(0, min(row, self.cfg.grid_rows - 1))
        
        section = self.cfg.section_map[row][col]
        
        cx = (col + 0.5) * (self.screen_w / self.cfg.grid_cols)
        cy = (row + 0.5) * (self.screen_h / self.cfg.grid_rows)
        dist = math.hypot(screen_x - cx, screen_y - cy)
        max_dist = math.hypot(self.screen_w / self.cfg.grid_cols / 2, self.screen_h / self.cfg.grid_rows / 2)
        conf = max(0.0, 1.0 - (dist / max_dist))
        
        return row, col, section, float(conf)

    def draw_grid_overlay(self, sx, sy, row, col, section):
        grid_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        
        cell_w = int(self.screen_w / self.cfg.grid_cols)
        cell_h = int(self.screen_h / self.cfg.grid_rows)
        
        # Fill active cell
        if section is not None:
            color = self.cfg.section_colors.get(section, [128, 128, 128])
            tl = (col * cell_w, row * cell_h)
            br = ((col + 1) * cell_w, (row + 1) * cell_h)
            cv2.rectangle(grid_img, tl, br, color, -1)
        
        # Draw lines and labels
        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                tl = (c * cell_w, r * cell_h)
                br = ((c + 1) * cell_w, (r + 1) * cell_h)
                cv2.rectangle(grid_img, tl, br, (255, 255, 255), 1)
                
                sec_lbl = self.cfg.section_map[r][c]
                if sec_lbl:
                    cv2.putText(grid_img, sec_lbl, (tl[0] + 10, tl[1] + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    
        # Draw gaze dot
        cv2.circle(grid_img, (int(sx), int(sy)), 15, (0, 0, 255), -1)
        
        # Controls
        cv2.putText(grid_img, "[Q] Quit  [C] Recalibrate  [S] Snapshot  [P] Pause  [G] Toggle Grid",
                    (20, self.screen_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
        cv2.imshow("Eye Tracking - Gaze Grid", grid_img)

    def run(self):
        self.run_calibration()
        
        cv2.namedWindow("Eye Tracking - Gaze Grid", cv2.WINDOW_NORMAL)
        print("[EyeTrack] Starting tracking loop...")
        
        while True:
            iter_start = time.time()
            ret, frame = self.cap.read()
            if not ret: break
            
            if self.is_paused:
                cv2.putText(frame, "PAUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Eye Tracking - Camera Feed", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p'): self.is_paused = False
                elif key == ord('q'): break
                continue
                
            self.frame_count += 1
            ts_ms = int(iter_start * 1000)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            face_detected = False
            raw_x, raw_y = 0.0, 0.0
            sm_x, sm_y = 0.0, 0.0
            grid_r, grid_c = -1, -1
            section = None
            conf = 0.0
            iris_size = 0.0
            
            res = None
            try:
                res = self.landmarker.detect_for_video(mp_img, self._next_ts())
            except Exception as e:
                print(f"[Warn] MediaPipe error: {e}")
                
            if res and res.face_landmarks:
                face_detected = True
                self.total_face_frames += 1
                
                lm = res.face_landmarks[0]
                (ix, iy), iris_size = self.get_iris_data(lm)
                
                # Draw mesh
                if self.debug_mode:
                    for p in lm:
                        px = int(p.x * frame.shape[1])
                        py = int(p.y * frame.shape[0])
                        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
                        
                cv2.circle(frame, (int(ix * frame.shape[1]), int(iy * frame.shape[0])), 4, (255, 255, 0), -1)
                
                # Transform and filter
                raw_x, raw_y = self.apply_transform(ix, iy)
                sm_x, sm_y = self.gaze_filter.update(raw_x, raw_y)
                
                grid_r, grid_c, section, conf = self.get_grid_cell(sm_x, sm_y)
                
            self.metrics.update(ts_ms, section, iris_size)
            
            # FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            
            # Record Data
            row = [
                ts_ms, self.frame_count,
                raw_x, raw_y, sm_x, sm_y,
                grid_r, grid_c, section if section else "", conf,
                self.metrics.dwell_time_ms,
                self.metrics.get_nrevisit(section),
                self.metrics.get_transition_rate(),
                self.metrics.iris_delta,
                fps, face_detected, self.calibration_quality
            ]
            self.csv_writer.writerow(row)
            
            if self.frame_count % self.cfg.csv_buffer_size == 0:
                self.csv_file.flush()
                
            # Draw HUD
            hud_y = 30
            hud_color = (0, 255, 0)
            lines = [
                f"FPS: {fps:.1f}",
                f"Section: {section}",
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
                
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.run_calibration()
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
        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.flush()
        self.csv_file.close()
        
        # Write JSON Summary
        duration = time.time() - self.start_time
        summary = {
            "session_id": self.session_id,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time_iso": datetime.now().isoformat(),
            "duration_seconds": duration,
            "config": {
                "grid_rows": self.cfg.grid_rows,
                "grid_cols": self.cfg.grid_cols,
                "filter_type": self.cfg.filter_type,
            },
            "summary": {
                "total_frames": self.frame_count,
                "frames_with_face": self.total_face_frames,
                "face_detection_rate": self.total_face_frames / self.frame_count if self.frame_count > 0 else 0,
                "avg_fps": self.frame_count / duration if duration > 0 else 0,
                "sections": self.metrics.session_sections_summary,
                "calibration_quality": self.calibration_quality
            },
            "csv_path": self.csv_path
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"[EyeTrack] Session saved to {self.csv_path} and {self.json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive IDE - Eye Tracking Prototype")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    app = EyeTrackerApp(args.config)
    app.run()
