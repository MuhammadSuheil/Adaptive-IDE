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
        self.flip_horizontal = self.data['webcam'].get('flip_horizontal', True)
        
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
        self.transition_window_sec = self.data.get('metrics', {}).get('transition_window_sec', 5.0)
        
        self.calib_samples = self.data['calibration']['n_samples_per_point']
        self.calib_move_delay_sec = self.data['calibration'].get('move_delay_sec', 2.0)
        self.calib_radius = self.data['calibration']['dot_radius']
        self.calib_color = self.data['calibration']['dot_color_bgr']
        self.calib_stability_thresh = self.data['calibration'].get('stability_threshold', 0.015)
        self.calib_stability_frames = self.data['calibration'].get('stability_required_frames', 6)
        self.calib_mapping_method = self.data['calibration'].get('mapping_method', 'polynomial')
        
        self.iris_baseline_frames = self.data['iris']['baseline_frames']
        
        self.session_dir = self.data['output']['session_dir']
        if not os.path.isabs(self.session_dir):
            self.session_dir = os.path.join(script_dir, self.session_dir)
            
        self.csv_buffer_size = self.data['output']['csv_buffer_size']
        self.save_video = self.data['output']['save_video']
        
        self.model_path = self.data['mediapipe']['model_path']
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(script_dir, self.model_path)
            
        lm_cfg = self.data['mediapipe'].get('landmarks', {})
        self.left_iris_indices = lm_cfg.get('left_iris', [468, 469, 470, 471, 472])
        self.right_iris_indices = lm_cfg.get('right_iris', [473, 474, 475, 476, 477])
        self.left_eye_corners = lm_cfg.get('left_eye_corners', [33, 133])
        self.right_eye_corners = lm_cfg.get('right_eye_corners', [362, 263])

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
            return float(np.median(xs)), float(np.median(ys))
            
        elif self.type == "kalman":
            if self.last_val is None:
                self.kf.x = np.array([x, y, 0., 0.])
                self.last_val = (x, y)
                return x, y
            
            self.kf.predict()
            self.kf.update(np.array([x, y]))
            self.last_val = (float(self.kf.x[0]), float(self.kf.x[1]))
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
        self.transition_window_sec = cfg.transition_window_sec
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
            if iris_size > 0:
                self.iris_history.append(iris_size)
                if len(self.iris_history) >= self.iris_baseline_frames:
                    self.iris_baseline = np.mean(self.iris_history)
                    self.iris_delta = 0.0
        else:
            if iris_size > 0:
                self.iris_delta = iris_size - self.iris_baseline
            
        # 2. Section Dwell & Transition
        if section != self.current_section:
            if self.current_section is not None:
                if self.current_section not in self.session_sections_summary:
                    self.session_sections_summary[self.current_section] = {
                        "total_dwell_ms": 0, "visit_count": 0, "nrevisit_count": 0, "max_continuous_dwell_ms": 0
                    }
                ss = self.session_sections_summary[self.current_section]
                ss["total_dwell_ms"] += self.dwell_time_ms
                ss["visit_count"] += 1
                if self.dwell_time_ms > ss["max_continuous_dwell_ms"]:
                    ss["max_continuous_dwell_ms"] = self.dwell_time_ms
            
            if section is not None:
                self.transition_history.append((ts_sec, section))
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
                
        # Clean up transition history (rolling window)
        while self.transition_history and (ts_sec - self.transition_history[0][0]) > self.transition_window_sec:
            self.transition_history.popleft()
            
    def get_transition_rate(self):
        return len(self.transition_history) / self.transition_window_sec

    def get_nrevisit(self, section):
        return self.nrevisit_counts.get(section, 0)

# -------------------------------------------------------------------------
# Mapper Models (Polynomial & Affine)
# -------------------------------------------------------------------------
class GazeMapper:
    def __init__(self, method="polynomial"):
        self.method = method
        self.model_x = None
        self.model_y = None
        self.affine_M = None

    def fit(self, iris_pts, screen_pts):
        """
        iris_pts: Nx2 array of normalized eye features [norm_x, norm_y]
        screen_pts: Nx2 array of target screen coordinates [x, y]
        """
        iris_pts = np.array(iris_pts, dtype=np.float32)
        screen_pts = np.array(screen_pts, dtype=np.float32)

        if self.method == "polynomial":
            # 2nd degree polynomial terms: [1, x, y, x^2, y^2, x*y]
            X = np.column_stack([
                np.ones(len(iris_pts)),
                iris_pts[:, 0],
                iris_pts[:, 1],
                iris_pts[:, 0]**2,
                iris_pts[:, 1]**2,
                iris_pts[:, 0] * iris_pts[:, 1]
            ])
            # Ridge regression pseudo-inverse for stability
            ridge = 1e-4 * np.eye(X.shape[1])
            self.model_x, _, _, _ = np.linalg.lstsq(X.T @ X + ridge, X.T @ screen_pts[:, 0], rcond=None)
            self.model_y, _, _, _ = np.linalg.lstsq(X.T @ X + ridge, X.T @ screen_pts[:, 1], rcond=None)
            
            pred_x = X @ self.model_x
            pred_y = X @ self.model_y
            errors = np.hypot(pred_x - screen_pts[:, 0], pred_y - screen_pts[:, 1])
            quality = max(0.0, 1.0 - (np.mean(errors) / 300.0))
            return quality

        elif self.method == "homography":
            H, mask = cv2.findHomography(iris_pts, screen_pts, cv2.RANSAC, 5.0)
            self.affine_M = H
            return float(np.sum(mask)) / len(mask) if mask is not None else 0.0

        else: # "affine"
            M, inliers = cv2.estimateAffinePartial2D(iris_pts, screen_pts)
            self.affine_M = M
            return float(np.sum(inliers)) / len(inliers) if inliers is not None else 0.0

    def predict(self, norm_x, norm_y):
        if self.method == "polynomial" and self.model_x is not None:
            feat = np.array([1.0, norm_x, norm_y, norm_x**2, norm_y**2, norm_x * norm_y])
            sx = np.dot(feat, self.model_x)
            sy = np.dot(feat, self.model_y)
            return float(sx), float(sy)

        elif self.method == "homography" and self.affine_M is not None:
            pt = np.array([norm_x, norm_y, 1.0], dtype=np.float32)
            dst = self.affine_M @ pt
            if dst[2] != 0:
                return float(dst[0] / dst[2]), float(dst[1] / dst[2])

        elif self.affine_M is not None:
            src = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            dst = cv2.transform(src, self.affine_M)
            return float(dst[0][0][0]), float(dst[0][0][1])

        return 0.0, 0.0

# -------------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------------
class EyeTrackerApp:
    def __init__(self, config_path):
        self.cfg = Config(config_path)
        
        os.makedirs(self.cfg.session_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}.csv")
        self.json_path = os.path.join(self.cfg.session_dir, f"session_{self.session_id}_{timestamp}_summary.json")
        
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
            "iris_size_delta", "fps_actual", "face_detected", "calibration_quality"
        ])
        
        self.calibration_quality = 0.0
        self.frame_count = 0
        self.total_face_frames = 0
        self.start_time = time.time()
        self.is_paused = False
        self.show_grid = True
        self.debug_mode = False
        self._last_ts_ms = 0
        
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

    def get_normalized_eye_vector(self, landmarks):
        """
        Extracts iris positions normalized relative to eye corners (Head-movement invariant).
        Returns: (norm_x, norm_y), iris_size
        """
        # Left eye landmarks
        left_corner_outer = landmarks[self.cfg.left_eye_corners[0]]
        left_corner_inner = landmarks[self.cfg.left_eye_corners[1]]
        left_iris_pts = [landmarks[i] for i in self.cfg.left_iris_indices if i < len(landmarks)]
        
        # Right eye landmarks
        right_corner_inner = landmarks[self.cfg.right_eye_corners[0]]
        right_corner_outer = landmarks[self.cfg.right_eye_corners[1]]
        right_iris_pts = [landmarks[i] for i in self.cfg.right_iris_indices if i < len(landmarks)]

        if not left_iris_pts or not right_iris_pts:
            return (0.5, 0.5), 0.0

        # Centers
        l_iris_x = sum(p.x for p in left_iris_pts) / len(left_iris_pts)
        l_iris_y = sum(p.y for p in left_iris_pts) / len(left_iris_pts)
        
        r_iris_x = sum(p.x for p in right_iris_pts) / len(right_iris_pts)
        r_iris_y = sum(p.y for p in right_iris_pts) / len(right_iris_pts)

        # Normalize Left Eye (x: 0 outer to 1 inner)
        l_eye_w = math.hypot(left_corner_inner.x - left_corner_outer.x, left_corner_inner.y - left_corner_outer.y)
        l_norm_x = (l_iris_x - left_corner_outer.x) / l_eye_w if l_eye_w > 0 else 0.5
        l_norm_y = (l_iris_y - left_corner_outer.y) / l_eye_w if l_eye_w > 0 else 0.5

        # Normalize Right Eye (x: 0 inner to 1 outer)
        r_eye_w = math.hypot(right_corner_outer.x - right_corner_inner.x, right_corner_outer.y - right_corner_inner.y)
        r_norm_x = (r_iris_x - right_corner_inner.x) / r_eye_w if r_eye_w > 0 else 0.5
        r_norm_y = (r_iris_y - right_corner_inner.y) / r_eye_w if r_eye_w > 0 else 0.5

        norm_x = (l_norm_x + r_norm_x) / 2.0
        norm_y = (l_norm_y + r_norm_y) / 2.0

        # Iris size relative to face width
        l_size = math.hypot(left_iris_pts[0].x - left_iris_pts[2].x, left_iris_pts[0].y - left_iris_pts[2].y) if len(left_iris_pts) >= 3 else 0.01
        r_size = math.hypot(right_iris_pts[0].x - right_iris_pts[2].x, right_iris_pts[0].y - right_iris_pts[2].y) if len(right_iris_pts) >= 3 else 0.01
        iris_size = (l_size + r_size) / 2.0

        return (norm_x, norm_y), iris_size

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
        
        iris_pts = []
        screen_pts = []
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cell_w = self.screen_w / grid_c
        cell_h = self.screen_h / grid_r

        MOVE_TIME = self.cfg.calib_move_delay_sec
        
        for r in range(grid_r):
            for c in range(grid_c):
                dot_x = int((c + 0.5) * cell_w)
                dot_y = int((r + 0.5) * cell_h)
                dot_num = r * grid_c + c + 1
                total_dots = grid_r * grid_c
                
                # Phase 1: Countdown to move eyes
                move_start = time.time()
                while time.time() - move_start < MOVE_TIME:
                    elapsed = time.time() - move_start
                    remaining = MOVE_TIME - elapsed
                    bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    cv2.circle(bg, (dot_x, dot_y), self.cfg.calib_radius, self.cfg.calib_color, -1)
                    cv2.putText(bg, f"Focus your eyes on the RED DOT ({dot_num}/{total_dots})",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(bg, f"Hold gaze still in {remaining:.1f}s...",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                    cv2.imshow("Calibration", bg)
                    cv2.waitKey(30)
                
                # Phase 2: Collect samples with stability gate
                samples_collected = 0
                collected_x, collected_y = [], []
                stable_streak = 0
                prev_ix, prev_iy = None, None
                
                while samples_collected < self.cfg.calib_samples:
                    ret, frame = self.cap.read()
                    if not ret: continue
                    
                    if self.cfg.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = self._next_ts()
                    
                    try:
                        res = self.landmarker.detect_for_video(mp_img, ts_ms)
                        if not res.face_landmarks:
                            stable_streak = 0
                            cv2.waitKey(1)
                            continue
                        
                        (ix, iy), _ = self.get_normalized_eye_vector(res.face_landmarks[0])
                        
                        if prev_ix is not None:
                            drift = math.hypot(ix - prev_ix, iy - prev_iy)
                            if drift < self.cfg.calib_stability_thresh:
                                stable_streak += 1
                            else:
                                stable_streak = 0
                        prev_ix, prev_iy = ix, iy
                        
                        if stable_streak >= self.cfg.calib_stability_frames:
                            collected_x.append(ix)
                            collected_y.append(iy)
                            samples_collected += 1
                        
                        copy_bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
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
                        cv2.waitKey(1)
                        
                    except Exception as e:
                        stable_streak = 0
                
                avg_ix = float(np.mean(collected_x))
                avg_iy = float(np.mean(collected_y))
                iris_pts.append([avg_ix, avg_iy])
                screen_pts.append([dot_x, dot_y])
                
        cv2.destroyWindow("Calibration")
        
        # Train Mapper
        self.calibration_quality = self.mapper.fit(iris_pts, screen_pts)
        print(f"[EyeTrack] Calibration complete! Quality ({self.mapper.method}): {self.calibration_quality:.2f}")
        if self.calibration_quality < 0.6:
            print("[EyeTrack] WARNING: Low quality (<0.6). Consider recalibrating with 'C'.")

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
                    
        cv2.circle(grid_img, (int(sx), int(sy)), 15, (0, 0, 255), -1)
        
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
            
            if self.cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
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
                (norm_x, norm_y), iris_size = self.get_normalized_eye_vector(lm)
                
                if self.debug_mode:
                    for p in lm:
                        px = int(p.x * frame.shape[1])
                        py = int(p.y * frame.shape[0])
                        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
                
                # Transform via mapper model & filter
                raw_x, raw_y = self.mapper.predict(norm_x, norm_y)
                sm_x, sm_y = self.gaze_filter.update(raw_x, raw_y)
                
                grid_r, grid_c, section, conf = self.get_grid_cell(sm_x, sm_y)
                
            self.metrics.update(ts_ms, section, iris_size)
            
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            
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
                "mapping_method": self.cfg.calib_mapping_method,
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
