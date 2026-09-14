import os
import cv2
import csv
import json
import uuid
import time
import math
import threading
import numpy as np
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from modules import Config, WebcamStream, GazeFilter, MetricsEngine, GazeMapper, BlinkDetector


def estimate_head_pose(landmarks, image_w, image_h):
    """
    Estimates head pose (pitch, yaw, roll) in degrees and face position using cv2.solvePnP
    on 6 key MediaPipe facial landmarks:
      - 1: Nose tip
      - 152: Chin
      - 33: Left eye left corner
      - 263: Right eye right corner
      - 61: Left mouth corner
      - 291: Right mouth corner
    Returns: pitch_deg, yaw_deg, roll_deg, face_width_ratio, center_x_ratio, center_y_ratio
    """
    if not landmarks or len(landmarks) < 292:
        return 0.0, 0.0, 0.0, 0.0, 0.5, 0.5

    # 3D Reference Model Points (in arbitrary unit space)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -330.0, -65.0),     # Chin (152)
        (-225.0, 170.0, -135.0),  # Left eye left corner (33)
        (225.0, 170.0, -135.0),   # Right eye right corner (263)
        (-150.0, -150.0, -125.0), # Left mouth corner (61)
        (150.0, -150.0, -125.0)   # Right mouth corner (291)
    ], dtype=np.float64)

    target_indices = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        (landmarks[idx].x * image_w, landmarks[idx].y * image_h)
        for idx in target_indices
    ], dtype=np.float64)

    focal_length = image_w
    center = (image_w / 2.0, image_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0, 0.0, 0.5, 0.5

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    # Fix 180-degree decomposition wrap-around from OpenCV solvePnP
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0

    if yaw > 90.0:
        yaw -= 180.0
    elif yaw < -90.0:
        yaw += 180.0

    # Face geometry metrics
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    face_width_ratio = math.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
    
    nose = landmarks[1]
    chin = landmarks[152]
    center_x_ratio = (left_eye.x + right_eye.x + nose.x + chin.x) / 4.0
    center_y_ratio = (left_eye.y + right_eye.y + nose.y + chin.y) / 4.0

    return float(pitch), float(yaw), float(roll), float(face_width_ratio), float(center_x_ratio), float(center_y_ratio)


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
        self.blink_detector = BlinkDetector(self.cfg)
        self.mapper = GazeMapper(
            method=self.cfg.calib_mapping_method,
            rbf_kernel=self.cfg.calib_rbf_kernel,
            rbf_smoothing=self.cfg.calib_rbf_smoothing,
            output_clamp=self.cfg.calib_output_clamp
        )
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ms", "frame_index", "capture_frame_id",
            "gaze_x_raw", "gaze_y_raw", "gaze_x_smooth", "gaze_y_smooth",
            "grid_row", "grid_col", "section", "confidence",
            "dwell_time_ms", "nrevisit_count", "transition_rate",
            "iris_size_delta", "fps_actual", "capture_fps", "capture_age_ms",
            "preprocess_ms", "inference_ms", "mapping_metrics_ms", "total_processing_ms",
            "dropped_frames_total",
            "face_detected", "gaze_status", "calibration_quality",
            "head_pitch", "head_yaw", "head_pose_shifted",
            "is_blinking", "total_blinks", "blink_rate_bpm"
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
        self.tracking_stop = threading.Event()
        self.result_lock = threading.Lock()
        self.tracking_thread = None
        self.latest_result = None
        self.dropped_frames = 0
        self.stage_timings = {
            "capture_age_ms": [], "preprocess_ms": [], "inference_ms": [],
            "mapping_metrics_ms": [], "logging_ms": [], "total_processing_ms": []
        }
        self.ui_frame_count = 0
        self.ui_started = None
        self._sync_frame_id = 0
        self.processing_started_at = None
        self.processing_last_at = None
        self.grid_templates = {}

        # Baseline head pose captured during Phase 0 gate
        self.baseline_pose = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "face_width_ratio": 0.35,
            "center_x_ratio": 0.5,
            "center_y_ratio": 0.45
        }
        
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
        """Returns ((norm_x, norm_y, ear), iris_size).
        Glasses robustness: each eye is weighted by its own iris consistency
        (lower variance = more reliable). This prevents glare on one lens
        from pulling the averaged gaze position off-centre.
        """
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

        # Iris spread variance — larger spread → less reliable (likely glare)
        l_var = sum(math.hypot(p.x - l_iris_x, p.y - l_iris_y) for p in left_iris_pts) / len(left_iris_pts)
        r_var = sum(math.hypot(p.x - r_iris_x, p.y - r_iris_y) for p in right_iris_pts) / len(right_iris_pts)
        # Inverse-variance weights (lower spread → higher weight)
        l_w = 1.0 / max(l_var, 1e-6)
        r_w = 1.0 / max(r_var, 1e-6)
        total_w = l_w + r_w
        l_w /= total_w
        r_w /= total_w

        def eye_local(iris_x, iris_y, corner_a, corner_b):
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

        l_norm_x, l_norm_y = eye_local(l_iris_x, l_iris_y, left_corner_outer, left_corner_inner)
        r_norm_x, r_norm_y = eye_local(r_iris_x, r_iris_y, right_corner_inner, right_corner_outer)

        # Weighted average instead of simple 50/50
        norm_x = l_norm_x * l_w + r_norm_x * r_w
        norm_y = l_norm_y * l_w + r_norm_y * r_w

        ear = self.calculate_ear(landmarks)

        l_size = math.hypot(left_iris_pts[0].x - left_iris_pts[2].x,
                            left_iris_pts[0].y - left_iris_pts[2].y) if len(left_iris_pts) >= 3 else 0.01
        r_size = math.hypot(right_iris_pts[0].x - right_iris_pts[2].x,
                            right_iris_pts[0].y - right_iris_pts[2].y) if len(right_iris_pts) >= 3 else 0.01
        iris_size = (l_size + r_size) / 2.0

        return (norm_x, norm_y, ear), iris_size

    def _next_ts(self):
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    # ─────────────────────────────────────────────
    # Phase 0: Head Positioning & Alignment Gate
    # ─────────────────────────────────────────────
    def run_head_positioning_gate(self):
        if not self.cfg.hp_enabled:
            return True

        print("\n[EyeTrack] Launching Phase 0: Head Positioning Gate...")
        cv2.namedWindow("Head Alignment Gate", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Head Alignment Gate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        stable_count = 0
        countdown_start = None
        frame_id = 0

        target_cx = int(self.screen_w * self.cfg.hp_target_center_x_ratio)
        target_cy = self.screen_h // 2
        target_w = int(self.screen_w * self.cfg.hp_target_face_width_ratio)
        target_h = int(target_w * 1.3)

        # Checklist item labels (in order)
        checklist_labels = [
            "Face Detected",
            "Distance OK",
            "Centered",
            "Head Angle OK",
            "Stable"
        ]

        def draw_checklist(canvas, states):
            """Draw segmented readiness bar at the bottom of the screen."""
            bar_x = 60
            bar_y = self.screen_h - 110
            item_w = (self.screen_w - 120) // len(checklist_labels)
            for i, (label, done) in enumerate(zip(checklist_labels, states)):
                x1 = bar_x + i * item_w
                y1 = bar_y
                x2 = x1 + item_w - 6
                y2 = bar_y + 54
                color = (0, 200, 80) if done else (60, 60, 60)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
                tick = "[OK] " if done else "[ ]  "
                cv2.putText(canvas, tick + label, (x1 + 8, y1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255) if done else (130, 130, 130), 1)

        def draw_distance_indicator(canvas, face_w_ratio):
            """Show MOVE CLOSER / MOVE BACK text with arrow near target box."""
            diff = face_w_ratio - self.cfg.hp_target_face_width_ratio
            tol = self.cfg.hp_size_tolerance_ratio * self.cfg.hp_target_face_width_ratio
            if diff < -tol:
                text = "v  MOVE CLOSER  v"
                color = (0, 165, 255)
            elif diff > tol:
                text = "^  MOVE BACK  ^"
                color = (0, 165, 255)
            else:
                text = "Distance OK"
                color = (0, 200, 80)
            cv2.putText(canvas, text,
                        (target_cx - 140, target_cy + target_h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        while not self.exit_requested:
            if self.stream is not None:
                ret, frame, new_id, _ = self.stream.read_latest(frame_id, timeout=0.2)
                if ret: frame_id = new_id
            else:
                ret, frame = self.read_frame()

            if not ret or frame is None:
                stable_count = 0
                countdown_start = None
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.exit_requested = True
                    cv2.destroyWindow("Head Alignment Gate")
                    return False
                continue

            mp_img = self.make_mediapipe_image(frame)
            try:
                res = self.landmarker.detect_for_video(mp_img, self._next_ts())
            except Exception as exc:
                print(f"[EyeTrack] Face guide inference failed: {exc}")
                res = None

            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Aspect-preserving live camera beneath the central circular guide.
            fh, fw = frame.shape[:2]
            scale = min(self.screen_w / fw, self.screen_h / fh)
            pw, ph = int(fw * scale), int(fh * scale)
            ox, oy = (self.screen_w - pw) // 2, (self.screen_h - ph) // 2
            canvas[oy:oy+ph, ox:ox+pw] = cv2.resize(frame, (pw, ph))
            radius = int(min(pw, ph) * 0.32)
            target_cx, target_cy = self.screen_w // 2, self.screen_h // 2
            target_w = target_h = radius * 2

            # Draw target bounding box
            box_left = target_cx - target_w // 2
            box_right = target_cx + target_w // 2
            box_top = target_cy - target_h // 2
            box_bottom = target_cy + target_h // 2

            # Checklist state
            ck_face = False
            ck_dist = False
            ck_center = False
            ck_angle = False
            is_aligned = False
            face_w_ratio_live = 0.0

            status_msg = "POSITION YOUR FACE IN THE CIRCLE"
            msg_color = (0, 165, 255)

            if res and res.face_landmarks:
                lm = res.face_landmarks[0]
                pitch, yaw, roll, face_w_ratio, cx_ratio, cy_ratio = estimate_head_pose(
                    lm, frame.shape[1], frame.shape[0]
                )
                face_w_ratio_live = face_w_ratio

                face_cx = int(self.screen_w * cx_ratio)
                face_cy = int(self.screen_h * cy_ratio)

                self.baseline_pose = {
                    "pitch": pitch, "yaw": yaw, "roll": roll,
                    "face_width_ratio": face_w_ratio,
                    "center_x_ratio": cx_ratio, "center_y_ratio": cy_ratio
                }

                dx = abs(cx_ratio - self.cfg.hp_target_center_x_ratio)
                dy = abs(cy_ratio - self.cfg.hp_target_center_y_ratio)
                dist_diff = abs(face_w_ratio - self.cfg.hp_target_face_width_ratio)
                dist_tol = self.cfg.hp_size_tolerance_ratio * self.cfg.hp_target_face_width_ratio

                ck_face = True
                anchors = np.array([[ox + lm[i].x * pw, oy + lm[i].y * ph]
                                    for i in (10, 152, 234, 454)])
                guide_center = np.array([target_cx, target_cy])
                distances = np.linalg.norm(anchors - guide_center, axis=1) / radius
                ck_dist = bool(np.all(distances <= 1.05) and
                               np.all(distances[:2] >= 0.75) and
                               np.all(distances[2:] >= 0.40))
                ck_center = np.linalg.norm(anchors.mean(axis=0)-guide_center) <= radius * 0.18
                for anchor in anchors:
                    cv2.circle(canvas, tuple(anchor.astype(int)), 5, (0,255,255), -1)
                ck_angle = (abs(yaw) <= self.cfg.hp_max_yaw_deg and
                            abs(pitch) <= self.cfg.hp_max_pitch_deg and abs(roll) <= 10.0)

                # Face marker
                # Guide checks use the same transformed pixels as the live preview.

                if not ck_dist:
                    status_msg = ("MOVE BACK - KEEP FACE INSIDE CIRCLE" if np.any(distances > 1.05)
                                  else "MOVE CLOSER - FILL CIRCLE WITH YOUR FACE")
                    msg_color = (0, 165, 255)
                elif not ck_center:
                    status_msg = "CENTER FOREHEAD, CHIN AND CHEEKS IN THE CIRCLE"
                    msg_color = (0, 165, 255)
                elif not ck_angle:
                    status_msg = f"LOOK STRAIGHT AT SCREEN  (Yaw {yaw:.1f}°  Pitch {pitch:.1f}°)"
                    msg_color = (0, 100, 255)
                else:
                    is_aligned = True
                    status_msg = "POSTURE CONFIRMED — HOLD STILL!"
                    msg_color = (0, 255, 0)

                # Distance is evaluated against the circle rather than a separate width target.

            checklist_states = [ck_face, ck_dist, ck_center, ck_angle, is_aligned]

            if is_aligned:
                stable_count += 1
            else:
                stable_count = 0
                countdown_start = None

            # Final checklist "Stable" state
            checklist_states[4] = stable_count >= self.cfg.hp_stability_frames_required

            box_color = (0, 255, 0) if is_aligned else msg_color
            cv2.circle(canvas, (target_cx, target_cy), radius, box_color, 4)
            cv2.line(canvas, (target_cx - 20, target_cy), (target_cx + 20, target_cy), box_color, 2)
            cv2.line(canvas, (target_cx, target_cy - 20), (target_cx, target_cy + 20), box_color, 2)

            # UI Header
            cv2.putText(canvas, "PHASE 0: HEAD POSITIONING & ALIGNMENT GATE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            cv2.putText(canvas, status_msg, (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)
            cv2.putText(canvas,
                        "Fill the circle with your face and hold for 5 seconds. [Q] Quit",
                        (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

            # Countdown bar
            if stable_count >= self.cfg.hp_stability_frames_required:
                if countdown_start is None:
                    countdown_start = time.perf_counter()
                elapsed = time.perf_counter() - countdown_start
                remaining = self.cfg.hp_countdown_seconds - elapsed
                cv2.ellipse(canvas, (target_cx, target_cy), (radius+10, radius+10),
                            -90, 0, 360 * min(elapsed / max(self.cfg.hp_countdown_seconds, 0.1), 1),
                            (0,255,0), 6)
                if remaining <= 0:
                    cv2.destroyWindow("Head Alignment Gate")
                    print(f"[EyeTrack] Gate passed! Baseline pose locked: {self.baseline_pose}")
                    return True
                cv2.putText(canvas, f"STARTING IN {math.ceil(remaining)}s...",
                            (target_cx - 180, box_bottom + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            draw_checklist(canvas, checklist_states)

            cv2.imshow("Head Alignment Gate", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.exit_requested = True
                cv2.destroyWindow("Head Alignment Gate")
                return False

        cv2.destroyWindow("Head Alignment Gate")
        return False

    # ─────────────────────────────────────────────
    # Phase 1: Modular N-Point Calibration Engine
    # ─────────────────────────────────────────────
    def generate_calibration_points(self):
        margin = min(max(self.cfg.calib_target_margin, 0.0), 0.30)
        min_x = self.screen_w * margin
        max_x = (self.screen_w - 1) * (1.0 - margin)
        min_y = self.screen_h * margin
        max_y = (self.screen_h - 1) * (1.0 - margin)

        mode = self.cfg.calib_layout_mode

        if mode == "3x3":
            xs = np.linspace(min_x, max_x, 3)
            ys = np.linspace(min_y, max_y, 3)
            points = []
            for r in range(3):
                for c in range(3):
                    points.append((int(xs[c]), int(ys[r])))
            return points

        elif mode == "4x4":
            xs = np.linspace(min_x, max_x, 4)
            ys = np.linspace(min_y, max_y, 4)
            points = []
            for r in range(4):
                for c in range(4):
                    points.append((int(xs[c]), int(ys[r])))
            return points

        elif mode == "13_point":
            # Standard 13-point calibration layout (3x3 outer/mid grid + 4 inner quadrant points)
            xs_3 = np.linspace(min_x, max_x, 3)
            ys_3 = np.linspace(min_y, max_y, 3)
            pts_3x3 = [(int(x), int(y)) for y in ys_3 for x in xs_3]

            mid_left_x = int(self.screen_w * 0.25)
            mid_right_x = int(self.screen_w * 0.75)
            mid_top_y = int(self.screen_h * 0.25)
            mid_bot_y = int(self.screen_h * 0.75)

            inner_4 = [
                (mid_left_x, mid_top_y),
                (mid_right_x, mid_top_y),
                (mid_left_x, mid_bot_y),
                (mid_right_x, mid_bot_y)
            ]
            return pts_3x3 + inner_4

        else:
            # Fallback 3x3
            xs = np.linspace(min_x, max_x, 3)
            ys = np.linspace(min_y, max_y, 3)
            return [(int(x), int(y)) for y in ys for x in xs]

    def run_calibration(self):
        print(f"\n[EyeTrack] Starting Phase 1 Fast Calibration ({self.cfg.calib_layout_mode})...")
        self.gaze_filter.reset()
        
        points = self.generate_calibration_points()
        total_dots = len(points)
        
        eye_features = []
        screen_pts = []
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        MOVE_TIME = self.cfg.calib_move_delay_sec
        calibration_frame_id = 0
        calibration_unique_frames = 0
        
        for idx, (dot_x, dot_y) in enumerate(points):
            dot_num = idx + 1
            
            move_start = time.time()
            while time.time() - move_start < MOVE_TIME:
                elapsed = time.time() - move_start
                remaining = MOVE_TIME - elapsed
                bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

                # Pulse animation: dot radius oscillates during move delay
                if self.cfg.calib_dot_pulse:
                    pulse_t = (elapsed % 0.5) / 0.5  # 0->1 twice per second
                    pulse_r = int(self.cfg.calib_radius * (0.6 + 0.4 * abs(math.sin(pulse_t * math.pi))))
                else:
                    pulse_r = self.cfg.calib_radius

                cv2.circle(bg, (dot_x, dot_y), pulse_r, self.cfg.calib_color, -1)
                cv2.putText(bg, f"LOOK AT THE RED DOT ({dot_num}/{total_dots})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(bg, f"Move eyes only. Recording in {remaining:.1f}s...",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                cv2.imshow("Calibration", bg)
                cv2.waitKey(30)
            
            samples_collected = 0
            collected_feats = []
            stable_streak = 0
            prev_ix, prev_iy = None, None
            point_started = time.time()
            head_warning_text = ""
            
            while samples_collected < self.cfg.calib_samples:
                if time.time() - point_started > self.cfg.calib_point_timeout_sec:
                    break
                if self.stream is not None:
                    ret, frame, new_frame_id, _ = self.stream.read_latest(
                        calibration_frame_id, timeout=0.2
                    )
                    if ret:
                        calibration_frame_id = new_frame_id
                else:
                    ret, frame = self.read_frame()
                if not ret or frame is None: continue
                calibration_unique_frames += 1
                
                mp_img = self.make_mediapipe_image(frame)
                ts_ms = self._next_ts()
                
                try:
                    res = self.landmarker.detect_for_video(mp_img, ts_ms)
                    if not res.face_landmarks:
                        stable_streak = 0
                        cv2.waitKey(1)
                        continue

                    lm = res.face_landmarks[0]

                    # Phase 3: Check Head Pose during calibration
                    pitch, yaw, roll, _, _, _ = estimate_head_pose(lm, frame.shape[1], frame.shape[0])
                    yaw_drift = abs(yaw - self.baseline_pose["yaw"])
                    pitch_drift = abs(pitch - self.baseline_pose["pitch"])

                    if yaw_drift > self.cfg.pose_max_calib_yaw_deg or pitch_drift > self.cfg.pose_max_calib_pitch_deg:
                        stable_streak = 0
                        head_warning_text = f"⚠ HEAD TURNED ({yaw_drift:.1f}° yaw)  LOOK STRAIGHT"
                        # Non-blocking slim HUD bar — keeps dot visible, doesn't block the point
                        copy_bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                        cv2.circle(copy_bg, (dot_x, dot_y), self.cfg.calib_radius, (0, 165, 255), -1)
                        cv2.putText(copy_bg, f"RECORDING GAZE — {samples_collected}/{self.cfg.calib_samples} ({dot_num}/{total_dots})",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        # Slim warning bar at top
                        cv2.rectangle(copy_bg, (0, 0), (self.screen_w, 36), (0, 60, 180), -1)
                        cv2.putText(copy_bg, head_warning_text,
                                    (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.imshow("Calibration", copy_bg)
                        cv2.waitKey(1)
                        continue
                    else:
                        head_warning_text = ""

                    (ix, iy, ear), _ = self.get_normalized_eye_vector(lm)
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

                    is_locked = stable_streak >= self.cfg.calib_stability_frames
                    dot_color = (0, 255, 0) if is_locked else self.cfg.calib_color
                    cv2.circle(copy_bg, (dot_x, dot_y), self.cfg.calib_radius, dot_color, -1)

                    # Crosshair when stable
                    if is_locked and self.cfg.calib_dot_crosshair:
                        r = self.cfg.calib_radius
                        cv2.circle(copy_bg, (dot_x, dot_y), r // 3, (255, 255, 255), -1)
                        cv2.line(copy_bg, (dot_x - r, dot_y), (dot_x + r, dot_y), (255, 255, 255), 1)
                        cv2.line(copy_bg, (dot_x, dot_y - r), (dot_x, dot_y + r), (255, 255, 255), 1)

                    # Progress arc (sweeping from top, clockwise)
                    if samples_collected > 0:
                        arc_angle = int(360 * samples_collected / self.cfg.calib_samples)
                        arc_r = self.cfg.calib_radius + 14
                        cv2.ellipse(copy_bg, (dot_x, dot_y), (arc_r, arc_r),
                                    -90, 0, arc_angle, (0, 220, 100), 4)

                    # Progress ring (shrinks as samples fill)
                    ring_r = self.cfg.calib_radius + 24 - int(24 * samples_collected / self.cfg.calib_samples)
                    cv2.circle(copy_bg, (dot_x, dot_y), ring_r, (0, 200, 0), 1)

                    cv2.putText(copy_bg, f"RECORDING GAZE — {samples_collected}/{self.cfg.calib_samples} ({dot_num}/{total_dots})",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                    # Slim head pose HUD bar at top (always visible, non-blocking)
                    if head_warning_text:
                        cv2.rectangle(copy_bg, (0, 0), (self.screen_w, 36), (0, 60, 180), -1)
                        cv2.putText(copy_bg, head_warning_text,
                                    (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    elif not is_locked:
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
            if len(kept) < max(5, self.cfg.calib_samples // 3):
                kept = samples
            point_std = float(np.max(np.std(kept[:, :2], axis=0)))
            if point_std > self.cfg.calib_max_sample_std:
                print(f"[EyeTrack] Warning: Calibration point {dot_num} was noisy (std={point_std:.4f}).")

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
            **self.mapper.diagnostics, "feature_span_x": span_x, "feature_span_y": span_y,
            "unique_camera_frames": calibration_unique_frames,
        }
        print(f"[EyeTrack] Calibration complete! Quality ({self.mapper.method}): {self.calibration_quality:.2f}")
        print(f"[EyeTrack] Validation: {self.mapper.diagnostics}")
        if self.calibration_quality < self.cfg.calib_min_quality:
            print(f"[EyeTrack] Quality ({self.calibration_quality:.2f}) is below {self.cfg.calib_min_quality:.2f}.")
            cv2.namedWindow("Calibration Quality Check", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Calibration Quality Check", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            panel = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.putText(panel, f"CALIBRATION QUALITY: {self.calibration_quality * 100:.1f}%", (80, self.screen_h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            cv2.putText(panel, "Press [SPACE] or [ENTER] to PROCEED TO TRACKING ANYWAY", (80, self.screen_h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(panel, "Press [R] to Retry calibration, or [Q] to Quit.", (80, self.screen_h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
            while True:
                cv2.imshow("Calibration Quality Check", panel)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' ') or key == 13 or key == 10:
                    cv2.destroyWindow("Calibration Quality Check")
                    print("[EyeTrack] Calibration accepted manually by user.")
                    return True
                if key == ord('r'):
                    cv2.destroyWindow("Calibration Quality Check")
                    return False
                if key == ord('q'):
                    self.exit_requested = True
                    cv2.destroyWindow("Calibration Quality Check")
                    return False
        return True

    # ─────────────────────────────────────────────
    # Phase 4: Held-Out Post-Calibration Validation Screen
    # ─────────────────────────────────────────────
    def run_validation_screen(self):
        if not self.cfg.val_enabled:
            return True

        print("\n[EyeTrack] Running Phase 4 Held-Out Post-Calibration Validation...")
        cv2.namedWindow("Post-Calibration Validation", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Post-Calibration Validation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        val_points = [
            (int(self.screen_w * 0.50), int(self.screen_h * 0.50)), # Center
            (int(self.screen_w * 0.30), int(self.screen_h * 0.30)), # Top-Left
            (int(self.screen_w * 0.70), int(self.screen_h * 0.30)), # Top-Right
            (int(self.screen_w * 0.30), int(self.screen_h * 0.70)), # Bottom-Left
            (int(self.screen_w * 0.70), int(self.screen_h * 0.70)), # Bottom-Right
        ]

        errors_px = []
        corner_errors = []
        frame_id = 0

        for idx, (vdot_x, vdot_y) in enumerate(val_points):
            move_start = time.time()
            while time.time() - move_start < 0.6:
                bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.circle(bg, (vdot_x, vdot_y), 18, (255, 255, 0), -1)
                cv2.putText(bg, f"VALIDATION CHECK ({idx+1}/5) — LOOK AT THE YELLOW DOT",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("Post-Calibration Validation", bg)
                cv2.waitKey(20)

            sample_preds = []
            samples_target = 10
            start_t = time.time()

            while len(sample_preds) < samples_target and time.time() - start_t < 4.0:
                if self.stream is not None:
                    ret, frame, new_id, _ = self.stream.read_latest(frame_id, timeout=0.2)
                    if ret: frame_id = new_id
                else:
                    ret, frame = self.read_frame()

                if not ret or frame is None: continue

                mp_img = self.make_mediapipe_image(frame)
                res = self.landmarker.detect_for_video(mp_img, self._next_ts())

                if res and res.face_landmarks:
                    (norm_x, norm_y, ear), _ = self.get_normalized_eye_vector(res.face_landmarks[0])
                    if self.cfg.min_ear <= ear <= self.cfg.max_ear:
                        pred_x, pred_y = self.mapper.predict(norm_x, norm_y, ear)
                        sample_preds.append((pred_x, pred_y))

                bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.circle(bg, (vdot_x, vdot_y), 18, (255, 255, 0), -1)
                cv2.putText(bg, f"CHECKING GAZE ACCURACY... ({len(sample_preds)}/{samples_target})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("Post-Calibration Validation", bg)
                cv2.waitKey(10)

            if sample_preds:
                avg_pred_x = float(np.mean([p[0] for p in sample_preds]))
                avg_pred_y = float(np.mean([p[1] for p in sample_preds]))
                err = math.hypot(avg_pred_x - vdot_x, avg_pred_y - vdot_y)
                errors_px.append(err)
                if idx > 0:
                    corner_errors.append(err)
            else:
                errors_px.append(500.0)
                if idx > 0: corner_errors.append(500.0)

        median_err = float(np.median(errors_px))
        p95_err = float(np.percentile(errors_px, 95))
        max_corner_err = float(np.max(corner_errors)) if corner_errors else median_err

        print(f"[EyeTrack] Validation Results: Median={median_err:.1f}px, P95={p95_err:.1f}px, CornerMax={max_corner_err:.1f}px")

        passed = (
            median_err <= self.cfg.val_max_median_error_px and
            p95_err <= self.cfg.val_max_p95_error_px and
            max_corner_err <= self.cfg.val_max_corner_error_px
        )

        bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        if passed:
            cv2.putText(bg, "VALIDATION PASSED PERFECTLY!", (80, self.screen_h // 2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.putText(bg, f"Median Error: {median_err:.1f}px | P95: {p95_err:.1f}px", (80, self.screen_h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
            cv2.putText(bg, "Starting tracking session...", (80, self.screen_h // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
            cv2.imshow("Post-Calibration Validation", bg)
            cv2.waitKey(1500)
            cv2.destroyWindow("Post-Calibration Validation")
            return True
        else:
            print(f"[EyeTrack] Validation Accuracy Notice.")
            cv2.putText(bg, f"VALIDATION ACCURACY NOTICE (Median: {median_err:.1f}px)", (80, self.screen_h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
            cv2.putText(bg, "Press [SPACE] or [ENTER] to PROCEED TO TRACKING ANYWAY",
                        (80, self.screen_h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(bg, "Press [V] to Retry Validation, [R] to Restart Calibration, or [Q] to Quit.",
                        (80, self.screen_h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

            while True:
                cv2.imshow("Post-Calibration Validation", bg)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' ') or key == 13 or key == 10:
                    cv2.destroyWindow("Post-Calibration Validation")
                    print("[EyeTrack] Validation accepted manually by user.")
                    return True
                if key == ord('v'):
                    cv2.destroyWindow("Post-Calibration Validation")
                    return self.run_validation_screen()
                if key == ord('r'):
                    cv2.destroyWindow("Post-Calibration Validation")
                    return False
                if key == ord('q'):
                    self.exit_requested = True
                    cv2.destroyWindow("Post-Calibration Validation")
                    return False

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
            # Phase 0: Gate
            if not self.run_head_positioning_gate():
                if self.exit_requested or not self.wait_for_calibration_retry():
                    return False
                continue

            # Phase 1 & 2: Calibration
            if not self.run_calibration():
                if self.exit_requested or not self.wait_for_calibration_retry():
                    return False
                continue

            # Phase 4: Validation Screen
            if not self.run_validation_screen():
                if self.exit_requested or not self.wait_for_calibration_retry():
                    return False
                continue

            return True
        return False

    def is_gaze_on_screen(self, screen_x, screen_y):
        if not self.cfg.gaze_boundary_enabled:
            return True
        return 0.0 <= screen_x <= self.screen_w and 0.0 <= screen_y <= self.screen_h

    def get_grid_cell(self, screen_x, screen_y):
        if not self.is_gaze_on_screen(screen_x, screen_y):
            return -1, -1, self.cfg.off_screen_label, 0.0

        if self.cfg.gaze_boundary_enabled:
            pad_x = self.screen_w * self.cfg.gaze_boundary_pad_x
            pad_y = self.screen_h * self.cfg.gaze_boundary_pad_y
            active_w = max(self.screen_w - 2 * pad_x, 1.0)
            active_h = max(self.screen_h - 2 * pad_y, 1.0)

            col = int((screen_x - pad_x) / (active_w / self.cfg.grid_cols))
            row = int((screen_y - pad_y) / (active_h / self.cfg.grid_rows))
        else:
            col = int(screen_x / (self.screen_w / self.cfg.grid_cols))
            row = int(screen_y / (self.screen_h / self.cfg.grid_rows))

        col = max(0, min(col, self.cfg.grid_cols - 1))
        row = max(0, min(row, self.cfg.grid_rows - 1))

        section = self.cfg.section_map[row][col]

        cell_w = self.screen_w / self.cfg.grid_cols
        cell_h = self.screen_h / self.cfg.grid_rows
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        max_dist = math.hypot(cell_w / 2, cell_h / 2)
        dist = math.hypot(screen_x - cx, screen_y - cy)
        conf = max(0.0, 1.0 - (dist / max_dist))

        return row, col, section, float(conf)

    def draw_grid_overlay(self, sx, sy, row, col, section):
        """Draws the IDE section grid overlay with active cell highlight."""
        is_off_screen = (section == self.cfg.off_screen_label)

        if self.cfg.gaze_boundary_enabled:
            pad_x = int(self.screen_w * self.cfg.gaze_boundary_pad_x)
            pad_y = int(self.screen_h * self.cfg.gaze_boundary_pad_y)
            active_w = self.screen_w - 2 * pad_x
            active_h = self.screen_h - 2 * pad_y
        else:
            pad_x = pad_y = 0
            active_w, active_h = self.screen_w, self.screen_h

        # Build base grid frame (background)
        grid_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        if is_off_screen:
            grid_img[:] = (25, 25, 100)
        else:
            grid_img[:] = (20, 20, 20)
        cv2.rectangle(grid_img, (pad_x, pad_y),
                      (pad_x + active_w, pad_y + active_h), (0, 0, 0), -1)

        cell_w = int(active_w / self.cfg.grid_cols)
        cell_h = int(active_h / self.cfg.grid_rows)

        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                sec_name = self.cfg.section_map[r][c]
                color = self.cfg.section_colors.get(sec_name, [100, 100, 100])
                x1 = pad_x + c * cell_w
                y1 = pad_y + r * cell_h
                x2 = pad_x + (c + 1) * cell_w if c < self.cfg.grid_cols - 1 else pad_x + active_w
                y2 = pad_y + (r + 1) * cell_h if r < self.cfg.grid_rows - 1 else pad_y + active_h

                is_active_cell = (not is_off_screen and r == row and c == col)
                if is_active_cell:
                    # Semi-transparent active cell highlight
                    overlay = grid_img.copy()
                    hl_color = tuple(min(255, int(v * 0.45)) for v in color)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), hl_color, -1)
                    cv2.addWeighted(overlay, 0.5, grid_img, 0.5, 0, grid_img)
                    # Bright thick border for active cell
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, 4)
                else:
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, 2)

                label = f"{sec_name} ({r},{c})"
                cv2.putText(grid_img, label, (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if is_off_screen:
            cv2.putText(grid_img, "GAZE OFF SCREEN",
                        (self.screen_w // 2 - 200, self.screen_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cx, cy = int(sx), int(sy)
        if is_off_screen:
            cv2.line(grid_img, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 0, 255), 4)
            cv2.line(grid_img, (cx - 30, cy + 30), (cx + 30, cy - 30), (0, 0, 255), 4)
            cv2.circle(grid_img, (cx, cy), 35, (0, 0, 255), 3)
            cv2.putText(grid_img, f"OFF-SCREEN GAZE ({cx}, {cy})", (cx + 40, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 220), 2)
        else:
            cv2.circle(grid_img, (cx, cy), 15, (0, 0, 255), -1)

        cv2.imshow("Eye Tracking - Gaze Grid", grid_img)

    def draw_session_summary(self):
        """Show a 4-second end-of-session summary with dwell time bar chart."""
        sections_data = self.metrics.session_sections_summary
        if not sections_data:
            return

        # Sort by dwell time descending, keep top 5
        sorted_sections = sorted(
            [(s, d) for s, d in sections_data.items() if d.get("total_dwell_ms", 0) > 0],
            key=lambda x: x[1]["total_dwell_ms"], reverse=True
        )[:5]

        bg = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        cv2.putText(bg, "SESSION COMPLETE", (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 180), 3)

        face_rate = (self.total_face_frames / self.frame_count * 100) if self.frame_count > 0 else 0
        tracking_s = (time.time() - self.tracking_start_time) if self.tracking_start_time else 0
        total_blinks = self.blink_detector.blink_count
        avg_bpm = (total_blinks / (tracking_s / 60.0)) if tracking_s > 0 else 0.0
        cv2.putText(bg, f"Tracking: {tracking_s:.0f}s  |  Face: {face_rate:.0f}%  |  Quality: {self.calibration_quality * 100:.0f}%  |  Blinks: {total_blinks} ({avg_bpm:.1f}/min)",
                    (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        cv2.putText(bg, "Top Gaze Sections:", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        max_dwell = sorted_sections[0][1]["total_dwell_ms"] if sorted_sections else 1
        bar_max_w = self.screen_w - 400
        colors_map = self.cfg.section_colors

        for i, (sec, data) in enumerate(sorted_sections):
            dwell_ms = data["total_dwell_ms"]
            bar_w = int(bar_max_w * dwell_ms / max(max_dwell, 1))
            by = 240 + i * 70
            color = tuple(colors_map.get(sec, [150, 150, 150]))
            cv2.rectangle(bg, (80, by), (80 + bar_w, by + 50), color, -1)
            cv2.putText(bg, f"{sec}  {dwell_ms / 1000:.1f}s  ({data.get('visit_count', 0)} visits)",
                        (80 + bar_w + 12, by + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)

        cv2.putText(bg, "Session data saved. Closing...",
                    (80, self.screen_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)

        cv2.namedWindow("Session Summary", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Session Summary", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Session Summary", bg)
        cv2.waitKey(4000)
        cv2.destroyWindow("Session Summary")

    def draw_help_overlay(self, base_frame):
        """Returns a copy of base_frame with a semi-transparent help panel."""
        overlay = base_frame.copy()
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 420, 320
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (180, 180, 180), 2)
        cv2.addWeighted(overlay, 0.82, base_frame, 0.18, 0, overlay)

        hotkeys = [
            ("Q",       "Quit session"),
            ("C",       "Recalibrate"),
            ("P",       "Pause / Resume"),
            ("G",       "Toggle grid overlay"),
            ("D",       "Toggle debug landmarks"),
            ("S",       "Screenshot snapshot"),
            ("H",       "Toggle this help panel"),
        ]
        cv2.putText(overlay, "KEYBOARD SHORTCUTS", (panel_x + 12, panel_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 200), 2)
        for i, (key, desc) in enumerate(hotkeys):
            y = panel_y + 66 + i * 34
            cv2.putText(overlay, f"[{key}]", (panel_x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 60), 2)
            cv2.putText(overlay, desc, (panel_x + 72, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)
        return overlay

    def start_tracking_worker(self):
        self.tracking_stop.clear()
        self.last_frame_time = None
        self.tracking_thread = threading.Thread(
            target=self.tracking_worker, name="EyeTrackingInference", daemon=True
        )
        self.tracking_thread.start()

    def stop_tracking_worker(self):
        self.tracking_stop.set()
        if self.tracking_thread is not None and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)
        self.tracking_thread = None

    def tracking_worker(self):
        last_frame_id = 0
        while not self.tracking_stop.is_set():
            if self.is_paused:
                self.tracking_stop.wait(0.01)
                continue

            if self.stream is not None:
                ret, frame, frame_id, capture_ts = self.stream.read_latest(last_frame_id, timeout=0.1)
            else:
                ret, frame = self.read_frame()
                self._sync_frame_id += 1
                frame_id, capture_ts = self._sync_frame_id, time.perf_counter()
            if not ret or frame is None:
                continue

            if frame_id > last_frame_id + 1 and last_frame_id > 0:
                self.dropped_frames += frame_id - last_frame_id - 1
            last_frame_id = frame_id
            processing_started = time.perf_counter()
            if self.processing_started_at is None:
                self.processing_started_at = processing_started
            self.processing_last_at = processing_started
            capture_age_ms = max(0.0, (processing_started - capture_ts) * 1000.0)
            ts_ms = int(time.time() * 1000)

            self.frame_count += 1
            if self.last_frame_time is not None:
                instantaneous_fps = 1.0 / max(processing_started - self.last_frame_time, 1e-6)
                self.fps_ema = instantaneous_fps if self.fps_ema == 0 else 0.9 * self.fps_ema + 0.1 * instantaneous_fps
            self.last_frame_time = processing_started

            preprocess_started = time.perf_counter()
            mp_img = self.make_mediapipe_image(frame)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

            inference_started = time.perf_counter()
            try:
                res = self.landmarker.detect_for_video(mp_img, self._next_ts())
            except Exception as exc:
                print(f"[Warn] MediaPipe error: {exc}")
                res = None
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

            mapping_started = time.perf_counter()
            face_detected = False
            raw_x = raw_y = sm_x = sm_y = 0.0
            grid_r = grid_c = -1
            section = None
            conf = iris_size = 0.0
            gaze_status = "face_missing"
            pitch = yaw = roll = 0.0
            head_pose_shifted = False
            is_blinking = False
            total_blinks = self.blink_detector.blink_count
            blink_bpm = self.blink_detector.get_blink_rate_bpm()

            if res and res.face_landmarks:
                face_detected = True
                self.total_face_frames += 1
                lm = res.face_landmarks[0]

                # Phase 3: Estimate Live Head Pose & Shift
                pitch, yaw, roll, _, _, _ = estimate_head_pose(lm, frame.shape[1], frame.shape[0])
                yaw_drift = abs(yaw - self.baseline_pose["yaw"])
                pitch_drift = abs(pitch - self.baseline_pose["pitch"])

                if yaw_drift > self.cfg.pose_tracking_warn_yaw_deg or pitch_drift > self.cfg.pose_tracking_warn_pitch_deg:
                    head_pose_shifted = True

                (norm_x, norm_y, ear), iris_size = self.get_normalized_eye_vector(lm)
                is_blinking, total_blinks, blink_bpm = self.blink_detector.process(ear, time.time())
                if self.debug_mode:
                    for point in lm:
                        cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])),
                                   1, (255, 255, 255), -1)
                if not (self.cfg.min_ear <= ear <= self.cfg.max_ear):
                    gaze_status = "eyes_invalid_or_blink"
                else:
                    raw_x, raw_y = self.mapper.predict(norm_x, norm_y, ear)
                    sm_x, sm_y = self.gaze_filter.update(raw_x, raw_y)
                    grid_r, grid_c, section, conf = self.get_grid_cell(sm_x, sm_y)
                    gaze_status = "gaze_outside_screen" if section == self.cfg.off_screen_label else "on_screen"

            self.metrics.update(ts_ms, section, iris_size)
            mapping_ms = (time.perf_counter() - mapping_started) * 1000.0
            capture_fps = self.observed_capture_fps()
            total_processing_ms = (time.perf_counter() - processing_started) * 1000.0

            row = [
                ts_ms, self.frame_count, frame_id,
                raw_x, raw_y, sm_x, sm_y, grid_r, grid_c, section or "", conf,
                self.metrics.dwell_time_ms, self.metrics.get_nrevisit(section),
                self.metrics.get_transition_rate(), self.metrics.iris_delta,
                self.fps_ema, capture_fps if capture_fps is not None else "",
                capture_age_ms, preprocess_ms, inference_ms, mapping_ms,
                total_processing_ms, self.dropped_frames,
                face_detected, gaze_status, self.calibration_quality,
                pitch, yaw, head_pose_shifted,
                is_blinking, total_blinks, blink_bpm
            ]
            logging_started = time.perf_counter()
            self.csv_writer.writerow(row)
            if self.frame_count % self.cfg.csv_buffer_size == 0:
                self.csv_file.flush()
            logging_ms = (time.perf_counter() - logging_started) * 1000.0
            actual_total_ms = (time.perf_counter() - processing_started) * 1000.0

            timing_values = {
                "capture_age_ms": capture_age_ms,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "mapping_metrics_ms": mapping_ms,
                "logging_ms": logging_ms,
                "total_processing_ms": actual_total_ms,
            }
            for name, value in timing_values.items():
                self.stage_timings[name].append(value)

            result = {
                "timestamp_ms": ts_ms, "frame": frame, "sm_x": sm_x, "sm_y": sm_y,
                "grid_r": grid_r, "grid_c": grid_c, "section": section,
                "gaze_status": gaze_status, "fps": self.fps_ema,
                "inference_ms": inference_ms, "capture_fps": capture_fps,
                "dwell_ms": self.metrics.dwell_time_ms,
                "nrevisit": self.metrics.get_nrevisit(section),
                "transition_rate": self.metrics.get_transition_rate(),
                "iris_delta": self.metrics.iris_delta,
                "head_pose_shifted": head_pose_shifted,
                "pitch": pitch, "yaw": yaw,
                "is_blinking": is_blinking, "total_blinks": total_blinks,
                "blink_bpm": blink_bpm
            }
            with self.result_lock:
                self.latest_result = result

    def draw_tracking_hud(self, frame, result):
        is_off_screen = result["section"] == self.cfg.off_screen_label
        color = (40, 40, 200) if is_off_screen else (0, 255, 0)
        status = (f"Status: {result['gaze_status']}" if result["gaze_status"] != "on_screen"
                  else f"Section: {result['section']}")
        is_blinking_str = " [BLINK]" if result.get("is_blinking", False) else ""
        lines = [
            f"Processing FPS: {result['fps']:.1f}",
            f"Capture FPS: {(result['capture_fps'] or 0):.1f}",
            f"Inference: {result['inference_ms']:.1f}ms", status,
            f"Dwell: {result['dwell_ms'] / 1000.0:.1f}s",
            f"NRevisit: {result['nrevisit']}",
            f"Trans Rate: {result['transition_rate']:.2f}/s",
            f"Blinks: {result.get('total_blinks', 0)} ({result.get('blink_bpm', 0.0):.1f}/min){is_blinking_str}",
            f"Iris D: {result['iris_delta']:.4f}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (10, 30 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Phase 3 HUD Warning Banner
        if result.get("head_pose_shifted") and self.cfg.pose_show_text_warning:
            cv2.rectangle(frame, (10, frame.shape[0] - 50), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 165, 255), -1)
            cv2.putText(frame, "WARNING: Head pose shifted - please return to center",
                        (20, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        if not self.calibrate_until_ready():
            self.cleanup()
            return

        self.tracking_start_time = time.time()
        self.ui_started = time.perf_counter()
        print(f"[EyeTrack] Camera diagnostics: {self.camera_diagnostics()}")
        cv2.namedWindow("Eye Tracking - Gaze Grid", cv2.WINDOW_NORMAL)
        print("[EyeTrack] Starting decoupled capture/inference/UI pipeline...")
        self.start_tracking_worker()

        refresh_interval = 1.0 / max(float(self.cfg.display_refresh_fps), 1.0)
        last_render = 0.0
        display_frame = None
        show_help = False

        while not self.exit_requested:
            if (self.tracking_thread is not None and not self.tracking_thread.is_alive()
                    and not self.tracking_stop.is_set()):
                print("[EyeTrack] ERROR: inference worker stopped unexpectedly.")
                break
            now = time.perf_counter()
            if now - last_render >= refresh_interval:
                with self.result_lock:
                    result = self.latest_result
                if result is not None:
                    display_frame = result["frame"].copy()
                    self.draw_tracking_hud(display_frame, result)
                    if self.is_paused:
                        cv2.putText(display_frame, "PAUSED", (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 0, 255), 2)
                    if show_help:
                        display_frame = self.draw_help_overlay(display_frame)
                    else:
                        cv2.putText(display_frame, "[H] Help",
                                    (display_frame.shape[1] - 120, display_frame.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
                    cv2.imshow("Eye Tracking - Camera Feed", display_frame)
                    if self.show_grid:
                        self.draw_grid_overlay(result["sm_x"], result["sm_y"], result["grid_r"],
                                               result["grid_c"], result["section"])
                    self.ui_frame_count += 1
                last_render = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                self.stop_tracking_worker()
                if not self.calibrate_until_ready():
                    break
                self.gaze_filter.reset()
                self.latest_result = None
                self.start_tracking_worker()
            elif key == ord('p'):
                self.is_paused = not self.is_paused
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                if not self.show_grid:
                    try: cv2.destroyWindow("Eye Tracking - Gaze Grid")
                    except cv2.error: pass
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('s') and display_frame is not None:
                path = f"snapshot_{int(time.time() * 1000)}.png"
                cv2.imwrite(path, display_frame)
                print(f"[EyeTrack] Saved {path}")
            time.sleep(0.001)

        self.stop_tracking_worker()
        if self.tracking_start_time is not None and self.frame_count > 0:
            self.draw_session_summary()
        self.cleanup()

    def cleanup(self):
        print("\n[EyeTrack] Cleaning up...")
        self.stop_tracking_worker()
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
        processing_span = ((self.processing_last_at - self.processing_started_at)
                           if self.processing_started_at is not None and self.processing_last_at is not None else 0.0)
        processing_fps = ((self.frame_count - 1) / processing_span
                          if self.frame_count > 1 and processing_span > 0 else 0.0)
        ui_duration = (time.perf_counter() - self.ui_started) if self.ui_started else 0.0
        stage_summary = {}
        for name, samples in self.stage_timings.items():
            stage_summary[name] = {
                "mean": float(np.mean(samples)) if samples else 0.0,
                "p50": float(np.percentile(samples, 50)) if samples else 0.0,
                "p95": float(np.percentile(samples, 95)) if samples else 0.0,
            }
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
                "layout_mode": self.cfg.calib_layout_mode,
                "async_capture": self.cfg.async_capture,
                "display_refresh_fps": self.cfg.display_refresh_fps,
                "latest_frame_only": self.cfg.latest_frame_only
            },
            "summary": {
                "total_frames": self.frame_count,
                "frames_with_face": self.total_face_frames,
                "face_detection_rate": self.total_face_frames / self.frame_count if self.frame_count > 0 else 0,
                "avg_fps": processing_fps,
                "processing_fps_ema_final": self.fps_ema,
                "ui_fps": self.ui_frame_count / ui_duration if ui_duration > 0 else 0.0,
                "dropped_frames": self.dropped_frames,
                "dropped_frame_ratio": (self.dropped_frames / (self.frame_count + self.dropped_frames)
                                        if self.frame_count + self.dropped_frames > 0 else 0.0),
                "stage_timings_ms": stage_summary,
                "camera": camera_stats,
                "sections": self.metrics.session_sections_summary,
                "total_blinks": self.blink_detector.blink_count,
                "avg_blink_rate_bpm": ((self.blink_detector.blink_count / (tracking_duration / 60.0)) if tracking_duration > 0 else 0.0),
                "calibration_quality": self.calibration_quality,
                "calibration_diagnostics": self.calibration_diagnostics,
                "baseline_head_pose": self.baseline_pose
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
