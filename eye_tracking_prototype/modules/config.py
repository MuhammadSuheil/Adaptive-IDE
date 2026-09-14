import os
import yaml

class Config:
    def __init__(self, path):
        # Resolve path relative to the project root (parent of modules/)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)
        
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
            
        with open(path, 'r') as f:
            self.data = yaml.safe_load(f)
        
        self.webcam_idx = self.data['webcam']['device_index']
        self.webcam_fps = self.data['webcam']['fps_target']
        self.webcam_w = self.data['webcam']['width']
        self.webcam_h = self.data['webcam']['height']
        self.inference_w = self.data['webcam'].get('inference_width', self.webcam_w)
        self.inference_h = self.data['webcam'].get('inference_height', self.webcam_h)
        self.flip_horizontal = self.data['webcam'].get('flip_horizontal', True)
        self.async_capture = self.data['webcam'].get('async_capture', True)
        self.display_refresh_fps = self.data.get('display', {}).get('refresh_fps', 15)
        self.latest_frame_only = self.data.get('performance', {}).get('latest_frame_only', True)
        
        self.grid_rows = self.data['grid']['rows']
        self.grid_cols = self.data['grid']['cols']
        self.section_map = self.data['grid']['section_map']
        self.section_colors = self.data['grid']['section_colors']
        
        boundary_cfg = self.data.get('gaze_boundary', {})
        self.gaze_boundary_enabled = boundary_cfg.get('enabled', True)
        self.gaze_boundary_pad_x = boundary_cfg.get('padding_x', 0.15)
        self.gaze_boundary_pad_y = boundary_cfg.get('padding_y', 0.15)
        self.off_screen_label = boundary_cfg.get('off_screen_label', 'off_screen')
        
        self.filter_type = self.data['filter']['type']

        self.ema_alpha = self.data['filter']['ema_alpha']
        self.median_window = self.data['filter']['median_window']
        self.kalman_p_noise = self.data['filter'].get('kalman_process_noise', 0.1)
        self.kalman_m_noise = self.data['filter'].get('kalman_measurement_noise', 4.0)
        # Adaptive EMA
        self.adaptive_ema = self.data['filter'].get('adaptive_ema', False)
        self.adaptive_ema_low_v_alpha = self.data['filter'].get('adaptive_ema_low_v_alpha', 0.12)
        self.adaptive_ema_high_v_alpha = self.data['filter'].get('adaptive_ema_high_v_alpha', 0.70)
        self.adaptive_ema_saccade_threshold_px = self.data['filter'].get('adaptive_ema_saccade_threshold_px', 80.0)
        
        self.dwell_threshold = self.data['dwell']['threshold_ms']
        self.transition_window_sec = self.data.get('metrics', {}).get('transition_window_sec', 5.0)
        
        # Blink Detection & Counting
        blink_cfg = self.data.get('blink', {})
        self.blink_enabled = blink_cfg.get('enabled', True)
        self.blink_ear_threshold = blink_cfg.get('ear_threshold', 0.20)
        self.blink_min_frames = blink_cfg.get('min_blink_frames', 1)
        self.blink_max_frames = blink_cfg.get('max_blink_frames', 10)
        
        # Head Positioning Gate (Phase 0)
        hp_cfg = self.data.get('head_positioning', {})
        self.hp_enabled = hp_cfg.get('enabled', True)
        self.hp_fullscreen = hp_cfg.get('fullscreen', True)
        self.hp_guide_height_ratio = float(hp_cfg.get('guide_height_ratio', 0.42))
        self.hp_guide_width_to_height = float(hp_cfg.get('guide_width_to_height', 0.72))
        if not (0.15 <= self.hp_guide_height_ratio <= 0.8 and
                0.5 <= self.hp_guide_width_to_height <= 1.0):
            raise ValueError("Head guide height must be 0.15..0.8 and width/height 0.5..1.0")
        self.hp_target_face_width_ratio = hp_cfg.get('target_face_width_ratio', 0.35)
        self.hp_target_center_x_ratio = hp_cfg.get('target_center_x_ratio', 0.5)
        self.hp_target_center_y_ratio = hp_cfg.get('target_center_y_ratio', 0.45)
        self.hp_alignment_tolerance_ratio = hp_cfg.get('alignment_tolerance_ratio', 0.08)
        self.hp_size_tolerance_ratio = hp_cfg.get('size_tolerance_ratio', 0.10)
        self.hp_max_yaw_deg = hp_cfg.get('max_yaw_deg', 10.0)
        self.hp_max_pitch_deg = hp_cfg.get('max_pitch_deg', 8.0)
        self.hp_stability_frames_required = hp_cfg.get('stability_frames_required', 15)
        self.hp_countdown_seconds = max(0.1, float(hp_cfg.get('countdown_seconds', 5)))

        # Head Pose Monitoring (Phase 3)
        pose_cfg = self.data.get('head_pose', {})
        self.pose_max_calib_yaw_deg = pose_cfg.get('max_calib_yaw_deg', 8.0)
        self.pose_max_calib_pitch_deg = pose_cfg.get('max_calib_pitch_deg', 6.0)
        self.pose_tracking_warn_yaw_deg = pose_cfg.get('tracking_warn_yaw_deg', 15.0)
        self.pose_tracking_warn_pitch_deg = pose_cfg.get('tracking_warn_pitch_deg', 12.0)
        self.pose_show_text_warning = pose_cfg.get('show_text_warning', True)

        self.calib_layout_mode = self.data['calibration'].get('layout_mode', '3x3')
        self.calib_samples = self.data['calibration']['n_samples_per_point']
        self.calib_move_delay_sec = self.data['calibration'].get('move_delay_sec', 0.8)
        self.calib_target_margin = self.data['calibration'].get('target_margin', 0.0)
        self.calib_radius = self.data['calibration']['dot_radius']
        self.calib_color = self.data['calibration']['dot_color_bgr']
        self.calib_stability_thresh = self.data['calibration'].get('stability_threshold', 0.025)
        self.calib_stability_frames = self.data['calibration'].get('stability_required_frames', 4)
        self.calib_mapping_method = self.data['calibration'].get('mapping_method', 'rbf')
        self.calib_rbf_kernel = self.data['calibration'].get('rbf_kernel', 'thin_plate_spline')
        self.calib_rbf_smoothing = self.data['calibration'].get('rbf_smoothing', 0.0)
        self.calib_output_clamp = self.data['calibration'].get('output_clamp', True)
        self.calib_smooth_pursuit_enabled = self.data['calibration'].get('smooth_pursuit_enabled', False)
        self.calib_point_timeout_sec = self.data['calibration'].get('point_timeout_sec', 12.0)
        self.calib_max_sample_std = self.data['calibration'].get('max_sample_std', 0.040)
        self.calib_outlier_mad_scale = self.data['calibration'].get('outlier_mad_scale', 3.5)
        self.calib_min_feature_span_x = self.data['calibration'].get('min_feature_span_x', 0.015)
        self.calib_min_feature_span_y = self.data['calibration'].get('min_feature_span_y', 0.010)
        self.calib_min_quality = self.data['calibration'].get('min_quality', 0.35)
        # Dot animation
        self.calib_dot_pulse = self.data['calibration'].get('dot_pulse_animation', True)
        self.calib_dot_crosshair = self.data['calibration'].get('dot_crosshair_on_lock', True)

        # Validation Screen (Phase 4)
        val_cfg = self.data.get('validation', {})
        self.val_enabled = val_cfg.get('enabled', True)
        self.val_failure_strategy = val_cfg.get('failure_strategy', 'retry_validation_only')
        self.val_max_median_error_px = val_cfg.get('max_median_error_px', 180)
        self.val_max_p95_error_px = val_cfg.get('max_p95_error_px', 350)
        self.val_max_corner_error_px = val_cfg.get('max_corner_error_px', 300)

        eye_validity = self.data.get('eye_validity', {})
        self.min_ear = eye_validity.get('min_ear', 0.08)
        self.max_ear = eye_validity.get('max_ear', 0.65)
        
        self.iris_baseline_frames = self.data['iris']['baseline_frames']
        
        self.session_dir = self.data['output']['session_dir']
        if not os.path.isabs(self.session_dir):
            self.session_dir = os.path.join(project_root, self.session_dir)
            
        self.csv_buffer_size = self.data['output']['csv_buffer_size']
        self.save_video = self.data['output']['save_video']
        
        self.model_path = self.data['mediapipe']['model_path']
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(project_root, self.model_path)
            
        lm_cfg = self.data['mediapipe'].get('landmarks', {})
        self.left_iris_indices = lm_cfg.get('left_iris', [468, 469, 470, 471, 472])
        self.right_iris_indices = lm_cfg.get('right_iris', [473, 474, 475, 476, 477])
        self.left_eye_corners = lm_cfg.get('left_eye_corners', [33, 133])
        self.right_eye_corners = lm_cfg.get('right_eye_corners', [362, 263])
        self.left_eyelid = lm_cfg.get('left_eyelid', [159, 145, 133, 33])
        self.right_eyelid = lm_cfg.get('right_eyelid', [386, 374, 263, 362])
