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
        self.flip_horizontal = self.data['webcam'].get('flip_horizontal', True)
        self.async_capture = self.data['webcam'].get('async_capture', True)
        
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
        
        self.dwell_threshold = self.data['dwell']['threshold_ms']
        self.transition_window_sec = self.data.get('metrics', {}).get('transition_window_sec', 5.0)
        
        self.calib_samples = self.data['calibration']['n_samples_per_point']
        self.calib_move_delay_sec = self.data['calibration'].get('move_delay_sec', 2.0)
        self.calib_radius = self.data['calibration']['dot_radius']
        self.calib_color = self.data['calibration']['dot_color_bgr']
        self.calib_stability_thresh = self.data['calibration'].get('stability_threshold', 0.015)
        self.calib_stability_frames = self.data['calibration'].get('stability_required_frames', 6)
        self.calib_mapping_method = self.data['calibration'].get('mapping_method', 'hybrid')
        
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
