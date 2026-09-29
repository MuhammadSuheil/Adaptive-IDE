import csv
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
EYE_ROOT = ROOT / 'eye_tracking_prototype'


class EyeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(EYE_ROOT))
        try:
            spec = importlib.util.spec_from_file_location(
                'eye_integration_app', EYE_ROOT / 'eye_tracking_prototype.py')
            cls.eye = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.eye)
        finally:
            sys.path.remove(str(EYE_ROOT))

    def test_launcher_arguments_create_shared_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / 'shared'
            apps = []

            def fake_run(app):
                apps.append(app)
                app.csv_file.flush()
                app.csv_file.close()

            with patch.object(self.eye, 'WebcamStream'), \
                 patch.object(self.eye.vision.FaceLandmarker, 'create_from_options'), \
                 patch.object(self.eye.EyeTrackerApp, 'run', fake_run):
                result = self.eye.main([
                    '--config', str(EYE_ROOT / 'config.yaml'),
                    '--session-dir', str(session), '--session-id', 'shared', '--wait-for-hrv',
                ])
            self.assertEqual(result, 0)
            self.assertEqual(apps[0].session_id, 'shared')
            self.assertTrue(apps[0].wait_for_hrv)
            self.assertEqual(Path(apps[0].csv_path), session / 'eye_tracking.csv')
            self.assertEqual(Path(apps[0].json_path), session / 'eye_summary.json')
            with (session / 'eye_tracking.csv').open() as stream:
                columns = next(csv.reader(stream))
            self.assertIn('timestamp_ms', columns)
            self.assertIn('gaze_status', columns)

    def test_worker_records_capture_time_and_blank_invalid_iris(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.eye, 'WebcamStream') as camera, \
                 patch.object(self.eye.vision.FaceLandmarker, 'create_from_options') as model:
                app = self.eye.EyeTrackerApp(str(EYE_ROOT / 'config.yaml'), session_dir=directory)
                model.return_value.detect_for_video.return_value.face_landmarks = []
                capture_time = app.clock_origin_mono + .1
                frame = self.eye.np.zeros((480, 640, 3), dtype=self.eye.np.uint8)
                def read(*args, **kwargs):
                    app.tracking_stop.set()  # Finish after this one frame.
                    return True, frame, 100, capture_time
                camera.return_value.start.return_value.read_latest.side_effect = read
                camera.return_value.start.return_value.observed_fps.return_value = 60.0
                try:
                    app.tracking_worker()
                    app.csv_file.flush()
                    with Path(app.csv_path).open() as source:
                        row = next(csv.DictReader(source))
                    self.assertNotIn(None, row)
                    self.assertEqual(int(row['timestamp_ms']), int((app.clock_origin_epoch + .1) * 1000))
                    self.assertEqual(row['iris_size_delta'], '')
                    self.assertEqual(row['iris_valid'], 'False')
                    self.assertEqual(row['capture_frame_id'], '100')
                finally:
                    app.csv_file.close()

    def test_standalone_keeps_original_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.eye.Config(str(EYE_ROOT / 'config.yaml'))
            config.session_dir = directory
            with patch.object(self.eye, 'Config', return_value=config), \
                 patch.object(self.eye, 'WebcamStream'), \
                 patch.object(self.eye.vision.FaceLandmarker, 'create_from_options'):
                app = self.eye.EyeTrackerApp('config.yaml', session_id='standalone')
            try:
                self.assertEqual(Path(app.csv_path).parent, Path(directory))
                self.assertTrue(Path(app.csv_path).name.startswith('session_standalone_'))
                self.assertTrue(app.json_path.endswith('_summary.json'))
            finally:
                app.csv_file.close()

    def test_blink_and_one_closed_eye_do_not_reach_mapper_or_filter(self):
        for left, right in ((.15, .15), (.1, .4)):
            with self.subTest(ears=(left, right)), tempfile.TemporaryDirectory() as directory:
                with patch.object(self.eye, 'WebcamStream') as camera, \
                     patch.object(self.eye.vision.FaceLandmarker, 'create_from_options') as model, \
                     patch.object(self.eye, 'estimate_head_pose', return_value=(0, 0, 0, .2, .5, .5)):
                    app = self.eye.EyeTrackerApp(str(EYE_ROOT / 'config.yaml'), session_dir=directory)
                    model.return_value.detect_for_video.return_value.face_landmarks = [[object()]]
                    app.get_normalized_eye_vector = Mock(return_value=((.5, .5, (left+right)/2, left, right), .1))
                    app.mapper = Mock()
                    app.gaze_filter = Mock()
                    frame = self.eye.np.zeros((480, 640, 3), dtype=self.eye.np.uint8)
                    def read(*args, **kwargs):
                        app.tracking_stop.set()
                        return True, frame, 1, app.clock_origin_mono
                    camera.return_value.start.return_value.read_latest.side_effect = read
                    camera.return_value.start.return_value.observed_fps.return_value = 60.
                    try:
                        app.tracking_worker()
                        app.csv_file.flush()
                        app.mapper.predict.assert_not_called()
                        app.gaze_filter.update.assert_not_called()
                        app.gaze_filter.invalidate.assert_called_once()
                        with Path(app.csv_path).open() as source:
                            row = next(csv.DictReader(source))
                        self.assertEqual(row['gaze_status'], 'eyes_invalid_or_blink')
                        self.assertEqual(row['gaze_x_smooth'], '')
                        self.assertEqual(row['dwell_time_ms'], '0')
                        self.assertEqual(row['iris_valid'], 'False')
                        self.assertIsNone(app.latest_result['sm_x'])
                    finally:
                        app.csv_file.close()

    def test_calibration_audit_preserves_manual_decision_and_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.eye, 'WebcamStream'), \
                 patch.object(self.eye.vision.FaceLandmarker, 'create_from_options'):
                app = self.eye.EyeTrackerApp(str(EYE_ROOT / 'config.yaml'), session_dir=directory)
            try:
                app.calibration_audit.extend([
                    {'phase': 'head_gate', 'decision': 'passed', 'resets': 2},
                    {'phase': 'held_out_validation', 'decision': 'accepted_manual',
                     'passed_thresholds': False, 'sample_error_p95_px': 800},
                ])
                app.save_calibration_audit()
                data = json.loads((Path(directory) / 'eye_summary_calibration.json').read_text())
                self.assertFalse(data['attempts'][1]['passed_thresholds'])
                self.assertEqual(data['attempts'][1]['decision'], 'accepted_manual')
                self.assertEqual(data['config']['webcam']['fourcc'], 'MJPG')
            finally:
                app.csv_file.close()

    def test_interrupt_flushes_app_through_cleanup(self):
        with patch.object(self.eye, 'EyeTrackerApp') as factory:
            app = factory.return_value
            app.csv_file.closed = False
            app.run.side_effect = KeyboardInterrupt
            self.assertEqual(self.eye.main(['--session-dir', 'shared']), 0)
            app.cleanup.assert_called_once()

    def test_normal_exit_does_not_cleanup_twice(self):
        with patch.object(self.eye, 'EyeTrackerApp') as factory:
            app = factory.return_value
            app.csv_file.closed = True
            self.assertEqual(self.eye.main([]), 0)
            app.cleanup.assert_not_called()

    def test_hrv_screen_opens_only_after_eye_validation_is_accepted(self):
        app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
        app.exit_requested = False
        app.save_calibration_audit = Mock()
        steps = Mock()
        for method in ('run_head_positioning_gate', 'run_calibration',
                       'run_validation_screen', 'wait_for_hrv_calibration'):
            action = Mock(return_value=True)
            setattr(app, method, action)
            steps.attach_mock(action, method)
        self.assertTrue(app.calibrate_until_ready())
        self.assertEqual([call[0] for call in steps.mock_calls], [
            'run_head_positioning_gate', 'run_calibration',
            'run_validation_screen', 'wait_for_hrv_calibration',
        ])
        app.run_validation_screen.return_value = False
        app.wait_for_calibration_retry = Mock(return_value=False)
        app.wait_for_hrv_calibration.reset_mock()
        self.assertFalse(app.calibrate_until_ready())
        app.wait_for_hrv_calibration.assert_not_called()

    def test_hrv_screen_waits_for_late_sensor_status(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
            app.session_dir = directory
            app.wait_for_hrv = True
            app.exit_requested = False
            app.screen_w, app.screen_h = 1280, 800
            app.get_hrv_calib_status = Mock(side_effect=[
                None, None, {'status': 'calibrating'},
                {'status': 'calibrating', 'accepted_rr_seconds': 30},
                {'status': 'ready', 'accepted_rr_seconds': 120},
            ])
            with patch.object(self.eye, 'cv2') as ui:
                ui.waitKey.return_value = -1
                self.assertTrue(app.wait_for_hrv_calibration())
            self.assertTrue((Path(directory) / 'eye_calibration_ready').is_file())
            self.assertEqual(app.get_hrv_calib_status.call_count, 5)
            ui.destroyWindow.assert_called_once()

    def test_hrv_screen_handles_timeout_and_cancel(self):
        for status, key in [('unavailable', -1), ('calibrating', ord('q'))]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
                app.session_dir = directory
                app.wait_for_hrv = True
                app.exit_requested = False
                app.screen_w, app.screen_h = 1280, 800
                app.get_hrv_calib_status = Mock(return_value={'status': status})
                with patch.object(self.eye, 'cv2') as ui:
                    ui.waitKey.return_value = key
                    self.assertFalse(app.wait_for_hrv_calibration())
                ui.destroyWindow.assert_called_once()

    def test_eye_only_does_not_wait_for_hrv(self):
        app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
        app.wait_for_hrv = False
        with patch.object(self.eye, 'cv2') as ui:
            self.assertTrue(app.wait_for_hrv_calibration())
        ui.namedWindow.assert_not_called()

    def test_hrv_already_ready_skips_screen_and_releases_task(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
            app.session_dir = directory
            app.wait_for_hrv = True
            app.get_hrv_calib_status = Mock(return_value={'status': 'ready'})
            with patch.object(self.eye, 'cv2') as ui:
                self.assertTrue(app.wait_for_hrv_calibration())
            ui.namedWindow.assert_not_called()
            self.assertTrue((Path(directory) / 'eye_calibration_ready').is_file())

    def test_hrv_screen_displays_remaining_accepted_rr(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.eye.EyeTrackerApp.__new__(self.eye.EyeTrackerApp)
            app.session_dir = directory
            app.wait_for_hrv = True
            app.exit_requested = False
            app.screen_w, app.screen_h = 1280, 800
            app.get_hrv_calib_status = Mock(return_value={
                'status': 'calibrating', 'target_seconds': 120,
                'accepted_rr_seconds': 85, 'elapsed_seconds': 100,
            })
            with patch.object(self.eye, 'cv2') as ui:
                ui.waitKey.return_value = ord('q')
                self.assertFalse(app.wait_for_hrv_calibration())
            texts = [call.args[1] for call in ui.putText.call_args_list]
            self.assertIn('SISA RR DIBUTUHKAN: 35.0 DETIK', texts)


if __name__ == '__main__':
    unittest.main()
