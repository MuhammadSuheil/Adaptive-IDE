import csv
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
