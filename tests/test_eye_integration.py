import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


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
                    '--session-dir', str(session), '--session-id', 'shared',
                ])
            self.assertEqual(result, 0)
            self.assertEqual(apps[0].session_id, 'shared')
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


if __name__ == '__main__':
    unittest.main()
