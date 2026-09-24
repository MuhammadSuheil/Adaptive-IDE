import csv
from pathlib import Path
import tempfile
from datetime import datetime
import unittest
from unittest.mock import Mock, patch

import fuse_and_analyze as fusion
import run_multimodal as runner


class MultimodalTests(unittest.TestCase):
    def test_invalid_arguments_do_not_start_recorders(self):
        cases = [['--skip-eye', '--skip-hrv']]
        cases += [['--duration', value] for value in ('0', '-1', 'nan', 'inf')]
        cases += [['--target-baseline', value] for value in ('0', '-1', 'nan', 'inf')]
        with patch.object(runner.subprocess, 'Popen') as spawn:
            for args in cases:
                with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                    runner.main(args)
                self.assertEqual(raised.exception.code, 2)
            spawn.assert_not_called()

    def test_sessions_started_same_second_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 9, 19, 12, 0, 0)
            first = runner.setup_session_directory(Path(directory), 'P01', 'task')
            marker = first / 'hrv.csv'
            marker.write_text('original')
            second = runner.setup_session_directory(Path(directory), 'P01', 'task')
            self.assertNotEqual(first, second)
            self.assertEqual(marker.read_text(), 'original')

    def test_interrupt_during_startup_stops_hrv(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Mock()
            proc.poll.return_value = None
            proc.wait.side_effect = lambda timeout: setattr(proc.poll, 'return_value', 0)
            with patch.object(runner, 'setup_session_directory', return_value=Path(directory)), \
                 patch.object(runner.subprocess, 'Popen', return_value=proc) as spawn, \
                 patch.object(runner.time, 'sleep', side_effect=KeyboardInterrupt):
                self.assertEqual(runner.main([]), 1)  # No source recordings yet.
            spawn.assert_called_once()
            proc.send_signal.assert_called_once()
            proc.wait.assert_called_once()

    def test_recorder_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / 'hrv.csv').touch()
            proc = Mock()
            proc.poll.return_value = 1
            with patch.object(runner, 'setup_session_directory', return_value=session), \
                 patch.object(runner.subprocess, 'Popen', return_value=proc), \
                 patch.object(runner.time, 'sleep'):
                self.assertEqual(runner.main(['--skip-eye']), 1)

    def test_fusion_aligns_eye_frames_and_creates_four_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / 'hrv.csv').write_text(
                'phase,window_start,window_end,window_seconds,rmssd_ms,sdnn_ms,status\n'
                'baseline,,,120,50,60,ready\n'
                'task,2026-09-19T12:00:00+00:00,2026-09-19T12:00:25+00:00,25,25,30,ready\n',
                encoding='utf-8')
            start = fusion.parse_iso_to_ms('2026-09-19T12:00:00+00:00')
            (session / 'eye_tracking.csv').write_text(
                'timestamp_ms,frame_index,gaze_status,face_detected,section\n'
                f'{start + 1000},1,on_screen,True,editor\n'
                f'{start + 26000},2,off_screen,True,off_screen\n', encoding='utf-8')
            (session / 'rr_raw.csv').write_text('raw_id,rr_original_ms\n1,800\n')
            originals = {p: p.read_bytes() for p in session.glob('*.csv')}
            report = fusion.fuse_session(session)
            self.assertEqual(report['total_windows'], 1)
            self.assertEqual(len(list(session.glob('*.csv'))), 4)
            with (session / 'multimodal_timeline.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['eye_frames_count'], '1')
            self.assertEqual(row['rmssd_ratio'], '0.5')
            self.assertEqual(row['dominant_section'], 'editor')
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)

    def test_short_session_still_writes_csv_header(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / 'hrv.csv').write_text('phase,rmssd_ms,status\nbaseline,50,ready\n')
            (session / 'eye_tracking.csv').write_text('timestamp_ms,frame_index\n')
            report = fusion.fuse_session(session)
            self.assertEqual(report['total_windows'], 0)
            with (session / 'multimodal_timeline.csv').open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, fusion.TIMELINE_COLUMNS)
                self.assertEqual(list(reader), [])

    def test_eye_exit_stops_hrv_before_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            hrv = Mock()
            eye = Mock()
            hrv.poll.return_value = None
            eye.poll.return_value = 0
            def finish(timeout=None):
                hrv.poll.return_value = 0
                (session / 'hrv.csv').touch()
                (session / 'eye_tracking.csv').touch()
            hrv.wait.side_effect = finish
            def fuse(path):
                hrv.wait.assert_called_once()
                self.assertEqual(path, session)
                return {'timeline_file': 'multimodal_timeline.csv'}
            with patch.object(runner, 'setup_session_directory', return_value=session), \
                 patch.object(runner.subprocess, 'Popen', side_effect=[hrv, eye]), \
                 patch.object(runner.time, 'sleep'), \
                 patch.object(runner, 'fuse_session', side_effect=fuse) as fusion_mock:
                self.assertEqual(runner.main([]), 0)
            hrv.send_signal.assert_called_once()
            hrv.kill.assert_not_called()
            fusion_mock.assert_called_once()

    def test_mock_hrv_flag_passed_to_hrv_process(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            hrv = Mock()
            hrv.poll.return_value = 0
            with patch.object(runner, 'setup_session_directory', return_value=session), \
                 patch.object(runner.subprocess, 'Popen', return_value=hrv) as spawn, \
                 patch.object(runner.time, 'sleep'):
                runner.main(['--skip-eye', '--mock-hrv', '--target-baseline', '15'])
            spawn.assert_called_once()
            cmd = spawn.call_args[0][0]
            self.assertIn('--mock', cmd)
            self.assertIn('--target-baseline', cmd)
            self.assertIn('15.0', cmd)

    def test_mock_sensor_records_valid_rr_data(self):
        import asyncio
        from hrv_monitor_prototype.record_rr import record_sensor
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            asyncio.run(record_sensor(duration=1.2, session_dir=session, mock=True))
            raw_path = session / 'rr_raw.csv'
            self.assertTrue(raw_path.exists())
            with raw_path.open(encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
            self.assertGreater(len(reader), 0)
            self.assertIn('rr_original_ms', reader[0])
            first_rr = float(reader[0]['rr_original_ms'])
            self.assertGreaterEqual(first_rr, 600.0)
            self.assertLessEqual(first_rr, 1200.0)

    def test_hrv_calib_status_file_is_written(self):
        import asyncio
        import json
        from hrv_monitor_prototype.calibrate import SessionCalibration
        from hrv_monitor_prototype.record_rr import record_sensor
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            calibration = SessionCalibration(task_duration=1.0, target_seconds=2.0, maximum_seconds=10.0)
            asyncio.run(record_sensor(duration=1.5, session_dir=session, monitor=calibration, mock=True))
            status_path = session / 'hrv_calib_status.json'
            self.assertTrue(status_path.exists())
            with status_path.open(encoding='utf-8') as f:
                data = json.load(f)
            self.assertIn('text', data)
            self.assertIn('Kalibrasi:', data['text'])
            self.assertIn('RR diterima:', data['text'])
            self.assertIn('status', data)


if __name__ == '__main__':
    unittest.main()
