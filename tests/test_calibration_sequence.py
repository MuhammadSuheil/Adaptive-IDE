"""Concurrent calibration; task recording waits until both sensors are ready."""

import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

from hrv_monitor_prototype.calibrate import SessionCalibration
from hrv_monitor_prototype.record_rr import RRRecorder


class CalibrationSequenceTests(unittest.TestCase):
    origin = datetime(2026, 9, 24, tzinfo=timezone.utc)

    def send_rr(self, calibration, recorder, elapsed):
        recorder.record(bytes([0x16, 60, 0, 4]),
                        (self.origin + timedelta(seconds=elapsed)).isoformat(),
                        elapsed, phase=calibration.phase)
        calibration.receive(recorder.last_rows)
        calibration.tick(elapsed)

    def test_hrv_finishes_first_but_task_waits_for_eye(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            session = Path(directory)
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5,
                                         task_duration=25, wait_for_eye=True)
            recorder = RRRecorder(raw)
            monitor.start(session)
            try:
                for elapsed in (1, 2, 3):
                    self.send_rr(monitor, recorder, elapsed)
                self.assertEqual(monitor.result['status'], 'ready')
                self.assertEqual(monitor.result['accepted_rr_seconds'], 3)
                self.assertIsNone(monitor.task_start)
                self.assertEqual(monitor.phase, 'waiting_for_eye')
                self.send_rr(monitor, recorder, 4)
                self.assertEqual(len(monitor.rows), 3)
                self.assertFalse(monitor.tick(600))
                self.assertIsNone(monitor.task_start)
                (session / 'eye_calibration_ready').touch()
                monitor.tick(600)
                self.assertEqual(monitor.task_start, 600)
                for elapsed in range(601, 626):
                    self.send_rr(monitor, recorder, elapsed)
                self.assertTrue(monitor.tick(625))
            finally:
                monitor.finish(625, 'monitor_finished')
            with (session / 'hrv.csv').open() as stream:
                baseline, task = list(csv.DictReader(stream))
            self.assertEqual(float(baseline['window_seconds']), 3)
            self.assertEqual(baseline['window_start'], self.origin.isoformat())
            self.assertEqual(task['window_start'], (self.origin + timedelta(seconds=600)).isoformat())
            self.assertEqual(task['window_end'], (self.origin + timedelta(seconds=625)).isoformat())
            self.assertEqual(task['status'], 'ready')

    def test_eye_finishes_first_without_resetting_hrv_progress(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            session = Path(directory)
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5, wait_for_eye=True)
            recorder = RRRecorder(raw)
            monitor.start(session)
            try:
                self.send_rr(monitor, recorder, 1)
                status = json.loads((session / 'hrv_calib_status.json').read_text())
                self.assertEqual(status['accepted_rr_seconds'], 1)
                self.assertEqual(status['remaining_seconds'], 2)
                (session / 'eye_calibration_ready').touch()
                self.send_rr(monitor, recorder, 2)
                self.send_rr(monitor, recorder, 3)
                self.assertEqual(monitor.result['accepted_rr_seconds'], 3)
                self.assertEqual(monitor.task_start, 3)
                self.assertEqual(monitor.phase, 'task')
            finally:
                monitor.finish(3, 'stopped')

    def test_no_rr_timeout_includes_time_spent_on_eye_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5, wait_for_eye=True)
            monitor.start(directory)
            try:
                self.assertFalse(monitor.tick(4.9))
                self.assertTrue(monitor.tick(5))
                self.assertEqual(monitor.result['reason'], 'timeout')
                self.assertIsNone(monitor.task_start)
            finally:
                monitor.finish(5, 'monitor_finished')

    def test_standalone_starts_task_without_eye_signal(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5)
            recorder = RRRecorder(raw)
            monitor.start(directory)
            try:
                for elapsed in (1, 2, 3):
                    self.send_rr(monitor, recorder, elapsed)
                self.assertEqual(monitor.task_start, 3)
                self.assertEqual(monitor.phase, 'task')
            finally:
                monitor.finish(3, 'stopped')


if __name__ == '__main__':
    unittest.main()
