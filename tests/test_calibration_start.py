"""Baseline starts at the shared Q trigger, independently of BLE startup."""

import asyncio
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hrv_monitor_prototype.calibrate import SessionCalibration
from hrv_monitor_prototype.record_rr import RRRecorder, record_sensor


class CalibrationStartTests(unittest.TestCase):
    origin = datetime(2026, 10, 27, tzinfo=timezone.utc)

    def send(self, monitor, recorder, elapsed, count=1):
        recorder.record(bytes([0x16, 60]) + bytes([0, 4]) * count,
                        (self.origin + timedelta(seconds=elapsed)).isoformat(),
                        elapsed, phase=monitor.phase)
        originals = [dict(row) for row in recorder.last_rows]
        monitor.receive(recorder.last_rows)
        self.assertEqual(recorder.last_rows, originals)
        return monitor.tick(elapsed)

    def trigger(self, session, seconds):
        timestamp = (self.origin + timedelta(seconds=seconds)).isoformat()
        (session / 'calibration_start.json').write_text(
            json.dumps({'started_at_utc': timestamp}), encoding='utf-8')
        return timestamp

    def test_preview_rr_does_not_count_or_timeout(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5, wait_for_eye=True)
            session = Path(directory)
            monitor.start(session)
            try:
                recorder = RRRecorder(raw)
                self.assertFalse(self.send(monitor, recorder, 1))
                self.assertFalse(monitor.tick(600))
                stale = json.loads((session / 'hrv_calib_status.json').read_text())
                self.assertFalse(stale['start_ready'])
                self.assertFalse(self.send(monitor, recorder, 600))
                self.assertIsNone(monitor.result)
                self.assertEqual(monitor.rows, [])
                status = json.loads((session / 'hrv_calib_status.json').read_text())
                self.assertEqual(status['phase'], 'waiting_for_start')
                self.assertEqual(status['elapsed_seconds'], 0)
                self.assertTrue(status['start_ready'])
            finally:
                monitor.finish(600, 'interrupted')

    def test_start_timestamp_excludes_crossing_intervals_and_keeps_task_clock(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            session = Path(directory)
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5, wait_for_eye=True)
            monitor.start(session)
            try:
                recorder = RRRecorder(raw)
                self.send(monitor, recorder, 1)
                start = self.trigger(session, 10)
                # One interval before Q and one crossing Q: neither is baseline.
                self.send(monitor, recorder, 10.5, count=2)
                used, seconds = monitor.progress(10.5)
                self.assertEqual(seconds, 0)
                self.assertEqual(used, [])
                for elapsed in (11.5, 12.5, 13.5):
                    self.send(monitor, recorder, elapsed)
                self.assertEqual(monitor.result['status'], 'ready')
                self.assertEqual(monitor.result['accepted_rr_seconds'], 3)
                self.assertEqual(monitor.result['calibration_started_at_utc'], start)
                self.assertEqual(monitor.phase, 'waiting_for_eye')
                self.trigger(session, 20)  # Eye retries must never reset HRV.
                monitor.tick(20)
                self.assertEqual(monitor.result['calibration_started_at_utc'], start)
                (session / 'eye_calibration_ready').write_text(
                    (self.origin + timedelta(seconds=25)).isoformat())
                monitor.tick(25)
                self.assertEqual(monitor.task_start, 25)
                self.assertEqual(monitor.task_hrv.origin, self.origin + timedelta(seconds=25))
            finally:
                monitor.finish(25, 'interrupted')
            with (session / 'hrv.csv').open() as stream:
                baseline = next(csv.DictReader(stream))
            self.assertEqual(datetime.fromisoformat(baseline['window_start']), self.origin + timedelta(seconds=10))
            self.assertEqual(float(baseline['window_seconds']), 3.5)

    def test_timeout_starts_at_q_not_at_connection(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as raw:
            session = Path(directory)
            monitor = SessionCalibration(target_seconds=3, maximum_seconds=5, wait_for_eye=True)
            monitor.start(session)
            try:
                self.send(monitor, RRRecorder(raw), 1)
                self.trigger(session, 100)
                self.assertFalse(monitor.tick(104.9))
                self.assertTrue(monitor.tick(105))
                self.assertEqual(monitor.result['reason'], 'timeout')
            finally:
                monitor.finish(105, 'interrupted')

    def test_exit_before_q_writes_no_baseline_result(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            monitor = SessionCalibration(wait_for_eye=True)
            monitor.start(session)
            monitor.finish(900, 'interrupted')
            with (session / 'hrv.csv').open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_mock_recording_shares_q_timestamp_and_keeps_preparation_rr(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            monitor = SessionCalibration(target_seconds=.5, maximum_seconds=8,
                                         task_duration=.2, wait_for_eye=True)

            async def press_q_when_ready():
                while True:
                    try:
                        status = json.loads((session / 'hrv_calib_status.json').read_text())
                        if status.get('start_ready'):
                            break
                    except (OSError, ValueError):
                        pass
                    await asyncio.sleep(.02)
                started = datetime.now(timezone.utc).isoformat()
                (session / 'calibration_start.json').write_text(json.dumps({'started_at_utc': started}))
                # Simulate eye finishing before HRV; neither sensor is restarted.
                await asyncio.sleep(.05)
                (session / 'eye_calibration_ready').write_text(datetime.now(timezone.utc).isoformat())
                return started

            async def run():
                _, started = await asyncio.wait_for(asyncio.gather(
                    record_sensor(session_dir=session, monitor=monitor, mock=True),
                    press_q_when_ready()), timeout=12)
                return started

            started = asyncio.run(run())
            self.assertEqual(monitor.result['status'], 'ready')
            self.assertEqual(monitor.result['calibration_started_at_utc'], started)
            with (session / 'rr_raw.csv').open() as stream:
                raw = list(csv.DictReader(stream))
            self.assertEqual(raw[0]['phase'], 'waiting_for_start')
            self.assertTrue(any(row['phase'] == 'calibration' for row in raw))
            for row in raw:
                if datetime.fromisoformat(row['received_at']) < datetime.fromisoformat(started):
                    self.assertEqual(row['phase'], 'waiting_for_start')
            with (session / 'hrv.csv').open() as stream:
                baseline = next(csv.DictReader(stream))
            self.assertEqual(datetime.fromisoformat(baseline['window_start']), datetime.fromisoformat(started))

    def test_packet_immediately_after_q_is_logged_as_calibration(self):
        from hrv_monitor_prototype import record_rr

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            monitor = SessionCalibration(wait_for_eye=True)
            calls = []

            def noise(_mean, _deviation):
                calls.append(1)
                if len(calls) == 2:
                    # Publish Q immediately before the second mock packet callback.
                    (session / 'calibration_start.json').write_text(json.dumps(
                        {'started_at_utc': datetime.now(timezone.utc).isoformat()}))
                return 0

            with patch.object(record_rr.random, 'gauss', side_effect=noise):
                asyncio.run(record_sensor(session_dir=session, monitor=monitor, mock=True, duration=1.8))
            with (session / 'rr_raw.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]['phase'], 'waiting_for_start')
            self.assertEqual(rows[1]['phase'], 'calibration')


if __name__ == '__main__':
    unittest.main()
