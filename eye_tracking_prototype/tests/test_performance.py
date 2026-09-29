import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.performance import FrameRate, TimerResolution
from modules.metrics import MetricsEngine
from modules.stream import WebcamStream, open_camera


class PerformanceTests(unittest.TestCase):
    def test_variable_intervals_measure_throughput_not_average_reciprocals(self):
        rate = FrameRate()
        for timestamp in (0.0, 0.016, 0.048):
            rate.add(timestamp)
        self.assertAlmostEqual(rate.fps(.048), 2 / .048)
        self.assertEqual(rate.fps(3), 0)

    def test_copy_does_not_hold_capture_lock_or_share_mutable_array(self):
        stream = WebcamStream.__new__(WebcamStream)
        stream.lock = threading.Lock()
        stream.frame_ready = threading.Condition(stream.lock)
        stream.stopped = False
        stream.ret = True
        stream.frame_id = 7
        stream.capture_timestamp = 123.0
        copied = object()
        def copy():
            self.assertFalse(stream.lock.locked())
            return copied
        stream.frame = Mock()
        stream.frame.copy.side_effect = copy
        self.assertEqual(stream.read_latest(6), (True, copied, 7, 123.0))
        self.assertEqual(stream.read_latest(7, timeout=0), (False, None, 7, 0.0))

    def test_negotiated_format_can_differ_even_if_set_succeeds(self):
        with patch('modules.stream.cv2.VideoCapture') as factory:
            cap = factory.return_value
            cap.isOpened.return_value = True
            cap.set.return_value = True
            cap.get.return_value = 0
            cap.getBackendName.return_value = 'DSHOW'
            _, info = open_camera(1, 640, 480, 60, fourcc='MJPG', buffer_size=1)
            self.assertTrue(info['settings']['fourcc']['set_succeeded'])
            self.assertNotEqual(info['settings']['fourcc']['requested'], info['settings']['fourcc']['readback'])

    def test_stop_does_not_release_while_read_thread_is_still_alive(self):
        stream = WebcamStream.__new__(WebcamStream)
        stream.frame_ready = threading.Condition()
        stream.thread = Mock()
        stream.thread.is_alive.return_value = True
        stream.cap = Mock()
        with self.assertRaises(RuntimeError):
            stream.stop()
        stream.cap.release.assert_not_called()

    def test_timer_close_balances_successful_request_once(self):
        timer = TimerResolution(1)
        timer.active = True
        timer.api = Mock()
        timer.close()
        timer.close()
        timer.api.timeEndPeriod.assert_called_once_with(1)


class IrisValidityTests(unittest.TestCase):
    def make_metrics(self):
        return MetricsEngine(SimpleNamespace(dwell_threshold=300, transition_window_sec=5,
                                            iris_baseline_frames=60, iris_baseline_seconds=1.5))

    def test_baseline_duration_is_independent_of_sampling_rate(self):
        for fps in (30, 40, 60):
            metrics = self.make_metrics()
            for i in range(int(1.5 * fps)):
                metrics.update(i * 1000 / fps, None, 0.01)
            self.assertIsNone(metrics.iris_baseline)
            metrics.update(1500, None, 0.01)
            self.assertAlmostEqual(metrics.iris_baseline, .01)

    def test_invalid_frames_restart_baseline_and_never_reuse_delta(self):
        metrics = self.make_metrics()
        metrics.update(0, None, .01)
        metrics.update(1400, None, .5, iris_valid=False)
        metrics.update(1500, None, .01)
        self.assertIsNone(metrics.iris_baseline)
        for timestamp in range(1525, 3001, 25):
            metrics.update(timestamp, None, .01)
        metrics.update(3100, None, .02)
        self.assertAlmostEqual(metrics.iris_delta, .01)
        metrics.update(3200, None, 0)
        self.assertIsNone(metrics.iris_delta)
        metrics.update(3300, None, float('nan'))
        self.assertIsNone(metrics.iris_delta)


if __name__ == '__main__':
    unittest.main()
