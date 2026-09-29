import unittest
import numpy as np

from modules.config import Config
from modules.head_gate import HeadGate
from modules.filter import GazeFilter, eyes_are_open


class HeadGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config('config.yaml')
        self.cfg.hp_countdown_seconds = 1
        self.anchors = np.array([[0., -.8], [0., .8], [-.7, 0.], [.7, 0.]])
        self.pose = dict(pitch=2., yaw=5., roll=0., face_width_ratio=.2,
                         center_x_ratio=.5, center_y_ratio=.5)

    def test_acceptance_duration_is_independent_of_fps(self):
        for fps in (20, 40, 60):
            gate = HeadGate(self.cfg)
            for i in range(3 * fps):
                result = gate.update(i / fps, self.anchors, self.pose)
                if result['passed']:
                    break
            self.assertTrue(result['passed'])
            self.assertAlmostEqual(i / fps, 1.5, delta=2 / fps)

    def test_baseline_uses_median_not_last_sample(self):
        gate = HeadGate(self.cfg)
        for i in range(92):
            pose = dict(self.pose, yaw=6. if i >= 90 else 5.)
            result = gate.update(i / 60, self.anchors, pose)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['baseline']['yaw'], 5.)

    def test_brief_dropout_pauses_and_long_dropout_resets(self):
        gate = HeadGate(self.cfg)
        for i in range(51):
            gate.update(i / 60, self.anchors, self.pose)
        hold = gate.hold
        gate.update(51 / 60)
        gate.update(52 / 60)
        self.assertEqual(gate.hold, hold)
        gate.update(53 / 60, self.anchors, self.pose)
        self.assertEqual(gate.hold, hold)
        for i in range(54, 90):
            gate.update(i / 60)
        self.assertEqual(gate.hold, 0)
        self.assertGreater(gate.record['resets'], 0)

    def test_moving_head_inside_oval_does_not_pass(self):
        gate = HeadGate(self.cfg)
        for i in range(180):
            pose = dict(self.pose, yaw=5. * np.sin(i / 5))
            result = gate.update(i / 60, self.anchors, pose)
            self.assertFalse(result['passed'])
        self.assertIn('head_moving', gate.record['rejection_seconds'])

    def test_small_head_inside_oval_reports_fill_failure(self):
        gate = HeadGate(self.cfg)
        result = gate.update(0, self.anchors * .5, self.pose)
        self.assertIn('face_too_small', result['reasons'])

    def test_hysteresis_keeps_near_boundary_valid_only_after_entry(self):
        gate = HeadGate(self.cfg)
        gate.update(0, self.anchors, dict(self.pose, yaw=11.9))
        result = gate.update(.1, self.anchors, dict(self.pose, yaw=12.5))
        self.assertTrue(result['checks'][3])
        fresh = HeadGate(self.cfg)
        self.assertFalse(fresh.update(0, self.anchors, dict(self.pose, yaw=12.5))['checks'][3])


class GazeFilterTests(unittest.TestCase):
    def config(self):
        cfg = Config('config.yaml')
        cfg.adaptive_ema = False
        return cfg

    def test_ema_same_response_after_same_elapsed_time_at_40_and_60_fps(self):
        outputs = []
        for fps in (40, 60):
            filt = GazeFilter(self.config())
            filt.update(0, 0, 0)
            for i in range(1, fps // 2 + 1):
                output = filt.update(100, 100, i / fps)
            outputs.append(output)
        self.assertAlmostEqual(outputs[0][0], outputs[1][0], places=8)

    def test_single_spike_is_rejected_but_sustained_step_is_confirmed(self):
        filt = GazeFilter(self.config())
        filt.update(100, 100, 0)
        self.assertEqual(filt.update(2000, 100, 1 / 60), (100, 100))
        self.assertFalse(filt.sample_accepted)
        filt.update(100, 100, 2 / 60)
        self.assertTrue(filt.sample_accepted)
        for i in range(3, 7):
            result = filt.update(2000, 100, i / 60)
        self.assertTrue(filt.sample_accepted)
        self.assertGreater(result[0], 100)

    def test_invalid_samples_never_update_filter_and_long_gap_resets(self):
        filt = GazeFilter(self.config())
        filt.update(100, 100, 0)
        filt.invalidate(.1)
        self.assertEqual(filt.last_val, (100, 100))
        filt.invalidate(.5)
        self.assertIsNone(filt.last_val)
        self.assertEqual(filt.update(900, 900, .6), (900, 900))

    def test_one_closed_eye_is_invalid_even_if_blink_counting_disabled(self):
        cfg = self.config()
        cfg.blink_enabled = False
        self.assertFalse(eyes_are_open(cfg, .1, .4))
        self.assertFalse(eyes_are_open(cfg, float('nan'), .4))
        self.assertTrue(eyes_are_open(cfg, .3, .4))


if __name__ == '__main__':
    unittest.main()
