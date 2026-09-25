import unittest

from modules.eye_state import EyeStateClassifier


class EyeStateClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = EyeStateClassifier(
            fixation_radius_px=15.0,
            fixation_frames=6,
            saccade_threshold_px=80.0,
        )

    def test_stable_six_frames_are_fixation(self):
        states = [self.classifier.update(100 + offset, 200, True) for offset in (0, 2, -2, 1, -1, 0)]
        self.assertEqual(states[:-1], [EyeStateClassifier.UNCLASSIFIED] * 5)
        self.assertEqual(states[-1], EyeStateClassifier.FIXATION)

    def test_large_consecutive_jump_is_saccade(self):
        self.classifier.update(100, 100, True)
        self.assertEqual(
            self.classifier.update(180, 100, True),
            EyeStateClassifier.SACCADE,
        )

    def test_invalid_sample_is_unclassified_and_resets_fixation(self):
        for _ in range(5):
            self.assertEqual(self.classifier.update(100, 100, True), EyeStateClassifier.UNCLASSIFIED)
        self.assertEqual(self.classifier.update(0, 0, False), EyeStateClassifier.UNCLASSIFIED)
        self.assertEqual(self.classifier.update(100, 100, True), EyeStateClassifier.UNCLASSIFIED)


if __name__ == "__main__":
    unittest.main()
