from types import SimpleNamespace
import unittest

from eye_tracking_prototype import CSV_COLUMNS, EyeTrackerApp


class EarCalculationTests(unittest.TestCase):
    def test_calculate_ear_returns_per_eye_and_average(self):
        app = EyeTrackerApp.__new__(EyeTrackerApp)
        app.cfg = SimpleNamespace(
            left_eyelid=[0, 1, 2, 3],
            right_eyelid=[4, 5, 6, 7],
        )
        landmarks = [
            SimpleNamespace(x=0, y=1), SimpleNamespace(x=0, y=-1),
            SimpleNamespace(x=-2, y=0), SimpleNamespace(x=2, y=0),
            SimpleNamespace(x=10, y=2), SimpleNamespace(x=10, y=-2),
            SimpleNamespace(x=6, y=0), SimpleNamespace(x=14, y=0),
        ]

        left, right, average = app.calculate_ear(landmarks)

        self.assertAlmostEqual(left, 0.5, places=5)
        self.assertAlmostEqual(right, 0.5, places=5)
        self.assertAlmostEqual(average, 0.5, places=5)

    def test_csv_columns_include_new_eye_features(self):
        self.assertIn("EAR_Right", CSV_COLUMNS)
        self.assertIn("EAR_Left", CSV_COLUMNS)
        self.assertIn("Eye_State", CSV_COLUMNS)


if __name__ == "__main__":
    unittest.main()
