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

    def test_normalized_iris_radius_is_invariant_to_uniform_face_scale(self):
        app = EyeTrackerApp.__new__(EyeTrackerApp)
        app.cfg = SimpleNamespace(left_iris_indices=[0, 1, 2, 3, 4],
                                  right_iris_indices=[5, 6, 7, 8, 9],
                                  left_eye_corners=[10, 11], right_eye_corners=[12, 13],
                                  iris_size_mode='eye_width_ratio', inference_w=320, inference_h=240)
        app.calculate_ear = lambda _: (.3, .3, .3)
        points = [(0, 0), (.01, 0), (0, .01), (-.01, 0), (0, -.01),
                  (.2, 0), (.21, 0), (.2, .01), (.19, 0), (.2, -.01),
                  (-.05, 0), (.05, 0), (.15, 0), (.25, 0)]
        def size(scale):
            landmarks = [SimpleNamespace(x=.3 + x * scale, y=.3 + y * scale) for x, y in points]
            return app.get_normalized_eye_vector(landmarks)[1]
        self.assertAlmostEqual(size(1), size(1.5))
        self.assertGreater(size(1), 0)


if __name__ == "__main__":
    unittest.main()
