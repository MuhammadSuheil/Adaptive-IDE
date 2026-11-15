"""Pipeline boundaries exercised without fitting a real classifier."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class RegistryTest(unittest.TestCase):
    def test_every_experiment_builds_a_deduplicated_recipe_list(self):
        from batch02.registry import recipes_for

        for number in range(1, 14):
            with self.subTest(number=number):
                recipes = recipes_for(f"exp{number:02d}")
                self.assertTrue(recipes)
                self.assertEqual(len({r.key for r in recipes}), len(recipes))

    def test_baseline_has_an_explicit_no_ratio_control(self):
        from batch02.registry import recipes_for

        recipes = recipes_for("exp01")
        self.assertTrue(
            any(r.model == "logistic" and not r.baseline_ratio for r in recipes)
        )

    def test_feature_groups_never_admit_context_columns(self):
        from batch02.registry import feature_group

        features = [
            "rmssd_ms",
            "sdnn_ms",
            "rmssd_baseline_ratio",
            "blink_count",
            "blink_rate_per_minute",
            "ear_left_mean",
            "ear_left_std",
            "eye_state_other_ratio",
        ]
        self.assertEqual(feature_group("F5", features, False), ["rmssd_ms", "sdnn_ms"])
        self.assertNotIn("blink_count", feature_group("F1", features))
        with self.assertRaises(ValueError):
            feature_group("F0", ["participant_id"])


class PreprocessingTest(unittest.TestCase):
    def test_smote_adds_only_training_examples_for_both_minorities(self):
        from batch02.preprocessing import resample

        y = np.repeat([0, 1, 2], [5, 7, 9])
        x = np.column_stack([np.arange(21, dtype=float), np.sin(np.arange(21))])
        source = x.copy()
        sampled, labels, report = resample(x, y, "smote")
        np.testing.assert_array_equal(np.bincount(labels), [9, 9, 9])
        np.testing.assert_array_equal(x, source)
        self.assertEqual(report["added"], 6)
        self.assertEqual(sampled.shape, (27, 2))

    def test_mi_and_pca_transform_new_rows_without_refitting(self):
        from batch02.preprocessing import Preprocessor

        rng = np.random.default_rng(42)
        train = pd.DataFrame(rng.normal(size=(36, 15)))
        for selector in ("mi8", "mi12", "pca"):
            processor = Preprocessor(selector=selector).fit(
                train, np.tile([0, 1, 2], 12)
            )
            result = processor.transform(train.iloc[:2] * 100)
            self.assertEqual(result.shape[0], 2)
            self.assertTrue(np.isfinite(result).all())
            if selector != "pca":
                self.assertEqual(result.shape[1], int(selector[2:]))

    def test_imputation_and_scaling_never_learn_test_values(self):
        from batch02.preprocessing import Preprocessor

        train = pd.DataFrame(
            {"a": [1.0, 3.0, np.nan], "empty": [np.nan] * 3, "constant": [9.0] * 3}
        )
        fitted = Preprocessor().fit(train)
        result = fitted.transform(
            pd.DataFrame(
                {"a": [100.0, np.nan], "empty": [10.0, 20.0], "constant": [0.0, 0.0]}
            )
        )
        self.assertEqual(result.shape, (2, 1))
        self.assertAlmostEqual(result[1, 0], 0.0)
        self.assertEqual(fitted.columns_, ["a"])
        self.assertEqual(fitted.medians_["a"], 2.0)

    def test_log_ratio_zero_is_missing_only_in_derived_representation(self):
        from batch02.preprocessing import representation

        source = pd.DataFrame(
            {"rmssd_ms": [0.0, 3.0], "rmssd_baseline_ratio": [0.0, 2.0]}
        )
        result = representation(source, "N2")
        self.assertTrue(np.isnan(result.loc[0, "rmssd_baseline_ratio"]))
        self.assertEqual(source.loc[0, "rmssd_baseline_ratio"], 0.0)

    def test_class_weights_follow_medium_when_medium_is_smallest(self):
        from batch02.preprocessing import training_weights

        y = np.array([0, 0, 0, 1, 2, 2])
        weights = training_weights(y, "weight")
        np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2, 1, 1])

    def test_resampling_strategy_targets_all_minorities(self):
        from batch02.preprocessing import sampling_strategy

        self.assertEqual(sampling_strategy([0, 0, 0, 1, 2, 2]), {1: 3, 2: 3})


class PairingTest(unittest.TestCase):
    def test_intersection_pairs_utc_endpoints_not_length_specific_ids(self):
        from batch02.splits import paired_endpoints

        a = pd.DataFrame(
            {
                "session_id": ["s", "s"],
                "window_end": ["2026-01-01T07:00:00+07:00", "2026-01-01T00:00:05Z"],
                "y": [0, 1],
            }
        )
        b = pd.DataFrame(
            {
                "session_id": ["s", "s"],
                "window_end": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:05Z"],
                "y": [0, 2],
            }
        )
        paired = paired_endpoints({15: a, 25: b})
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired.iloc[0].session_id, "s")


class RunnerTest(unittest.TestCase):
    def test_pending_release_stops_before_estimator_factory(self):
        import tempfile

        from batch02.contracts import Config
        from batch02.runner import run_experiment

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises((ValueError, FileNotFoundError)),
        ):
            run_experiment(Config(Path(directory)), "exp01")


if __name__ == "__main__":
    unittest.main()
