"""Contracts for the batch-04 Train/selection/test workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

import test_batch04_notebooks as legacy

NOTEBOOKS = legacy.NOTEBOOKS
namespace = legacy.namespace


class Batch04RevisionTests(unittest.TestCase):
    ns: ClassVar[dict[str, Any]]
    neural: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls):
        cls.ns = namespace(NOTEBOOKS[0])
        cls.neural = namespace(NOTEBOOKS[1])
        cls.neural["torch"].set_num_threads(1)

    def fixture(self):
        helper = legacy.Batch04RuntimeTests()
        type(helper).ns = self.ns
        return helper.fixture()

    def test_all_train_rows_receive_exactly_one_internal_validation_assignment(self):
        ns = self.ns
        views, manifest, _ = ns["prepare_experiment"](self.fixture())
        self.assertEqual(manifest.get("schema_version"), 2)
        for view in views.values():
            train = view.loc[view._split.eq("train")]
            hits = sum(
                train[f"_fold_{i}"].eq("validation").astype(int) for i in range(5)
            )
            self.assertTrue(hits.eq(1).all())
            for i in range(5):
                for part in ("train", "validation"):
                    self.assertEqual(set(ns["fold_frame"](view, i, part)._y), {0, 1, 2})

    def test_selection_validation_is_not_used_in_any_internal_fold(self):
        views, _, _ = self.ns["prepare_experiment"](self.fixture())
        for view in views.values():
            held = view[view._split.isin(["validation", "test"])]
            for i in range(5):
                self.assertFalse(held[f"_fold_{i}"].isin(["train", "validation"]).any())

    def test_relative_features_handle_zero_denominators_without_editing_source(self):
        ns = self.ns
        self.assertIn("sensor_features", ns, "Relative sensor derivation is missing")
        d = ns["pd"].DataFrame(
            {
                "rmssd_ms": [20.0, 10.0],
                "sdnn_ms": [10.0, 0.0],
                "ear_left_mean": [0.3, 0.0],
                "ear_right_mean": [0.1, 0.0],
                "ear_left_std": [0.02, 0.0],
                "ear_right_std": [0.02, 0.0],
                "iris_size_estimate_mean": [0.5, 0.0],
                "iris_size_estimate_std": [0.1, 0.0],
            }
        )
        original = d.copy(deep=True)
        result = ns["sensor_features"](
            d, ["hrv_ratio", "ear_relative_gap", "ear_variation", "iris_variation"]
        )
        ns["np"].testing.assert_allclose(result.iloc[0], [2, 1, 0.1, 0.2])
        self.assertTrue(result.iloc[1].isna().all())
        ns["pd"].testing.assert_frame_equal(d, original)

    def test_shortlist_keeps_each_normalization_despite_lower_scores(self):
        ns = self.ns
        self.assertIn(
            "candidate_shortlist", ns, "Normalization-preserving selection is missing"
        )
        records = []
        for norm, score in [
            ("global", 0.9),
            ("session_center", 0.6),
            ("session_zscore", 0.5),
        ]:
            for family in ("logistic", "svm"):
                records.append(
                    {
                        "id": norm + family,
                        "recipe": {"family": family, "normalization": norm},
                        "validation": {"accuracy": score, "macro_f1": score},
                        "accuracy_std": 0.0,
                        "complexity": 23,
                    }
                )
        selected = ns["candidate_shortlist"](records)
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {r["recipe"]["normalization"] for r in selected}, set(ns["NORMALIZATIONS"])
        )

    def test_neural_epoch_selection_ignores_scoring_validation(self):
        ns = self.neural
        self.assertIn(
            "monitor_column",
            ns["neural_recipe"]("mlp"),
            "Independent epoch-monitor contract is missing",
        )
        views, _, _ = ns["prepare_experiment"](self.fixture())
        train = ns["fold_frame"](views[15], 0, "train")
        validation = ns["fold_frame"](views[15], 0, "validation")
        changed = validation.copy()
        changed[ns["GROUPS"]["fusion"]] = 1e8
        changed["_y"] = (changed._y + 1) % 3
        recipe = ns["neural_recipe"]("mlp", "session_center")
        recipe["config"].update(max_epochs=1)
        recipe["monitor_column"] = "_monitor_0"
        first = ns["fit_neural"](recipe, train, validation)
        second = ns["fit_neural"](recipe, train, changed)
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        self.assertEqual(first["monitor_history"], second["monitor_history"])
        ns["np"].testing.assert_allclose(
            ns["predict_neural"](first, validation)["scores"],
            ns["predict_neural"](second, validation)["scores"],
            atol=1e-6,
        )

    def test_checkpoint_schema_rejects_previous_pipeline_resume(self):
        ns = self.ns
        self.assertNotEqual(ns["CODE_VERSION"], "batch04-1.0.0")
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote"
            remote.mkdir()
            (remote / "run.json").write_text(
                json.dumps({"signature": {"code_version": "batch04-1.0.0"}})
            )
            with self.assertRaises(ValueError):
                ns["ArtifactStore"](
                    Path(tmp) / "local", remote, {"code_version": ns["CODE_VERSION"]}
                )

    def test_outer_and_monitor_support_cannot_cross_scoring_boundaries(self):
        ns = self.ns
        views, _, _ = ns["prepare_experiment"](self.fixture())
        combined = (
            ns["pd"].concat(views.values()).drop_duplicates(["session_id", "_end_ns"])
        )

        def disjoint(left, right):
            for sid, part in left.groupby("session_id"):
                b = right.loc[right.session_id.eq(sid), "_end_ns"].to_numpy()
                if len(b):
                    self.assertTrue(
                        (
                            abs(part._end_ns.to_numpy()[:, None] - b[None, :])
                            >= 45_000_000_000
                        ).all()
                    )

        for a, b in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            disjoint(combined[combined._split.eq(a)], combined[combined._split.eq(b)])
        for fold in range(5):
            disjoint(
                combined[combined[f"_monitor_{fold}"].eq("fit")],
                combined[combined[f"_monitor_{fold}"].eq("monitor")],
            )
            disjoint(
                combined[combined[f"_fold_{fold}"].eq("train")],
                combined[combined[f"_fold_{fold}"].eq("validation")],
            )

    def test_selection_values_do_not_change_internal_fitting_or_oof_predictions(self):
        ns = self.ns
        views, _, _ = ns["prepare_experiment"](self.fixture())
        changed = {w: d.copy() for w, d in views.items()}
        for d in changed.values():
            mask = d._split.eq("validation")
            d.loc[mask, ns["GROUPS"]["fusion"]] = 1e8
            d.loc[mask, "_y"] = (d.loc[mask, "_y"] + 1) % 3
        recipe = ns["base_recipe"]("logistic", "session_center")
        with tempfile.TemporaryDirectory() as tmp:
            scores = []
            predictions = []
            for index, source in enumerate((views, changed)):
                store = ns["ArtifactStore"](
                    Path(tmp) / str(index) / "local",
                    Path(tmp) / str(index) / "remote",
                    {"data": index},
                )
                record = ns["evaluate_candidate"](
                    store,
                    source,
                    15,
                    recipe,
                    "synthetic",
                    ns["fit_tabular"],
                    ns["predict_tabular"],
                )
                scores.append(record["validation"])
                predictions.append(store.artifact(record["id"], "oof_predictions.csv"))
            self.assertEqual(scores[0], scores[1])
            self.assertEqual(predictions[0], predictions[1])

    def test_soft_ensemble_rejects_svm_decision_scores(self):
        ns = self.ns
        score = {
            "labels": ns["np"].array([0]),
            "scores": ns["np"].array([[0.5, 0.4, 0.1]]),
            "score_type": "decision_score",
        }
        with self.assertRaises(ValueError):
            ns["combine_outputs"]([score], "probability_average")

    def test_tabular_selection_compares_real_individual_and_ensemble_models(self):
        ns = self.ns
        views, _, _ = ns["prepare_experiment"](self.fixture())
        with tempfile.TemporaryDirectory() as tmp:
            store = ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
            )
            records = {}
            for normalization in ns["NORMALIZATIONS"]:
                for family in ("logistic", "svm"):
                    recipe = ns["base_recipe"](family, normalization)
                    record = ns["evaluate_candidate"](
                        store,
                        views,
                        15,
                        recipe,
                        "synthetic",
                        ns["fit_tabular"],
                        ns["predict_tabular"],
                    )
                    records[record["id"]] = record
            train_keys = ns["endpoint_keys"](ns["outer_frame"](views[15], "train"))

            def fit_only_train(recipe, train, validation):
                self.assertIsNone(validation)
                self.assertEqual(ns["endpoint_keys"](train), train_keys)
                return ns["fit_tabular"](recipe, train, validation)

            finalists = ns["fit_finalists"](
                store, views, records, fit_only_train, ns["predict_tabular"]
            )
            leaderboard = ns["pd"].read_csv(store.remote / "selection_leaderboard.csv")
            self.assertEqual(
                leaderboard.kind.value_counts().to_dict(),
                {"single": 6, "vote": 1, "probability_average": 1},
            )
            self.assertEqual(finalists[0]["id"], leaderboard.iloc[0].id)
            soft = leaderboard.loc[leaderboard.kind.eq("probability_average")].iloc[0]
            parents = store.read(soft.id)["parents"]
            self.assertTrue(
                all(records[p]["recipe"]["family"] != "svm" for p in parents)
            )

            def forbidden_fit(*args, **kwargs):
                raise AssertionError("Resume must reuse completed full-Train models")

            repeated = ns["fit_finalists"](
                store, views, records, forbidden_fit, ns["predict_tabular"]
            )
            self.assertEqual(finalists, repeated)
            before = store.artifact(finalists[0]["id"], "model.pkl")
            ns["final_evaluation"](store, views, finalists, ns["predict_tabular"])
            self.assertEqual(
                before, store.artifact("final_" + finalists[0]["id"], "model.pkl")
            )

    def test_weighted_relative_models_restore_without_recomputing_training_statistics(
        self,
    ):
        ns = self.ns
        helper = legacy.Batch04RuntimeTests()
        type(helper).ns = ns
        d = helper.small()
        for family in ns["FAMILIES"]:
            recipe = ns["base_recipe"](family, "session_zscore")
            recipe.update(group="relative", balance="weight")
            if family == "xgboost":
                recipe["params"]["n_estimators"] = 2
            elif family == "catboost":
                recipe["params"]["iterations"] = 2
            elif family == "hist_gradient":
                recipe["params"]["max_iter"] = 2
            model = ns["fit_tabular"](recipe, d, None)
            before = ns["predict_tabular"](model, d.iloc[:5])
            restored = ns["load_bundle"](ns["dump_bundle"](model))
            after = ns["predict_tabular"](restored, d.iloc[:5])
            ns["np"].testing.assert_allclose(
                before["scores"], after["scores"], atol=1e-6
            )
            self.assertEqual(len(restored["columns"]), 27)
        nn = self.neural
        recipe = nn["neural_recipe"]("mlp", "session_zscore")
        recipe.update(group="relative", balance="weight", fixed_epochs=1)
        recipe["config"].update(max_epochs=1, label_smoothing=0.05)
        model = nn["fit_neural"](recipe, d, None)
        restored = nn["load_bundle"](nn["dump_bundle"](model))
        nn["np"].testing.assert_allclose(
            nn["predict_neural"](model, d.iloc[:5])["scores"],
            nn["predict_neural"](restored, d.iloc[:5])["scores"],
            atol=1e-6,
        )

    def test_monitor_epoch_choice_is_not_forced_to_last_training_epoch(self):
        ns = self.neural
        views, _, _ = ns["prepare_experiment"](self.fixture())
        d = ns["fold_frame"](views[15], 0, "train")
        recipe = ns["neural_recipe"]("mlp", "session_center")
        recipe["config"].update(max_epochs=3)
        model = ns["fit_neural"](recipe, d, None)
        history = model["monitor_history"]
        best = max(
            history,
            key=lambda r: (r["val_accuracy"], r["val_macro_f1"], -r["val_loss"]),
        )
        self.assertEqual(model["best_epoch"], best["epoch"])
        self.assertEqual(len(model["history"]), model["best_epoch"])


if __name__ == "__main__":
    unittest.main()
