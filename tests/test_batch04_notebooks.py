"""Batch-04 contracts on small synthetic recordings; no research-data training."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / f"models/batch-04/exp0{i}/EyeHeartCognitiveLoad.ipynb" for i in (1, 2)
]


def namespace(path) -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "batch04_notebook"}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code" and cell["id"] in {
            "imports",
            "data",
            "preprocessing",
            "storage",
            "engine",
            "tabular",
            "neural",
            "evaluation",
        }:
            exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)  # noqa: S102
    return ns


class Batch04DeliveryTests(unittest.TestCase):
    def test_notebooks_have_executable_cells_and_valid_execution_fields(self):
        for path in NOTEBOOKS:
            self.assertTrue(path.is_file(), f"Notebook belum tersedia: {path}")
            nb = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            ids = [c["id"] for c in nb["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")
                    self.assertTrue(
                        cell["execution_count"] is None
                        or isinstance(cell["execution_count"], int)
                    )
                    self.assertIsInstance(cell["outputs"], list)


class Batch04RuntimeTests(unittest.TestCase):
    ns: ClassVar[dict[str, Any]]
    neural: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in NOTEBOOKS):
            raise unittest.SkipTest("Menunggu notebook batch-04")
        cls.ns = namespace(NOTEBOOKS[0])
        cls.neural = namespace(NOTEBOOKS[1])
        cls.neural["torch"].set_num_threads(1)
        cls.ns["plt"].switch_backend("Agg")

    def fixture(self):
        ns = self.ns
        frames = {}
        for window in (15, 25, 45):
            rows = []
            for session in range(3):
                for i in range(1, 361):
                    end = ns["pd"].Timestamp("2026-01-01", tz="UTC") + ns[
                        "pd"
                    ].Timedelta(seconds=i * 5)
                    row: dict[str, Any] = {
                        c: 1 + (i % 13) / 20 for c in ns["SENSOR_FEATURES"]
                    }
                    row.update(
                        session_id=f"s{session}",
                        participant_id=f"p{session}",
                        window_start=(
                            end - ns["pd"].Timedelta(seconds=window)
                        ).isoformat(),
                        window_end=end.isoformat(),
                        window_seconds=window,
                        cognitive_load_label=ns["LABELS"][(i // 24) % 3],
                    )
                    rows.append(row)
            frames[window] = ns["pd"].DataFrame(rows)
        return frames

    def small(self, ns=None):
        ns = ns or self.ns
        rng = ns["np"].random.default_rng(4)
        d = ns["pd"].DataFrame(
            rng.uniform(0.1, 2, (36, len(ns["SENSOR_FEATURES"]))),
            columns=ns["SENSOR_FEATURES"],
        )
        d["session_id"] = ["s0"] * 18 + ["s1"] * 18
        d["_y"] = ns["np"].tile([0, 1, 2], 12)
        return d

    def test_split_matches_between_notebooks_and_preserves_outer_test(self):
        frames = self.fixture()
        views, manifest, _ = self.ns["prepare_experiment"](frames)
        other, second, _ = self.neural["prepare_experiment"](
            {w: d.sample(frac=1, random_state=9) for w, d in frames.items()}
        )
        self.assertEqual(manifest, second)
        old, _, _ = self.ns["prepare_views"](frames)
        for w, view in views.items():
            self.assertEqual(
                view.loc[view._split == "test", "_end_ns"].tolist(),
                old[w].loc[old[w]._split == "test", "_end_ns"].tolist(),
            )
            for fold in range(5):
                self.assertEqual(
                    view[f"_fold_{fold}"].tolist(), other[w][f"_fold_{fold}"].tolist()
                )
                for part in ("train", "validation", "test"):
                    self.assertEqual(
                        set(self.ns["fold_frame"](view, fold, part)._y), {0, 1, 2}
                    )

    def test_support_does_not_overlap_across_partitions_or_windows(self):
        views, _, _ = self.ns["prepare_experiment"](self.fixture())
        for fold in range(5):
            combined = (
                self.ns["pd"]
                .concat(views.values())
                .drop_duplicates(["session_id", "_end_ns"])
            )
            for _, d in combined.groupby("session_id"):
                for a, b in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ):
                    x = self.ns["fold_frame"](d, fold, a)._end_ns.to_numpy()
                    y = self.ns["fold_frame"](d, fold, b)._end_ns.to_numpy()
                    if len(x) and len(y):
                        self.assertTrue(
                            (abs(x[:, None] - y[None, :]) >= 45_000_000_000).all()
                        )

    def test_centered_negative_hrv_is_valid_and_future_values_do_not_change_fit(self):
        pd, np = self.ns["pd"], self.ns["np"]
        d = pd.DataFrame(
            {"session_id": ["a", "a", "b", "b"], "rmssd_ms": [10.0, 30.0, 100.0, 140.0]}
        )
        p = self.ns["SessionPreprocessor"](["rmssd_ms"], "session_center")
        x = p.fit_transform(d, [0, 1, 0, 1])
        np.testing.assert_allclose(
            x[:, 0], [-0.632455532, 0.632455532, -1.264911064, 1.264911064], atol=1e-8
        )
        before = p.centers_.copy()
        self.assertTrue(
            np.isfinite(
                p.transform(pd.DataFrame({"session_id": ["new"], "rmssd_ms": [1e8]}))
            ).all()
        )
        pd.testing.assert_frame_equal(before, p.centers_)
        self.assertEqual(p.pipeline_.named_steps["imputer"].statistics_[0], 0)

    def test_missing_empty_and_constant_columns_have_finite_fallbacks(self):
        pd, np = self.ns["pd"], self.ns["np"]
        d = pd.DataFrame(
            {
                "session_id": ["a", "a", "b"],
                "rmssd_ms": [1.0, 1.0, 1.0],
                "sdnn_ms": [np.nan] * 3,
            }
        )
        for mode in ("global", "session_center", "session_zscore"):
            p = self.ns["SessionPreprocessor"](["rmssd_ms", "sdnn_ms"], mode)
            p.fit(d, [0, 1, 2])
            x = p.transform(
                pd.DataFrame(
                    {"session_id": ["unknown"], "rmssd_ms": [2.0], "sdnn_ms": [np.nan]}
                )
            )
            self.assertEqual(x.shape, (1, 2))
            self.assertTrue(np.isfinite(x).all())

    def test_svm_predictions_use_base_decisions_and_all_models_roundtrip(self):
        d = self.small()
        for family in (
            "dummy",
            "logistic",
            "svm",
            "hist_gradient",
            "xgboost",
            "catboost",
        ):
            r = self.ns["base_recipe"](family)
            if family == "xgboost":
                r["params"]["n_estimators"] = 3
            elif family == "catboost":
                r["params"]["iterations"] = 3
            elif family == "hist_gradient":
                r["params"]["max_iter"] = 3
            model = self.ns["fit_tabular"](r, d, d.iloc[:9])
            before = self.ns["predict_tabular"](model, d)
            restored = self.ns["load_bundle"](self.ns["dump_bundle"](model))
            after = self.ns["predict_tabular"](restored, d)
            self.ns["np"].testing.assert_array_equal(before["labels"], after["labels"])
            if family == "svm":
                expected = model["estimator"].predict(
                    model["preprocessor"].transform(d)
                )
                self.ns["np"].testing.assert_array_equal(before["labels"], expected)
                self.assertEqual(before["score_type"], "decision_score")
            if family in ("xgboost", "catboost"):
                self.assertTrue(
                    any(
                        name.endswith((".json", ".cbm"))
                        for name in self.ns["native_artifacts"](model)
                    )
                )

    def test_voting_ties_follow_first_ranked_component_not_class_number(self):
        np = self.ns["np"]
        x = self.ns["vote_predictions"](
            [np.array([2, 1]), np.array([0, 0]), np.array([1, 2])]
        )
        np.testing.assert_array_equal(x["labels"], [2, 1])
        self.assertEqual(x["score_type"], "vote_fraction")

    def test_svm_fits_without_deprecated_probability_api(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model = self.ns["fit_tabular"](
                self.ns["base_recipe"]("svm"), self.small(), self.small().iloc[:9]
            )
        self.assertEqual(
            self.ns["predict_tabular"](model, self.small())["score_type"],
            "decision_score",
        )

    def test_fold_resume_and_locked_model_final_evaluation_do_not_refit(self):
        ns = self.ns
        views, _, _ = ns["prepare_experiment"](self.fixture())
        with tempfile.TemporaryDirectory() as tmp:
            store = ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
            )
            records = []
            for group in ("fusion", "eye", "hrv"):
                r = ns["base_recipe"]("logistic", "session_center")
                r["group"] = group
                records.append(
                    ns["evaluate_candidate"](
                        store,
                        views,
                        15,
                        r,
                        "synthetic",
                        ns["fit_tabular"],
                        ns["predict_tabular"],
                    )
                )

            def forbidden_fit(*args, **kwargs):
                raise AssertionError(
                    "Fitting must not occur during resume/final evaluation"
                )

            first = records[0]
            (store.remote / first["id"] / "done.json").unlink()
            resumed = ns["evaluate_candidate"](
                store,
                views,
                15,
                first["recipe"],
                "synthetic",
                forbidden_fit,
                ns["predict_tabular"],
            )
            self.assertEqual(first, resumed)
            finalists = ns["fit_finalists"](
                store,
                views,
                {r["id"]: r for r in records},
                ns["fit_tabular"],
                ns["predict_tabular"],
            )
            before = store.artifact(finalists[0]["id"], "model.pkl")

            def prediction_after_lock(model, frame):
                self.assertIsNotNone(store.read("selection"))
                return ns["predict_tabular"](model, frame)

            table = ns["final_evaluation"](
                store, views, finalists, prediction_after_lock
            )
            self.assertEqual(len(table), 1)
            final_key = "final_" + finalists[0]["id"]
            self.assertFalse(store.read(final_key)["refit"])
            self.assertEqual(store.artifact(final_key, "model.pkl"), before)
            stored = ns["load_bundle"](store.artifact(final_key, "model.pkl"))
            test = ns["fold_frame"](views[15], 0, "test")
            self.assertEqual(
                len(ns["predict_tabular"](stored, test)["labels"]), len(test)
            )

    def test_ranking_uses_mean_then_f1_then_variability(self):
        records: list[dict[str, Any]] = [
            {
                "id": "a",
                "validation": {"accuracy": 0.6, "macro_f1": 0.5},
                "accuracy_std": 0.1,
                "complexity": 1,
            },
            {
                "id": "b",
                "validation": {"accuracy": 0.6, "macro_f1": 0.6},
                "accuracy_std": 0.1,
                "complexity": 1,
            },
            {
                "id": "c",
                "validation": {"accuracy": 0.6, "macro_f1": 0.6},
                "accuracy_std": 0.01,
                "complexity": 1,
            },
        ]
        self.assertEqual(
            [r["id"] for r in sorted(records, key=self.ns["rank_key"])], ["c", "b", "a"]
        )

    def test_store_rejects_corruption_and_incomplete_copy_has_no_done_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "x"}
            )
            store.commit("ok", {"x": 1}, {"value": b"a"})
            self.assertEqual(store.read("ok"), {"x": 1})
            (store.remote / "ok" / "value").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                store.read("ok")
            with (
                patch.object(store, "_copy_verified", side_effect=OSError("offline")),
                self.assertRaises(OSError),
            ):
                store.commit("failed", {"x": 2}, {"value": b"b"})
            self.assertIsNone(store.read("failed"))
            with self.assertRaises(ValueError):
                self.ns["ArtifactStore"](
                    Path(tmp) / "local", Path(tmp) / "remote", {"data": "changed"}
                )

    def test_three_architectures_train_one_epoch_and_restore_probabilities(self):
        ns = self.neural
        d = self.small(ns)
        for architecture in ("mlp", "residual", "transformer"):
            recipe = ns["neural_recipe"](architecture, "session_center")
            recipe["config"]["max_epochs"] = 1
            recipe["fixed_epochs"] = 1
            model = ns["fit_neural"](recipe, d, d.iloc[:9])
            pred = ns["predict_neural"](model, d.iloc[:5])
            restored = ns["load_bundle"](ns["dump_bundle"](model))
            ns["np"].testing.assert_allclose(
                pred["scores"],
                ns["predict_neural"](restored, d.iloc[:5])["scores"],
                atol=1e-6,
            )
            self.assertEqual(model["best_epoch"], 1)
            ns["np"].testing.assert_allclose(pred["scores"].sum(1), 1.0, atol=1e-6)
            self.assertNotIn("val_loss", model["history"][0])

    def test_neural_checkpoint_average_loads_without_another_fit(self):
        ns = self.neural
        views, _, _ = ns["prepare_experiment"](self.fixture())
        recipe = ns["neural_recipe"]("mlp", "session_center")
        recipe["config"]["max_epochs"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            store = ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
            )
            record = ns["evaluate_candidate"](
                store,
                views,
                25,
                recipe,
                "synthetic",
                ns["fit_neural"],
                ns["predict_neural"],
            )
            finalists = ns["fit_finalists"](
                store,
                views,
                {record["id"]: record},
                ns["fit_neural"],
                ns["predict_neural"],
                neural=True,
            )
            ns["final_evaluation"](store, views, finalists, ns["predict_neural"])
            key = "final_" + finalists[0]["id"]
            self.assertFalse(store.read(key)["refit"])
            loaded = ns["load_bundle"](store.artifact(key, "model.pkl"))
            result = ns["predict_neural"](
                loaded, ns["fold_frame"](views[25], 0, "test")
            )
            self.assertEqual(result["score_type"], "probability")
            ns["np"].testing.assert_allclose(result["scores"].sum(1), 1.0, atol=1e-6)

    def test_holdout_values_and_labels_do_not_change_candidate_selection(self):
        ns = self.ns
        views, _, _ = ns["prepare_experiment"](self.fixture())
        changed = {w: d.copy() for w, d in views.items()}
        for d in changed.values():
            mask = d._split.eq("test")
            d.loc[mask, ns["GROUPS"]["fusion"]] = 1e8
            d.loc[mask, "_y"] = (d.loc[mask, "_y"] + 1) % 3
        recipe = ns["base_recipe"]("logistic", "session_center")
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for name, source in (("original", views), ("changed", changed)):
                store = ns["ArtifactStore"](
                    Path(tmp) / name / "local",
                    Path(tmp) / name / "remote",
                    {"data": name},
                )
                results.append(
                    ns["evaluate_candidate"](
                        store,
                        source,
                        15,
                        recipe,
                        "synthetic",
                        ns["fit_tabular"],
                        ns["predict_tabular"],
                    )
                )
            self.assertEqual(ns["rank_key"](results[0]), ns["rank_key"](results[1]))
            self.assertEqual(results[0]["validation"], results[1]["validation"])

    def test_neural_learning_curves_restore_missing_remote_marker_from_verified_cache(
        self,
    ):
        ns = self.neural
        history = (
            b"epoch,train_loss,val_loss,val_accuracy\n1,1.1,1.2,0.3\n2,1.0,1.1,0.4\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
            )
            records: dict[str, Any] = {}
            for window in (15, 25, 45):
                key = f"w{window}_example"
                folds = [f"{key}_f{i}" for i in range(5)]
                for fold_key in folds:
                    store.commit(
                        fold_key, {"id": fold_key}, {"monitor_history.csv": history}
                    )
                records[key] = {
                    "id": key,
                    "window": window,
                    "fold_keys": folds,
                    "recipe": {"config": {}},
                    "validation": {"accuracy": 0.4, "macro_f1": 0.4},
                    "accuracy_std": 0.0,
                    "complexity": 23,
                }
            (store.remote / "w15_example_f0" / "done.json").unlink()
            ns["learning_curves"](store, records)
            for window in (15, 25, 45):
                self.assertTrue(
                    (store.remote / f"learning_curve_{window}s.png")
                    .read_bytes()
                    .startswith(b"\x89PNG")
                )
            self.assertEqual(
                store.artifact("w15_example_f0", "monitor_history.csv"), history
            )
            self.assertTrue((store.remote / "w15_example_f0" / "done.json").is_file())

    def test_neural_artifact_read_does_not_trust_a_false_negative_exists_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.neural["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "x"}
            )
            store.commit("fold", {"id": "fold"}, {"history.csv": b"history"})
            original = Path.exists

            def delayed_metadata(path):
                return False if path.name == "done.json" else original(path)

            with patch.object(Path, "exists", delayed_metadata):
                self.assertEqual(store.artifact("fold", "history.csv"), b"history")

    def test_neural_remote_corruption_is_not_hidden_by_local_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.neural["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "x"}
            )
            store.commit("fold", {"id": "fold"}, {"history.csv": b"history"})
            (store.remote / "fold" / "history.csv").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                store.artifact("fold", "history.csv")

    def test_neural_cached_candidate_rebuilds_only_an_unverifiable_fold(self):
        ns = self.neural
        views, _, _ = ns["prepare_experiment"](self.fixture())
        recipe = self.ns["base_recipe"]("logistic", "session_center")
        with tempfile.TemporaryDirectory() as tmp:
            store = ns["ArtifactStore"](
                Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
            )
            record = ns["evaluate_candidate"](
                store,
                views,
                15,
                recipe,
                "synthetic",
                self.ns["fit_tabular"],
                self.ns["predict_tabular"],
            )
            key = record["fold_keys"][1]
            (store.remote / key / "done.json").unlink()
            local_marker = store.local / key / "done.json"
            if local_marker.is_file():
                local_marker.unlink()
            rebuilt = []

            def actual_fit(working, train, validation):
                rebuilt.append(len(train))
                return self.ns["fit_tabular"](working, train, validation)

            resumed = ns["evaluate_candidate"](
                store,
                views,
                15,
                recipe,
                "synthetic",
                actual_fit,
                self.ns["predict_tabular"],
            )
            self.assertIsNotNone(store.read(key))
            self.assertEqual(len(rebuilt), 1)
            self.assertEqual(record["validation"], resumed["validation"])


if __name__ == "__main__":
    unittest.main()
