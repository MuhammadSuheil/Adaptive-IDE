"""Lightweight regression checks; no full dataset training or Drive access."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / f"models/batch-03/exp0{i}/EyeHeartCognitiveLoad.ipynb" for i in (1, 2)
]


def load_namespace(path, sections):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "batch03_notebook", "__file__": str(path)}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell["id"] in sections:
            exec(
                compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec"),
                namespace,
            )
    return namespace


class Batch03ContractTests(unittest.TestCase):
    def test_notebooks_are_delivered_as_self_contained_executable_cells(self):
        for path in NOTEBOOKS:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Notebook belum dibuat: {path}")
                nb = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(nb["nbformat"], 4)
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), str(path), "exec")
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])


class Batch03RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in NOTEBOOKS):
            raise unittest.SkipTest("Menunggu implementasi notebook")
        cls.ns = load_namespace(
            NOTEBOOKS[0], {"imports", "data", "storage", "tabular", "evaluation"}
        )
        cls.neural = load_namespace(
            NOTEBOOKS[1], {"imports", "data", "storage", "neural", "evaluation"}
        )
        cls.neural["torch"].set_num_threads(1)
        cls.ns["plt"].switch_backend("Agg")

    def fixture(self):
        pd = self.ns["pd"]
        result = {}
        for window in (15, 25, 45):
            rows = []
            for session in range(3):
                for i in range(1, 361):
                    end = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(
                        seconds=i * 5
                    )
                    row = {f: 1.0 + (i % 19) / 20 for f in self.ns["SENSOR_FEATURES"]}
                    row.update(
                        session_id=f"s{session}",
                        participant_id=f"p{session}",
                        window_start=(end - pd.Timedelta(seconds=window)).isoformat(),
                        window_end=end.isoformat(),
                        window_seconds=window,
                        cognitive_load_label=["Low", "Medium", "High"][(i // 24) % 3],
                    )
                    rows.append(row)
            result[window] = pd.DataFrame(rows)
        return result

    def prepared(self):
        return self.ns["prepare_views"](self.fixture())

    def test_reporting_helpers_are_ready_before_first_training_cell(self):
        for path in NOTEBOOKS:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            namespace = {"__name__": "batch03_notebook"}
            for cell in notebook["cells"]:
                if cell["id"] == "screening":
                    break
                if cell["cell_type"] == "code" and cell["id"] in {
                    "imports",
                    "data",
                    "storage",
                    "tabular",
                    "neural",
                    "evaluation",
                }:
                    exec(
                        compile("".join(cell["source"]), cell["id"], "exec"), namespace
                    )
            self.assertTrue(callable(namespace.get("publish_leaderboard")), str(path))

    def test_split_is_row_order_invariant_and_identical_between_notebooks(self):
        views, manifest, _ = self.prepared()
        shuffled = {
            w: df.sample(frac=1, random_state=9) for w, df in self.fixture().items()
        }
        _, other, _ = self.neural["prepare_views"](shuffled)
        self.assertEqual(manifest, other)
        for view in views.values():
            self.assertEqual(
                set(view["_split"]), {"train", "validation", "test", "purged"}
            )

    def test_split_has_no_cross_partition_support_overlap(self):
        views, _, _ = self.prepared()
        pd = self.ns["pd"]
        merged = pd.concat(views.values()).drop_duplicates(["session_id", "_end_ns"])
        for _, group in merged.groupby("session_id"):
            retained = group[group["_split"] != "purged"]
            for left, right in (
                ("train", "test"),
                ("train", "validation"),
                ("validation", "test"),
            ):
                a = retained.loc[retained["_split"] == left, "_end_ns"].to_numpy()
                b = retained.loc[retained["_split"] == right, "_end_ns"].to_numpy()
                if len(a) and len(b):
                    self.assertTrue(
                        (abs(a[:, None] - b[None, :]) >= 45_000_000_000).all()
                    )

    def test_rejects_conflicting_labels_duplicate_endpoints_missing_features_and_bad_times(
        self,
    ):
        for kind in ("conflict", "duplicate", "missing", "time", "label"):
            views = self.fixture()
            if kind == "conflict":
                views[25].loc[0, "cognitive_load_label"] = "High"
            elif kind == "duplicate":
                views[15] = self.ns["pd"].concat([views[15], views[15].iloc[[0]]])
            elif kind == "missing":
                views[15] = views[15].drop(columns=["rmssd_ms"])
            elif kind == "time":
                views[15].loc[0, "window_start"] = "invalid"
            else:
                views[15].loc[0, "cognitive_load_label"] = "unknown"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.ns["prepare_views"](views)

    def test_feature_preprocessing_fits_only_training_and_accepts_empty_columns(self):
        np = self.ns["np"]
        pd = self.ns["pd"]
        x = pd.DataFrame({"rmssd_ms": [1.0, 3.0, np.nan, 5.0], "sdnn_ms": [np.nan] * 4})
        transform = self.ns["make_preprocessor"](
            list(x), {"scaler": "standard", "selector": "all"}
        )
        transformed = transform.fit_transform(x, [0, 1, 2, 0])
        self.assertEqual(transformed.shape, (4, 2))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transform.named_steps["imputer"].statistics_[0], 3.0)
        transform.transform(pd.DataFrame({"rmssd_ms": [1e9], "sdnn_ms": [3.0]}))
        self.assertEqual(transform.named_steps["imputer"].statistics_[0], 3.0)

    def test_balancing_fit_and_prediction_contract_on_synthetic_data(self):
        np = self.ns["np"]
        pd = self.ns["pd"]
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            rng.normal(size=(36, len(self.ns["SENSOR_FEATURES"]))),
            columns=self.ns["SENSOR_FEATURES"],
        )
        labels = np.array([0] * 18 + [1] * 10 + [2] * 8)
        for balance in ("none", "weight", "ros", "smote"):
            recipe = self.ns["base_recipe"]("logistic", "fusion")
            recipe["balance"] = balance
            model = self.ns["fit_tabular"](recipe, frame, labels)
            probabilities = self.ns["predict_tabular"](model, frame.iloc[:5])
            self.assertEqual(probabilities.shape, (5, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            self.assertEqual(
                model["preprocessor"].named_steps["imputer"].statistics_.shape[0], 23
            )

    def test_save_load_resume_and_corruption_do_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local"
            remote = Path(directory) / "remote"
            store = self.ns["ArtifactStore"](
                local, remote, {"data": "abc", "split": "xyz"}
            )
            store.commit("candidate", {"metric": 0.5}, {"array.npy": b"example"})
            result = store.read("candidate")
            self.assertEqual(result["metric"], 0.5)
            (remote / "candidate" / "array.npy").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                store.read("candidate")
            with self.assertRaises(ValueError):
                self.ns["ArtifactStore"](
                    local, remote, {"data": "changed", "split": "xyz"}
                )

    def test_failed_artifact_copy_never_commits_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.ns["ArtifactStore"](
                Path(directory) / "local",
                Path(directory) / "drive",
                {"data": "fixture"},
            )
            write = store._copy_verified

            def failed_copy(target, payload):
                if target.name == "record.json":
                    raise OSError("Simulasi storage tidak tersedia")
                write(target, payload)

            with (
                patch.object(store, "_copy_verified", failed_copy),
                self.assertRaises(OSError),
            ):
                store.commit(
                    "interrupted", {"accuracy": 0.5}, {"array.npy": b"partial"}
                )
            self.assertIsNone(store.read("interrupted"))
            self.assertFalse((store.remote / "interrupted" / "done.json").exists())

    def test_drive_copy_does_not_require_fsync_support(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "os.fsync", side_effect=OSError("Mount tidak menyediakan fsync")
            ):
                store = self.ns["ArtifactStore"](
                    Path(directory) / "local",
                    Path(directory) / "drive",
                    {"data": "fixture"},
                )
                store.commit("verified", {"saved": True}, {"payload.bin": b"verified"})
            self.assertEqual(store.read("verified"), {"saved": True})

    def test_model_bundle_round_trip_and_ensemble_probability_alignment(self):
        np = self.ns["np"]
        pd = self.ns["pd"]
        frame = pd.DataFrame(
            np.random.default_rng(8).normal(size=(36, 26)),
            columns=self.ns["SENSOR_FEATURES"],
        )
        model = self.ns["fit_tabular"](
            self.ns["base_recipe"]("logistic", "eye"), frame, np.arange(36) % 3
        )
        bundle = self.ns["dump_bundle"](model)
        restored = self.ns["load_bundle"](bundle)
        a = self.ns["predict_tabular"](model, frame)
        np.testing.assert_allclose(a, self.ns["predict_tabular"](restored, frame))
        mixed = self.ns["combine_probabilities"]([a, a[:, ::-1]], [0.25, 0.75])
        np.testing.assert_allclose(mixed, 0.25 * a + 0.75 * a[:, ::-1])
        with self.assertRaises(ValueError):
            self.ns["combine_probabilities"]([a, a[:-1]], [0.5, 0.5])

    def test_neural_architectures_one_step_and_round_trip(self):
        ns = self.neural
        torch = ns["torch"]
        np = ns["np"]
        torch.set_num_threads(1)
        for kind in ("mlp", "residual", "transformer"):
            config = {
                "architecture": kind,
                "width": 32,
                "dropout": 0.1,
                "weighted": False,
                "lr": 0.001,
                "seed": 42,
                "group": "eye",
            }
            model = ns["make_network"](config, 21)
            x = torch.randn(6, 21)
            loss = torch.nn.functional.cross_entropy(model(x), torch.arange(6) % 3)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            payload = ns["dump_neural_state"](model)
            other = ns["make_network"](config, 21)
            other.load_state_dict(ns["load_neural_state"](payload))
            model.eval()
            other.eval()
            with torch.no_grad():
                np.testing.assert_allclose(
                    model(x).numpy(), other(x).numpy(), atol=1e-6
                )

    def test_all_tabular_backends_hierarchy_and_feature_variants_fit_tiny_data(self):
        ns = self.ns
        np, pd = ns["np"], ns["pd"]
        frame = pd.DataFrame(
            np.random.default_rng(11).uniform(0.1, 2.0, (60, 26)),
            columns=ns["SENSOR_FEATURES"],
        )
        y = np.arange(60) % 3
        tiny = {
            "logistic": {"C": 0.1},
            "svm": {"C": 1.0},
            "knn": {"n_neighbors": 3},
            "random_forest": {},
            "extra_trees": {},
            "hist_gradient": {"max_iter": 2},
            "xgboost": {"n_estimators": 2},
            "catboost": {"iterations": 2},
        }
        for family, params in tiny.items():
            recipe = ns["base_recipe"](family, "hrv")
            recipe.update(params=params, ratio=True, log=True)
            if family != "knn":
                recipe["balance"] = "weight"
            model = ns["fit_tabular"](recipe, frame, y)
            probabilities = ns["predict_tabular"](model, frame.head())
            self.assertTrue(np.isfinite(probabilities).all(), family)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        for selector in ("mi8", "mi16", "pca"):
            recipe = ns["base_recipe"]("logistic", "eye")
            recipe["selector"] = selector
            model = ns["fit_tabular"](recipe, frame, y)
            self.assertEqual(ns["predict_tabular"](model, frame.head()).shape, (5, 3))
        recipe = ns["base_recipe"]("logistic", "eye")
        recipe["polynomial"] = True
        model = ns["fit_tabular"]({"kind": "hierarchy", "child": recipe}, frame, y)
        probabilities = ns["predict_tabular"](model, frame.head())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_candidate_resume_reads_completed_artifact_without_fitting_again(self):
        ns = self.ns
        view = self.prepared()[0][15]
        train, val = (
            ns["part_frame"](view, "train"),
            ns["part_frame"](view, "validation"),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ns["ArtifactStore"](
                Path(directory) / "local",
                Path(directory) / "drive",
                {"data": "fixture"},
            )
            recipe = ns["base_recipe"]("logistic", "eye")
            record = ns["evaluate_candidate"](store, 15, recipe, train, val, "test")

            def unexpected_fit(*args, **kwargs):
                raise AssertionError("Kandidat selesai dilatih ulang saat resume")

            with patch.dict(ns, {"fit_tabular": unexpected_fit}):
                resumed = ns["evaluate_candidate"](
                    store, 15, recipe, train, val, "test"
                )
            self.assertEqual(resumed, record)

    def test_final_selection_precedes_test_and_refit_excludes_test_rows(self):
        ns = self.ns
        views = self.prepared()[0]
        finalists = [
            {
                "id": f"fixture_{w}",
                "window": w,
                "recipe": ns["base_recipe"]("logistic", "eye"),
                "validation": {"accuracy": score, "macro_f1": score},
                "complexity": 1,
            }
            for w, score in ((15, 0.9), (25, 0.8), (45, 0.7))
        ]

        def refit(record, frame):
            self.assertNotIn("test", set(frame["_split"]))
            self.assertEqual(set(frame["_split"]), {"train", "validation"})
            return ns["fit_tabular"](record["recipe"], frame, frame["_y"].to_numpy())

        with tempfile.TemporaryDirectory() as directory:
            store = ns["ArtifactStore"](
                Path(directory) / "local",
                Path(directory) / "drive",
                {"data": "fixture"},
            )
            with patch.object(ns["plt"], "show", lambda: None):
                table = ns["final_evaluation"](
                    store, views, finalists, refit, ns["predict_tabular"]
                )
            ns["plt"].close("all")
            self.assertEqual(store.read("selection")["primary_id"], "fixture_15")
            self.assertEqual(list(table.loc[table["primary"], "window"]), [15])
            self.assertTrue((store.remote / "RESULTS.md").is_file())
            self.assertIsNotNone(store.read("complete"))

    def test_neural_training_checkpoint_and_refit_work_on_tiny_data(self):
        ns = self.neural
        np, pd = ns["np"], ns["pd"]
        frame = pd.DataFrame(
            np.random.default_rng(15).uniform(0.1, 2.0, (48, 26)),
            columns=ns["SENSOR_FEATURES"],
        )
        frame["_y"] = np.arange(48) % 3
        config = {
            "architecture": "mlp",
            "group": "eye",
            "width": 32,
            "dropout": 0.1,
            "weighted": True,
            "lr": 0.001,
            "seed": 42,
            "max_epochs": 2,
            "patience": 1,
        }
        frame["_y"] = np.array([0] * 18 + [1] * 10 + [2] * 8 + [0, 1, 2] * 4)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="The given NumPy array is not writable.*"
            )
            bundle = ns["fit_neural"](
                config,
                frame.iloc[:36],
                frame["_y"].iloc[:36].to_numpy(),
                frame.iloc[36:],
            )
        self.assertIn(bundle["best_epoch"], (1, 2))
        a = ns["predict_neural"](bundle, frame.iloc[36:])
        restored = ns["load_bundle"](ns["dump_bundle"](bundle))
        np.testing.assert_allclose(
            a, ns["predict_neural"](restored, frame.iloc[36:]), atol=1e-6
        )
        yv = frame["_y"].iloc[36:].to_numpy()
        training_weights = 36 / (3 * np.array([18, 10, 8]))
        expected_loss = np.average(
            -np.log(a[np.arange(12), yv]), weights=training_weights[yv]
        )
        self.assertAlmostEqual(
            bundle["history"][bundle["best_epoch"] - 1]["val_loss"],
            expected_loss,
            places=6,
        )
        record = {
            "recipe": {"kind": "neural", "config": config},
            "best_epoch": bundle["best_epoch"],
        }
        refitted = ns["refit_neural"](record, frame)
        self.assertEqual(len(refitted["history"]), bundle["best_epoch"])

    def test_svm_calibration_does_not_use_deprecated_probability_api(self):
        ns = self.ns
        np, pd = ns["np"], ns["pd"]
        frame = pd.DataFrame(
            np.random.default_rng(17).uniform(0.1, 2.0, (36, 26)),
            columns=ns["SENSOR_FEATURES"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model = ns["fit_tabular"](
                ns["base_recipe"]("svm", "eye"), frame, np.arange(36) % 3
            )
            self.assertEqual(ns["predict_tabular"](model, frame.head()).shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
