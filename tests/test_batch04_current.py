"""Contracts for the standalone batch-04 notebooks; no research-data training."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def definitions(experiment="exp01"):
    path = ROOT / f"models/batch-04/{experiment}/EyeHeartCognitiveLoad.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "EXPERIMENT": experiment}
    for cell in notebook["cells"]:
        if cell.get("id") in {
            "imports",
            "data",
            "features",
            "storage",
            "models",
            "neural",
            "search",
            "evaluation",
        }:
            exec(  # noqa: S102 -- execute reviewed repository notebooks only
                compile("".join(cell["source"]), f"{experiment}/{cell['id']}", "exec"),
                scope,
            )
    return scope


def synthetic(sessions=6, duration=2400):
    core = [
        "rmssd_ms",
        "sdnn_ms",
        "blink_rate_per_minute",
        "ear_left_mean",
        "ear_left_std",
        "ear_right_mean",
        "ear_right_std",
        "iris_size_estimate_mean",
        "iris_size_estimate_std",
        "iris_size_delta_mean",
    ]
    frames = {}
    for window in (15, 25, 45):
        rows = []
        for sid in range(sessions):
            for second in range(45, duration, 5):
                end = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(
                    seconds=second
                )
                label = ["Low", "Medium", "High"][(second // 240 + sid) % 3]
                row = dict(
                    zip(
                        core,
                        [
                            30 + sid + second / 1000,
                            35 + sid,
                            12.0,
                            0.3 + sid / 100,
                            0.03,
                            0.31 + sid / 100,
                            0.04,
                            0.24,
                            0.004,
                            -0.002,
                        ],
                    )
                )
                row.update(
                    session_id=f"s{sid}",
                    participant_id=f"p{sid}",
                    window_end=end.isoformat(),
                    window_start=(end - pd.Timedelta(seconds=window)).isoformat(),
                    window_seconds=window,
                    cognitive_load_label=label,
                    study_phase="active_task",
                    quality_tier="good",
                    head_pitch_mean=1.0,
                    head_pitch_std=0.5,
                    head_yaw_mean=2.0,
                    head_yaw_std=0.6,
                )
                row["window_id"] = (
                    f"{row['session_id']}|{row['window_start']}|{row['window_end']}"
                )
                rows.append(row)
        frames[window] = pd.DataFrame(rows)
    return frames


class NotebookContracts(unittest.TestCase):
    def test_four_independent_notebooks_have_no_executed_research_output(self):
        for i in range(1, 5):
            path = ROOT / f"models/batch-04/exp0{i}/EyeHeartCognitiveLoad.ipynb"
            self.assertTrue(path.is_file(), path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell["source"]), str(path), "exec")

    def test_missing_core_is_rejected_instead_of_silently_imputed(self):
        ns = definitions()
        data = synthetic(sessions=1, duration=100)
        data[15].loc[0, "rmssd_ms"] = np.nan
        with self.assertRaisesRegex(ValueError, "core"):
            ns["prepare_data"](data, candidates=2)

    def test_shortlist_controls_keep_distinct_model_families(self):
        ns = definitions()
        records = [
            {
                "id": str(i),
                "recipe": {"family": family},
                "validation": {
                    "accuracy": 1 - i / 10,
                    "macro_f1": 0.5,
                    "minimum_recall": 0.3,
                },
            }
            for i, family in enumerate(("logistic", "logistic", "hist", "svm"))
        ]
        leaders = ns["family_leaders"](records, 2)
        self.assertEqual([r["recipe"]["family"] for r in leaders], ["logistic", "hist"])

    def test_missing_optional_head_does_not_invalidate_the_core_branch(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=100)[15], 15)
        self.assertEqual(ns["available_groups"](frame), ["core", "core_head"])
        frame.loc[0, "head_pitch_mean"] = np.nan
        self.assertEqual(ns["available_groups"](frame), ["core"])
        frame["head_pitch_mean"] = frame.head_pitch_mean.astype(object)
        frame.loc[0, "head_pitch_mean"] = "unavailable"
        transformed = (
            ns["SensorTransform"](ns["base_recipe"]("logistic"))
            .fit(frame, [0, 1, 2])
            .transform(frame, [0])
        )
        self.assertTrue(np.isfinite(transformed).all())

    def test_wrong_quality_view_is_rejected_before_training(self):
        ns = definitions()
        frame = synthetic(sessions=1, duration=100)[15]
        frame.loc[0, "quality_tier"] = "poor"
        with self.assertRaisesRegex(ValueError, "good_acceptable"):
            ns["validate_frame"](frame, 15)

    def test_slicing_frame_does_not_duplicate_full_feature_cache(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        pre = ns["SensorTransform"](ns["base_recipe"]("logistic")).fit(
            frame, np.arange(140)
        )
        before = pre.transform(frame, [0, 1])
        self.assertTrue(frame.attrs["_cl_features"])
        subset = frame.iloc[[0, 1]]
        self.assertFalse(subset.attrs.get("_cl_features"))
        np.testing.assert_allclose(before, pre.transform(subset, [0, 1]))

    def test_setup_and_resume_work_in_project_paths_with_spaces(self):
        for number in range(1, 5):
            experiment = f"exp0{number}"
            with (
                self.subTest(experiment=experiment),
                tempfile.TemporaryDirectory() as temp,
            ):
                path = (
                    ROOT / f"models/batch-04/{experiment}/EyeHeartCognitiveLoad.ipynb"
                )
                cells = {
                    c["id"]: "".join(c["source"])
                    for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
                }
                scope = {"__name__": "__main__"}
                exec(  # noqa: S102 -- execute reviewed local notebook cells only
                    compile(cells["config"], str(path), "exec"), scope
                )
                project = Path(temp) / "Research Project With Spaces"
                scope["PROJECT_ROOT"] = project
                scope["DATASET_ROOT"] = project / "Dataset/v2_session_phase"
                scope["DATASET_ROOT"].mkdir(parents=True)
                exec(  # noqa: S102 -- execute reviewed local notebook cells only
                    compile(cells["setup"], str(path), "exec"), scope
                )
                ns = definitions(experiment)
                store = ns["start_run"](
                    project, experiment, {"15": "example"}, {"seed": 42}
                )
                store.commit("fold_0", {"rows": 3})
                resumed = ns["start_run"](
                    project,
                    experiment,
                    {"15": "example"},
                    {"seed": 42},
                    str(store.remote),
                )
                self.assertEqual(resumed.read("fold_0"), {"rows": 3})
                self.assertTrue(store.remote.is_relative_to(project))

    def test_split_is_sensor_value_blind_and_purged(self):
        ns = definitions()
        frames = synthetic()
        views, manifest, _ = ns["prepare_data"](frames, candidates=8)
        changed = {
            w: d.sample(frac=1, random_state=19).assign(rmssd_ms=500.0)
            for w, d in frames.items()
        }
        _, other, _ = ns["prepare_data"](changed, candidates=8)
        self.assertEqual(manifest, other)
        for frame in views.values():
            for sid, group in frame.groupby("session_id"):
                for left, right in [
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ]:
                    a = group.loc[group._split.eq(left), "_end"].to_numpy()
                    b = group.loc[group._split.eq(right), "_end"].to_numpy()
                    if len(a) and len(b):
                        self.assertGreaterEqual(
                            np.min(np.abs(a[:, None] - b)), 105_000_000_000, sid
                        )
            for repeat in range(2):
                hits = sum(
                    frame[f"_cv_{repeat}_{fold}"].eq("validation").astype(int)
                    for fold in range(5)
                )
                self.assertTrue(hits.loc[frame._split.eq("train")].eq(1).all())

    def test_context_stops_at_gaps_and_never_reads_future_or_another_session(self):
        ns = definitions()
        frame = synthetic(sessions=2, duration=90)[15]
        frame = frame.drop(index=3).reset_index(drop=True)
        frame = ns["validate_frame"](frame, 15)
        context = ns["history_indices"](frame, 60)
        for i, indices in enumerate(context):
            self.assertEqual(indices[-1], i)
            self.assertTrue((frame.iloc[indices]._end <= frame.iloc[i]._end).all())
            self.assertEqual(frame.iloc[indices].session_id.nunique(), 1)
            if len(indices) > 1:
                self.assertTrue(
                    (np.diff(frame.iloc[indices]._end) == 5_000_000_000).all()
                )

    def test_preprocessing_fits_training_only_and_preserves_signed_deviations(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=100)[15], 15)
        recipe = ns["base_recipe"]("logistic")
        recipe.update(normalization="session_center")
        pre = ns["SensorTransform"](recipe).fit(frame, np.array([0, 1, 2]))
        original = pre.transform(frame, np.array([0, 1, 2]))
        other = frame.copy()
        other.loc[3:, "rmssd_ms"] = 100000
        np.testing.assert_allclose(original, pre.transform(other, np.array([0, 1, 2])))
        self.assertTrue(np.isfinite(original).all())
        self.assertLess(original[:, 0].min(), 0)

    def test_causal_features_are_unchanged_when_future_values_or_labels_change(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=200)[15], 15)
        recipe = ns["base_recipe"]("logistic")
        recipe.update(normalization="causal", history=30, representation="history")
        pre = ns["SensorTransform"](recipe).fit(frame, np.arange(12))
        first = pre.transform(frame, np.arange(12))
        other = frame.copy()
        other.loc[12:, ns["CORE"]] = 10000
        other["cognitive_load_label"] = "High"
        np.testing.assert_allclose(first, pre.transform(other, np.arange(12)))

    def test_all_tabular_families_save_reload_and_align_class_scores(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        train = np.arange(0, 140)
        test = np.arange(140, len(frame))
        for family in ns["TABULAR_FAMILIES"]:
            with self.subTest(family=family):
                recipe = ns["base_recipe"](family)
                if family in ("forest", "extra", "hist", "xgboost", "catboost"):
                    recipe["params"].update(n_estimators=5) if family in (
                        "forest",
                        "extra",
                        "xgboost",
                    ) else recipe["params"].update(
                        max_iter=5
                    ) if family == "hist" else recipe["params"].update(iterations=5)
                model = ns["fit_model"](recipe, frame, train)
                before = ns["predict_model"](model, frame, test)
                restored = ns["unpack_model"](ns["pack_model"](model))
                after = ns["predict_model"](restored, frame, test)
                np.testing.assert_array_equal(before["labels"], after["labels"])
                self.assertEqual(before["scores"].shape, (len(test), 3))

    def test_ordinal_hierarchy_and_resampling_are_fitted_only_on_training(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        for objective in ("ordinal", "hierarchy"):
            recipe = ns["base_recipe"]("logistic")
            recipe.update(objective=objective)
            model = ns["fit_model"](recipe, frame, np.arange(140))
            pred = ns["predict_model"](model, frame, np.arange(140, len(frame)))
            np.testing.assert_allclose(pred["scores"].sum(axis=1), 1)
            self.assertTrue((pred["scores"] >= 0).all())
        for balance in ("weight", "ros", "smote"):
            recipe = ns["base_recipe"]("logistic")
            recipe["balance"] = balance
            self.assertEqual(
                ns["predict_model"](
                    ns["fit_model"](recipe, frame, np.arange(140)), frame, [140]
                )["scores"].shape,
                (1, 3),
            )

    def test_checkpoint_corruption_and_incompatible_resume_are_not_ignored(self):
        ns = definitions()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            store.commit("fold_0", {"score": 0.5}, {"payload.bin": b"content"})
            self.assertEqual(store.read("fold_0")["score"], 0.5)
            (root / "remote/fold_0/payload.bin").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                store.read("fold_0")
            with self.assertRaises(ValueError):
                ns["ArtifactStore"](root / "other", root / "remote", {"test": 2})

    def test_model_bundle_does_not_capture_dataset_metadata_after_reload(self):
        ns = definitions()
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        frame["participant_note"] = "TEST_SECRET_METADATA_NOT_A_MODEL_INPUT"
        model = ns["fit_model"](ns["base_recipe"]("logistic"), frame, np.arange(140))
        restored = ns["unpack_model"](ns["pack_model"](model))
        ns["predict_model"](restored, frame, [140, 141])
        self.assertNotIn(
            b"TEST_SECRET_METADATA_NOT_A_MODEL_INPUT", ns["pack_model"](restored)
        )

    def test_interrupted_remote_copy_recovers_from_complete_local_checkpoint(self):
        ns = definitions()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            real_copy = store._copy_verified

            def disconnected(target, payload):
                if target.is_relative_to(store.remote):
                    raise OSError("simulated Drive interruption")
                return real_copy(target, payload)

            store._copy_verified = disconnected
            with self.assertRaisesRegex(OSError, "Drive interruption"):
                store.commit("fold_0", {"score": 0.5}, {"payload.bin": b"content"})
            store._copy_verified = real_copy
            self.assertEqual(store.read("fold_0"), {"score": 0.5})
            self.assertEqual(store.artifact("fold_0", "payload.bin"), b"content")
            self.assertTrue((store.remote / "fold_0/done.json").is_file())

    def test_temporal_selection_preserves_sequence_and_current_controls(self):
        ns = definitions("exp04")
        frames, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        frame = frames[15]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            records = []
            for family in ("gru", "mlp"):
                recipe = ns["base_recipe"](family)
                recipe["network"].update(width=16, depth=1, max_epochs=2, patience=2)
                recipe["history"] = 15 if family == "gru" else 0
                records.append(
                    ns["evaluate_recipe"](store, frame, 15, recipe, repeats=(0, 1))
                )
            lock = ns["select_models"](store, {15: frame}, {15: records}, "exp04")
            roles = lock["windows"]["15"]["roles"]
            self.assertIn("sequence_control", roles)
            self.assertIn("current_control", roles)
            self.assertIn("dummy_control", roles)
            table = ns["evaluate_test"](store, {15: frame}, lock)
            self.assertEqual(int(table.primary.sum()), 1)
            self.assertIsNotNone(store.read("complete"))

    def test_all_neural_architectures_fit_and_roundtrip_on_synthetic_data(self):
        ns = definitions("exp03")
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        for family in ns["NEURAL_FAMILIES"]:
            with self.subTest(family=family):
                recipe = ns["base_recipe"](family)
                recipe["network"].update(width=32, depth=1, max_epochs=2)
                recipe["history"] = 15 if family in ("gru", "tcn") else 0
                model = ns["fit_model"](recipe, frame, np.arange(140), epochs=2)
                output = ns["predict_model"](model, frame, np.arange(140, len(frame)))
                restored = ns["unpack_model"](ns["pack_model"](model))
                again = ns["predict_model"](restored, frame, np.arange(140, len(frame)))
                np.testing.assert_allclose(output["scores"], again["scores"], atol=1e-7)
                np.testing.assert_allclose(output["scores"].sum(axis=1), 1, atol=1e-6)

    def test_completed_tuning_resumes_without_fitting_again(self):
        ns = definitions()
        frames, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        frame = frames[15]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            first = ns["optimize"](
                store, frame, 15, "resume_test", ["logistic"], 1, "exp01"
            )

            def forbidden_fit(*args, **kwargs):
                raise AssertionError("Completed folds must not fit again.")

            ns["fit_model"] = forbidden_fit
            second = ns["optimize"](
                store, frame, 15, "resume_test", ["logistic"], 1, "exp01"
            )
            self.assertEqual(first, second)

    def test_full_selection_stacking_and_test_use_locked_models(self):
        ns = definitions()
        frames, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        frame = frames[15]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            records = []
            for family in ("logistic", "hist"):
                recipe = ns["base_recipe"](family)
                if family == "hist":
                    recipe["params"]["max_iter"] = 5
                records.append(
                    ns["evaluate_recipe"](store, frame, 15, recipe, repeats=(0, 1))
                )
            real_publish = store.publish

            def interrupted_publish(name, payload):
                if name == "selection_leaderboard.csv":
                    raise OSError("simulated selection export interruption")
                return real_publish(name, payload)

            store.publish = interrupted_publish
            with self.assertRaisesRegex(OSError, "selection export interruption"):
                ns["select_models"](store, {15: frame}, {15: records}, "exp01")
            self.assertIsNone(store.read("selection"))
            store.publish = real_publish
            lock = ns["select_models"](store, {15: frame}, {15: records}, "exp01")
            payload = store.artifact(lock["primary"], "model.pkl")
            table = ns["evaluate_test"](store, {15: frame}, lock)
            self.assertEqual(int(table.primary.sum()), 1)
            self.assertEqual(payload, store.artifact(lock["primary"], "model.pkl"))
            self.assertIsNotNone(store.read("selection_w15_stack"))
            self.assertIsNotNone(store.read("complete"))
            self.assertEqual(
                ns["select_models"](store, {15: frame}, {15: records}, "exp01"), lock
            )

    def test_interrupted_trial_reuses_completed_folds(self):
        ns = definitions()
        frames, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        frame = frames[15]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "remote", {"test": 1})
            real_fit = ns["fit_model"]
            calls = []

            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated runtime interruption")
                return real_fit(*args, **kwargs)

            ns["fit_model"] = interrupted
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                ns["optimize"](
                    store, frame, 15, "interrupted", ["logistic"], 1, "exp01"
                )
            calls.clear()

            def counted(*args, **kwargs):
                calls.append(1)
                return real_fit(*args, **kwargs)

            ns["fit_model"] = counted
            records = ns["optimize"](
                store, frame, 15, "interrupted", ["logistic"], 1, "exp01"
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
