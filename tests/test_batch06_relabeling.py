"""Exercise the standalone Colab cells with separate synthetic CSV bundles."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

BATCH = Path(__file__).resolve().parents[1] / "models/batch-06"


def notebook(case):
    exp = {"v3": "exp03", "v4": "exp04"}[case]
    path = BATCH / exp / "EyeHeartCognitiveLoad.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    namespace = {}
    for source in code[1:-1]:
        exec(compile(source, str(path), "exec"), namespace)
    return namespace, code[-1]


def fixture(root, case, features):
    proxy = case == "v3"
    package = root / ("v3_behavioral_proxy" if proxy else "v4_self_reported_rating")
    window = package / "window_15s_step_5s"
    window.mkdir(parents=True)
    target = "cognitive_load_label" if proxy else "difficulty_rating"
    labels = ["Low", "Medium", "High"] if proxy else list(range(1, 11))
    rng = np.random.default_rng(17)
    rows = []
    for person, split in enumerate(["Train", "Train", "validation", "Test"]):
        for i, label in enumerate(labels):
            # One participant holds the only examples above the last ordinal threshold.
            if not proxy and label == 10 and person != 0:
                continue
            for tick in range(1 + i % 3):
                rows.append({"participant_id": f"P{person}", "session_id": f"S{person}", "split": split,
                             "episode_id": f"E{person}_{i}", "window_id": f"W{person}_{i}_{tick}",
                             target: label, **dict(zip(features, rng.normal(size=len(features))))})
    data = pd.DataFrame(rows)
    ids = ["participant_id", "session_id", target]
    data[["window_id", *ids, *features]].to_csv(window / "good_acceptable.csv", index=False)
    mapping = data[["window_id", "episode_id", *ids] + (["split"] if proxy else [])]
    mapping.iloc[::-1].to_csv(window / "window_assignments.csv", index=False)
    episodes = data[["episode_id", *ids]].drop_duplicates()
    episodes.to_csv(package / ("episode_labels.csv" if proxy else "episode_ratings.csv"), index=False)
    if proxy:
        data[["participant_id", "split"]].drop_duplicates().to_csv(package / "participant_splits.csv", index=False)
        weights = data.loc[data.split.eq("Train"), ["window_id", "episode_id", "participant_id", target]]
        weights.assign(sample_weight=1.0).iloc[::-1].to_csv(window / "train_sample_weights.csv", index=False)
    return data


class RelabelingTest(unittest.TestCase):
    def run_case(self, root, case):
        ns, run = notebook(case)
        data = fixture(root, case, ns["CORE"])
        ns.update(DATASET_ROOT=root, OUTPUT_ROOT=root / "runs", DOWNLOAD_ZIP=False)
        with patch("builtins.print"):
            exec(run, ns)
        out = ns["archive"].with_suffix("")
        with zipfile.ZipFile(ns["archive"]) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {out.name + "/"} | {
                f"{out.name}/{file.name}" for file in out.iterdir()})
        windows = pd.read_csv(out / "window_predictions.csv")
        episodes = pd.read_csv(out / "episode_predictions.csv")
        probabilities = windows.filter(regex="^p_")
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1))
        self.assertFalse(windows.duplicated(["model", "window_id"]).any())
        expected = windows.groupby(["model", "episode_id"])[probabilities.columns].mean()
        observed = episodes.set_index(["model", "episode_id"])[probabilities.columns]
        self.assertTrue(np.allclose(expected, observed.loc[expected.index]))
        return ns, data, windows, out

    def test_v3_official_split_and_weights(self):
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            ns, data, windows, out = self.run_case(root, "v3")
            selected = json.loads((out / "selection.json").read_text())["model"]
            for split, subset in windows.groupby("split"):
                self.assertEqual(set(subset.window_id), set(data.loc[data.split.eq(split), "window_id"]))
            self.assertEqual(set(windows.loc[windows.split.eq("Test"), "model"]), {selected, "dummy_mode"})
            weight_path = next(root.rglob("train_sample_weights.csv"))
            pd.read_csv(weight_path).iloc[1:].to_csv(weight_path, index=False)
            with self.assertRaisesRegex(ValueError, "bobot"):
                ns["read_bundle"](root, "v3")

    def test_v4_lopo_and_episode_prior(self):
        with tempfile.TemporaryDirectory() as location:
            _, data, windows, out = self.run_case(Path(location), "v4")
            folds = pd.read_csv(out / "folds.csv").set_index("fold")
            self.assertEqual(len(windows), len(data) * 6)
            self.assertTrue(windows.fold.map(folds.held_participant).eq(windows.participant_id).all())
            for person, preds in windows[windows.model.eq("dummy_median")].groupby("participant_id"):
                fitting = data[data.participant_id.ne(person)].drop_duplicates("episode_id")
                counts = fitting.difficulty_rating.value_counts(normalize=True)
                for label in range(1, 11):
                    self.assertTrue(np.allclose(preds[f"p_{label}"], counts.get(label, 0)))
                expected = fitting.difficulty_rating.quantile(0.5, interpolation="lower")
                self.assertTrue(preds.predicted.eq(expected).all())


if __name__ == "__main__":
    unittest.main()
