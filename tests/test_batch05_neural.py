"""Neural checkpoints, interrupted training, and sealed Test evaluation."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from test_batch05_preparation import definitions
from test_batch05_training import training_fixture


class Stage3Availability(unittest.TestCase):
    def test_notebook_provides_stage3_workflow(self):
        self.assertTrue("run_stage3" in definitions(), "Stage 3 is not implemented")


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = definitions()
        if "run_stage3" not in cls.ns:
            raise unittest.SkipTest("Stage 3 is not implemented")

    def setUp(self):
        self.frame = training_fixture(self.ns)
        self.config = dict(self.ns["NEURAL_CONFIG"], epochs=8)

    def test_snapshots_do_not_follow_later_optimizer_updates(self):
        model = self.ns["build_stage3_mlp"]()
        snapshot = self.ns["copy_weights"](model)
        original = {k: v.clone() for k, v in snapshot.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        self.assertTrue(all(torch.equal(snapshot[k], v) for k, v in original.items()))
        self.assertTrue(
            any(
                not torch.equal(snapshot[k], v.cpu())
                for k, v in model.state_dict().items()
            )
        )

    def test_checkpoint_ranking_uses_monitor_not_heldout_scores(self):
        a = {"macro_f1": 0.5, "recall": [0.3, 0.5, 0.6]}
        b = {"macro_f1": 0.6, "recall": [0.2, 0.6, 0.6]}
        rank = self.ns["checkpoint_rank"]
        self.assertLess(rank("loss", a, 0.8, 5), rank("loss", b, 0.9, 4))
        self.assertLess(rank("macro", b, 0.9, 4), rank("macro", a, 0.8, 5))
        self.assertLess(rank("loss", a, 0.8, 4), rank("loss", a, 0.8, 5))

    def test_cross_entropy_is_computed_from_logits_without_probability_clipping(self):
        self.assertTrue("logits_evaluation" in self.ns)
        probabilities, loss = self.ns["logits_evaluation"](
            torch.tensor([[1000.0, -1000.0, -1000.0]]), np.array([2])
        )
        self.assertAlmostEqual(loss, 2000.0)
        self.assertTrue(np.array_equal(probabilities, [[1.0, 0.0, 0.0]]))
        target = np.array([2])
        target.flags.writeable = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.ns["logits_evaluation"](torch.tensor([[0.0, 1.0, 2.0]]), target)
        self.assertEqual(len(caught), 0)

    def test_probability_ensemble_checks_rows_and_uses_all_seeds(self):
        average = self.ns["average_probabilities"]
        a = np.array([[0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
        b = np.array([[0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
        self.assertTrue(
            np.allclose(average([a, b]), [[0.5, 0.425, 0.075], [0.3, 0.4, 0.3]])
        )
        with self.assertRaises(ValueError):
            average([a, b[:1]])
        with self.assertRaises(ValueError):
            average([np.array([[np.nan, 0, 1]])])

    def test_oof_selection_ignores_high_validation_score_and_collapsed_candidate(self):
        base = {
            "finite_predictions": True,
            "recall": [0.4, 0.4, 0.4],
            "fold_std": 0.1,
            "estimators": 1,
            "normalization": "global",
            "source": "matched",
            "policy": None,
        }
        candidates = [
            dict(
                base,
                name="collapsed",
                macro_f1=0.9,
                accuracy=0.99,
                recall=[1.0, 0.0, 0.0],
            ),
            dict(
                base,
                name="oof_winner",
                macro_f1=0.6,
                accuracy=0.6,
                validation_accuracy=0.1,
            ),
            dict(
                base,
                name="validation_winner",
                macro_f1=0.5,
                accuracy=0.7,
                validation_accuracy=0.99,
            ),
        ]
        self.assertEqual(
            self.ns["select_stage3_candidate"](candidates)["name"], "oof_winner"
        )

    def test_monitor_or_test_cannot_be_added_to_optimizer_fitting(self):
        with self.assertRaises(ValueError):
            self.ns["validate_neural_roles"](
                self.frame, np.arange(60), np.arange(50, 90), np.arange(90, 135)
            )
        with self.assertRaises(ValueError):
            self.ns["validate_neural_roles"](
                self.frame, np.arange(60), np.arange(60, 90), np.arange(135, 180)
            )
        ids = np.arange(60, 90)
        changed = self.frame.copy()
        changed.loc[ids, "_y"] = 0
        with self.assertRaises(ValueError):
            self.ns["validate_neural_roles"](
                changed, np.arange(60), ids, np.arange(90, 135)
            )

    def test_resume_restores_optimizer_rng_and_selected_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole = self.ns["Stage3Store"](root, {"data": "synthetic"})
            first = self.ns["neural_job"](
                self.frame,
                np.arange(60),
                np.arange(60, 90),
                np.arange(90, 135),
                "global",
                42,
                whole,
                "neural",
                self.config,
            )
            interrupted = self.ns["Stage3Store"](root, {"data": "synthetic"})

            def interrupt(epoch):
                if epoch == 5:
                    raise RuntimeError("interrupted after a saved epoch")

            with self.assertRaises(RuntimeError):
                self.ns["neural_job"](
                    self.frame,
                    np.arange(60),
                    np.arange(60, 90),
                    np.arange(90, 135),
                    "global",
                    42,
                    interrupted,
                    "neural",
                    self.config,
                    on_epoch=interrupt,
                )
            self.assertIsNone(interrupted.get("neural"))
            continued = self.ns["Stage3Store"](
                root, {"data": "synthetic"}, interrupted.remote
            )
            second = self.ns["neural_job"](
                self.frame,
                np.arange(60),
                np.arange(60, 90),
                np.arange(90, 135),
                "global",
                42,
                continued,
                "neural",
                self.config,
            )
            for policy in ("loss", "macro", "last"):
                self.assertEqual(
                    first["policies"][policy]["epoch"],
                    second["policies"][policy]["epoch"],
                )
                self.assertEqual(
                    first["policies"][policy]["held_metrics"],
                    second["policies"][policy]["held_metrics"],
                )
                a = torch.load(whole.get("neural") / f"{policy}.pt", weights_only=True)
                b = torch.load(
                    continued.get("neural") / f"{policy}.pt", weights_only=True
                )
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertEqual(len(continued.completed), 1)

    def test_test_requires_unchanged_sealed_model_and_is_reused_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.ns["Stage3Store"](Path(directory), {"data": "synthetic"})
            with self.assertRaises(ValueError):
                self.ns["evaluate_stage3_test"](self.frame, store)

            self.ns["stage3_tree_job"](
                self.frame,
                np.arange(90),
                np.arange(90, 135),
                "global",
                store,
                "tree",
                save_model=True,
            )
            descriptor = {
                "kind": "tree",
                "jobs": ["tree"],
                "name": "tree",
                "normalization": "global",
            }
            self.ns["seal_stage3_model"](store, descriptor)
            first = self.ns["evaluate_stage3_test"](self.frame, store)
            second = self.ns["evaluate_stage3_test"](self.frame, store)
            self.assertEqual(first, second)
            self.assertEqual(first["metrics"]["rows"], 45)
            (store.remote / "jobs/tree/preprocessing.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                self.ns["evaluate_stage3_test"](self.frame, store)

    def test_monitor_and_validation_values_do_not_change_optimizer_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stores = [
                self.ns["Stage3Store"](root, {"data": "train_only"}) for _ in range(2)
            ]
            changed = self.frame.copy()
            changed.loc[60:134, self.ns["CORE"]] *= 3
            for frame, store in zip((self.frame, changed), stores):
                self.ns["neural_job"](
                    frame,
                    np.arange(60),
                    np.arange(60, 90),
                    np.arange(90, 135),
                    "global",
                    42,
                    store,
                    "neural",
                    self.config,
                )
            a = torch.load(stores[0].get("neural") / "last.pt", weights_only=True)
            b = torch.load(stores[1].get("neural") / "last.pt", weights_only=True)
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            states = [
                json.loads((store.get("neural") / "preprocessing.json").read_text())
                for store in stores
            ]
            self.assertEqual(states[0], states[1])


if __name__ == "__main__":
    unittest.main()
