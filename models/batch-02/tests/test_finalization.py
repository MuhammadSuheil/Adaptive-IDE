"""Search and evaluation protocol tests with no fitted research estimator."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class SearchProtocolTest(unittest.TestCase):
    def test_default_and_explicit_default_candidates_share_cache_identity(self):
        from batch02.registry import Recipe

        self.assertEqual(Recipe().key, Recipe(params={"C": 1.0}).key)

    def test_adaptive_search_stays_bounded_and_keeps_controls(self):
        from batch02.registry import controls
        from batch02.search import adaptive_search

        class ScoreFixture:
            def score(self, recipe, inner):
                # Controlled observations, not a model trained on fixture data.
                return {
                    "recipe": recipe,
                    "status": "ok",
                    "score": 0.4,
                    "n_features": 4 if not recipe.baseline_ratio else 5,
                    "epoch": None,
                }

        winner, records, budget = adaptive_search(ScoreFixture(), [])
        self.assertLessEqual(budget["effective_recipes"], 120)
        self.assertLessEqual(len(budget["final_pool"]), 24)
        for control in controls():
            self.assertIn(control.key, budget["final_pool"])
        self.assertFalse(winner["recipe"].baseline_ratio)
        self.assertEqual(len({r["recipe"].key for r in records}), len(records))

    def test_joint_inner_validation_uses_identical_endpoints_and_excludes_validation_people(
        self,
    ):
        from batch02.finalization import joint_inner_splits

        people = [f"p{i}" for i in range(9) for _ in range(3)]
        frame = pd.DataFrame(
            {
                "person_id": people,
                "session_id": people,
                "window_end": pd.date_range(
                    "2026-01-01", periods=27, freq="5s", tz="UTC"
                ),
                "y": [0, 1, 2] * 9,
            }
        )
        extended = pd.concat(
            [frame, frame.assign(window_end=frame.window_end + pd.Timedelta(days=1))],
            ignore_index=True,
        )
        a = joint_inner_splits(frame, frame, 42)
        b = joint_inner_splits(extended, frame, 42)
        for first, second in zip(a, b):
            self.assertEqual(
                frame.iloc[first["valid"]].window_end.tolist(),
                extended.iloc[second["valid"]].window_end.tolist(),
            )
            self.assertFalse(
                set(extended.iloc[second["train"]].person_id)
                & set(extended.iloc[second["valid"]].person_id)
            )


if __name__ == "__main__":
    unittest.main()
