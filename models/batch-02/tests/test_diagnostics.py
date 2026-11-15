"""Diagnostic grouping depends on source segments, not predictions."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class DiagnosticTest(unittest.TestCase):
    def test_low_mixed_group_ignores_zero_duration_phase(self):
        from batch02.diagnostics import low_activity_group

        row = SimpleNamespace(
            window_phase_mix='[{"study_phase":"active_task","duration_seconds":15},{"study_phase":"rating","duration_seconds":0}]'
        )
        self.assertEqual(low_activity_group(row), "active_only")
        row.window_phase_mix = '[{"study_phase":"active_task","duration_seconds":10},{"study_phase":"rating","duration_seconds":5}]'
        self.assertEqual(low_activity_group(row), "active_and_nonactive")

    def test_warmup_control_uses_full_stream_not_filtered_endpoint_count(self):
        from batch02.diagnostics import warmup_matrix

        stream = pd.DataFrame(
            {
                "session_id": ["s"] * 4,
                "window_end": pd.date_range(
                    "2026-01-01", periods=4, freq="5s", tz="UTC"
                ),
            }
        )
        result = warmup_matrix(stream, stream.iloc[[3]])
        self.assertEqual(result.history_count.iloc[0], 4)
        self.assertEqual(result.stream_age_seconds.iloc[0], 15)


if __name__ == "__main__":
    unittest.main()
