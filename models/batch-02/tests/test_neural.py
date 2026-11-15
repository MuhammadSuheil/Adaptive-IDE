"""Forward and support contracts only; no backward pass or optimizer step."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch tidak terpasang")
class ForwardTest(unittest.TestCase):
    def test_tabular_architectures_return_one_three_class_logit_row_per_input(self):
        import torch
        from batch02.networks import FeatureTransformer, TabularMLP

        for cls in (TabularMLP, FeatureTransformer):
            model = cls(8).eval()
            with torch.no_grad():
                result = model(torch.ones(4, 8))
            self.assertEqual(tuple(result.shape), (4, 3))
            self.assertTrue(torch.isfinite(result).all())

    def test_gru_uses_each_actual_sequence_length_not_padding(self):
        import torch
        from batch02.networks import SensorGRU

        model = SensorGRU(3).eval()
        x = torch.zeros(2, 7, 3)
        x[0, 1:] = 1000.0
        with torch.no_grad():
            a = model(x, torch.tensor([1, 7]))
            x[0, 1:] = -1000.0
            b = model(x, torch.tensor([1, 7]))
        np.testing.assert_allclose(a[0].numpy(), b[0].numpy(), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
