"""Small independent fixtures for the diagnostic calculations."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("batch01_audit",Path(__file__).with_name("audit_batch01.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_overlap_union_is_not_window_sum(self):
        self.assertEqual(audit.union_seconds([(0,25),(5,30),(40,65)]),55)

    def test_union_nested_intervals_and_exact_boundary(self):
        self.assertEqual(audit.union_seconds([(0,25),(1,5),(25,50)]),50)
        self.assertEqual(audit.union_seconds([]),0)

    def test_metrics_keep_absent_class_policy_explicit(self):
        result=audit.metrics(audit.confusion([0,0,1],[0,0,1]))
        self.assertEqual(result["accuracy"],1)
        self.assertEqual(result["balanced_accuracy"],1)
        self.assertAlmostEqual(result["macro_f1"],2/3)

    def test_majority_classifier_has_low_macro_f1(self):
        result=audit.metrics(audit.confusion([0,1,2,2],[2,2,2,2]))
        self.assertAlmostEqual(result["accuracy"],.5)
        self.assertAlmostEqual(result["balanced_accuracy"],1/3)
        self.assertAlmostEqual(result["macro_f1"],2/9)

    def test_thinning_uses_start_and_end_not_row_stride(self):
        frame=pd.DataFrame({"participant_id":["a"]*3,"session_id":["s"]*3,
                            "start_s":[0,5,25],"end_s":[25,30,50]})
        self.assertEqual(audit.thin_windows(frame).index.tolist(),[0,2])

    def test_recording_probe_separates_time_without_cl_labels(self):
        frame=pd.DataFrame([{"participant_id":person,"session_id":person,
                             "start_s":25*i,"end_s":25*(i+1),"sensor":offset}
                            for person,offset in [("a",0),("b",10)] for i in range(8)])
        result=audit.temporal_centroid_probe(frame,["sensor"])
        self.assertEqual(result["accuracy"],1)
        self.assertEqual(result["train_windows"],8)
        self.assertEqual(result["test_windows"],8)

    def test_confusion_support_and_prediction_axis(self):
        matrix=audit.confusion([0,0,1,2],[0,1,1,1])
        np.testing.assert_array_equal(matrix.sum(axis=1),[2,1,1])
        np.testing.assert_array_equal(matrix.sum(axis=0),[1,3,0])


if __name__=="__main__":
    unittest.main()
