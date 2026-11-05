"""Tests for alignment/denominators that can otherwise silently mislead EDA."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import pandas as pd

import generate_eda as eda


def event(t,kind,puzzle="",outcome=""):
    return {"event_time_ms":t,"event_type":kind,"puzzle_id":puzzle,
            "puzzle_title":puzzle,"difficulty_level":"medium","puzzle_set":"medium",
            "session_id":"original-recording-id","outcome":outcome}


class AlignmentTests(unittest.TestCase):
    def test_half_open_event_boundaries_and_ready_gate(self):
        events=[event(1000,"set_started"),event(1100,"puzzle_started","a"),event(2000,"attempt_submitted","a")]
        labels=eda.sample_context(np.array([499,1000,1100,1999,2000]),events,500)
        self.assertEqual(labels.study_phase.tolist(),["baseline","feedback","active_task","active_task","feedback"])

    def test_cross_question_hrv_is_not_attributed_to_one_question(self):
        events=[event(1000,"set_started"),event(1000,"puzzle_started","a"),event(12000,"puzzle_started","b")]
        context=eda.summarize_segments(eda.interval_segments(1000,26000,events,500))
        self.assertEqual(context["study_phase"],"active_task")
        self.assertEqual(context["difficulty_level"],"medium")
        self.assertEqual(context["puzzle_id"],"mixed")
        frame=pd.DataFrame([{**context,"status":"ready","rmssd_ms":25}])
        level,question=eda.hrv_eligibility(frame)
        self.assertTrue(level.iloc[0]);self.assertFalse(question.iloc[0])

    def test_feedback_crossing_not_eligible_at_level(self):
        events=[event(1000,"set_started"),event(1000,"puzzle_started","a"),event(12000,"attempt_submitted","a")]
        context=eda.summarize_segments(eda.interval_segments(1000,26000,events,500))
        frame=pd.DataFrame([{**context,"status":"ready","rmssd_ms":25}])
        self.assertFalse(eda.hrv_eligibility(frame)[0].iloc[0])

    def test_window_clipping_denominator_excludes_missing_gaze(self):
        source=pd.DataFrame({"timestamp_ms":[1000,2000,3000],"gaze_status":["gaze_outside_screen","face_missing","on_screen"],
                             "gaze_x_raw":[-288,"",1440],"gaze_y_raw":[400,"",900],
                             "gaze_x_smooth":[-288,0,1440],"gaze_y_smooth":[400,0,900],
                             "EAR_Left":[.4,0,.6],"EAR_Right":[.4,0,.6],"face_detected":[True,False,True],
                             "total_blinks":[0,0,1]})
        eye=eda.prepare_eye(source,{"width":2880,"height":1800})
        result=eda.window_features(eye,np.array([1000,4000]),np.array([3000,5000]))
        self.assertEqual(result.eye_frames_count.tolist(),[3,0])
        self.assertAlmostEqual(result.clipping_ratio_valid.iloc[0],.5)
        self.assertAlmostEqual(result.ear_left_mean.iloc[0],.5)
        self.assertEqual(result.blink_count.iloc[0],1)
        self.assertTrue(np.isnan(result.ear_left_mean.iloc[1]))
        self.assertTrue(np.isnan(eye.gaze_x_smooth.iloc[1]))

    def fixture(self,directory,status="ready"):
        directory.mkdir()
        eye=pd.DataFrame({"timestamp_ms":[1000,2000,3000],"session_id":["original-recording-id"]*3,
                          "gaze_status":["on_screen"]*3,"gaze_x_raw":[100,150,200],"gaze_y_raw":[200,250,300],
                          "gaze_x_smooth":[100,150,200],"gaze_y_smooth":[200,250,300],
                          "EAR_Left":[.4,.5,.6],"EAR_Right":[.4,.5,.6],"face_detected":[True]*3,
                          "Eye_State":["Unclassified","Fixation","Fixation"],"total_blinks":[0,0,0]})
        eye.to_csv(directory/"eye_tracking.csv",index=False)
        def iso(t):return pd.Timestamp(t,unit="ms",tz="UTC").isoformat()
        hrv=pd.DataFrame([{"session_id":"original-recording-id","phase":"baseline","window_start":iso(0),"window_end":iso(500),"status":"ready","rmssd_ms":50,"sdnn_ms":60},
                          {"session_id":"original-recording-id","phase":"task","window_start":iso(1000),"window_end":iso(26000),"status":status,"rmssd_ms":999,"sdnn_ms":999}])
        hrv.to_csv(directory/"hrv.csv",index=False)
        pd.DataFrame([event(1000,"set_started"),event(1000,"puzzle_started","a"),event(3000,"set_ended","a")]).to_csv(directory/"task_events.csv",index=False)
        (directory/"hrv_calib_status.json").write_text(json.dumps({"both_calibrated_at_utc":iso(500)}))

    def test_renamed_folder_uses_original_ids_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"renamed-participant"
            self.fixture(directory)
            before={p.name:eda.digest(p) for p in directory.iterdir()}
            session=eda.load_session(directory,5)
            self.assertEqual(session.original_id,"original-recording-id")
            self.assertEqual(session.eye.puzzle_id.tolist(),["a","a",""])
            self.assertEqual(session.questions.puzzle_id.tolist(),["a"])
            self.assertAlmostEqual(session.questions.active_seconds_in_eye_range.iloc[0],2)
            self.assertEqual(before,{p.name:eda.digest(p) for p in directory.iterdir()})

    def test_unready_hrv_numeric_values_remain_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"disconnected"
            self.fixture(directory,status="gap")
            session=eda.load_session(directory,5)
            self.assertTrue(session.hrv.rmssd_ms.isna().all())
            self.assertTrue(session.hrv.sdnn_ms.isna().all())
            self.assertTrue(session.hrv.rmssd_ratio.isna().all())
            self.assertFalse(session.hrv.eligible_level.any())

    def test_unready_baseline_does_not_produce_a_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"bad-baseline"
            self.fixture(directory)
            hrv=pd.read_csv(directory/"hrv.csv")
            hrv.loc[hrv.phase.eq("baseline"),"status"]="unavailable"
            hrv.to_csv(directory/"hrv.csv",index=False)
            session=eda.load_session(directory,5)
            self.assertTrue(np.isnan(session.baseline_rmssd))
            self.assertTrue(session.hrv.rmssd_ratio.isna().all())

    def test_archive_copy_with_post_task_tail_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);primary=root/"Baithar";copy=root/"Baithar - Copy"
            self.fixture(primary)
            eye=pd.read_csv(primary/"eye_tracking.csv");eye["study_phase"]="active_task"
            eye.to_csv(primary/"eye_tracking.csv",index=False)
            shutil.copytree(primary,copy)
            tail=eye.iloc[-1].copy();tail["timestamp_ms"]=4000;tail["study_phase"]="post_task"
            pd.concat([eye,pd.DataFrame([tail])],ignore_index=True).to_csv(copy/"eye_tracking.csv",index=False)
            selected,inventory=eda.select_session_directories(root)
            self.assertEqual([p.name for p in selected],["Baithar"])
            self.assertEqual(len(inventory),2)
            variant=next(r for r in inventory if r["folder"]==copy.name)
            self.assertTrue(variant["is_variant"]);self.assertFalse(variant["selected"])
            self.assertIn("post_task",variant["reason"])
            selected,_=eda.select_session_directories(root,[copy.name])
            self.assertEqual([p.name for p in selected],[copy.name])

    def test_same_id_with_changed_measurement_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);primary=root/"Baithar";copy=root/"Baithar - Copy"
            self.fixture(primary);shutil.copytree(primary,copy)
            eye=pd.read_csv(copy/"eye_tracking.csv");eye.loc[0,"EAR_Left"]=.99
            eye.to_csv(copy/"eye_tracking.csv",index=False)
            with self.assertRaisesRegex(ValueError,"pengukuran berbeda"):
                eda.select_session_directories(root)
            selected,_=eda.select_session_directories(root,[copy.name])
            self.assertEqual(len(selected),1)


if __name__=="__main__":
    unittest.main()
