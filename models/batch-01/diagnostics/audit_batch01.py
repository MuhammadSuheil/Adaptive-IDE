"""Audit archived batch-01 without changing source data or fitting CL models.

Run from the repository root:
    python models/batch-01/diagnostics/audit_batch01.py

Requires numpy, pandas and scipy. Derived tables go to diagnostics/results.
The identity-centroid probe is a recording-identity diagnostic, not a CL score.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fuse_and_analyze import calibration_ready_ms, context_at  # noqa: E402

HERE = Path(__file__).resolve().parent
DEST = HERE / "results"
DATA = ROOT / "analysis_sessions/dataset_fusion/output"
OUTPUTS = HERE.parent / "outputs"
FEATURES = json.loads((ROOT / "models/shared/feature_columns.json").read_text())["features"]
LEVELS = ["easy", "medium", "hard"]
LABELS = ["Low", "Medium", "High"]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def write_table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(DEST / name, index=False)
    return frame


def union_seconds(intervals):
    end = -np.inf
    seconds = 0.0
    for start, stop in sorted(intervals):
        seconds += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return seconds


def confusion(truth, predicted, n=3):
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)), 1)
    return matrix


def metrics(matrix):
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(len(correct)), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(len(correct)), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(len(correct)), where=precision+recall > 0)
    return {"accuracy": correct.sum()/matrix.sum(),
            "balanced_accuracy": recall[support > 0].mean(), "macro_f1": f1.mean()}


def thin_windows(frame):
    selected = []
    for _, group in frame.groupby(["participant_id", "session_id"], sort=True):
        end = -np.inf
        for row in group.sort_values(["start_s", "end_s"]).itertuples():
            if row.start_s >= end - 1e-6:
                selected.append(row.Index)
                end = row.end_s
    return frame.loc[selected].copy()


def temporal_centroid_probe(frame, feature_names):
    """First half trains, later half tests; thin windows and purge shared signal."""
    trains, tests = [], []
    for _, group in thin_windows(frame).groupby("participant_id", sort=True):
        ordered = group.sort_values("end_s")
        cut = len(ordered)//2
        train = ordered.iloc[:cut]
        test = ordered.iloc[cut:]
        test = test[test.start_s >= train.end_s.max() - 1e-6]
        assert len(train) and len(test)
        assert train.end_s.max() <= test.start_s.min() + 1e-6
        trains.append(train)
        tests.append(test)
    train, test = pd.concat(trains), pd.concat(tests)
    medians = train[feature_names].median()
    x = train[feature_names].fillna(medians)
    keep = x.std(ddof=0).gt(1e-12)
    names = x.columns[keep]
    mean, scale = x[names].mean(), x[names].std(ddof=0)
    x = (x[names]-mean)/scale
    t = (test[names].fillna(medians[names])-mean)/scale
    centroids = x.groupby(train.participant_id).mean()
    people = centroids.index.tolist()
    predicted = cdist(t.to_numpy(), centroids.to_numpy()).argmin(axis=1)
    truth = np.array([people.index(p) for p in test.participant_id])
    scores = metrics(confusion(truth, predicted, len(people)))
    return {**scores, "train_windows": len(train), "test_windows": len(test),
            "participants": len(people), "uniform_chance": 1/len(people)}


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder=HERE/"figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    retention=pd.read_csv(DEST/"retention_by_class.csv").set_index("difficulty_level").loc[LEVELS]
    geometry=pd.read_csv(DEST/"window_geometry.csv").groupby(["window_seconds","difficulty_level"]).label_eligible_windows_before_sensor_checks.sum().unstack()
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    x=np.arange(3)
    axes[0].bar(x-.18,retention.active_seconds/60,.34,color="#285f95",label="Durasi aktif pada ledger")
    axes[0].bar(x+.18,retention.retained_union_seconds/60,.34,color="#d18a27",label="Durasi unik terwakili dataset")
    axes[0].set(xticks=x,xticklabels=LABELS,ylabel="Menit, dijumlahkan dari 13 peserta",ylim=(0,180),title="Low lebih singkat dan kehilangan proporsi lebih besar")
    axes[0].legend(fontsize=8,loc="upper left")
    for key,label,color in zip(LEVELS,LABELS,["#285f95","#d18a27","#99456e"]):
        axes[1].plot(geometry.index,geometry[key],"o-",label=label,color=color)
    axes[1].set(xlabel="Panjang kandidat window (detik)",ylabel="Window murni aktif yang memungkinkan",title="Kelayakan label sebelum pemeriksaan sensor",xticks=geometry.index)
    axes[1].legend(fontsize=9)
    fig.text(.02,-.035,"Panel kanan hanya simulasi batas waktu. Belum menghitung fitur HRV/eye baru atau melatih model.",fontsize=9)
    fig.tight_layout()
    fig.savefig(folder/"duration_and_window_geometry.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    variance=pd.read_csv(DEST/"feature_variance.csv").sort_values("participant_eta2",ascending=False).head(8)
    fig,ax=plt.subplots(figsize=(11,5))
    y=np.arange(len(variance))
    ax.barh(y-.17,variance.participant_eta2*100,.32,color="#285f95",label="Kelompok rekaman/peserta (13)")
    ax.barh(y+.17,variance.difficulty_eta2*100,.32,color="#d18a27",label="Kelompok kesulitan (3)")
    ax.set(yticks=y,yticklabels=variance.feature,xlabel="Proporsi variasi oleh perbedaan rata-rata kelompok (%)",xlim=(0,104),title="Perbedaan antarrekaman kuat pada beberapa fitur")
    ax.invert_yaxis()
    ax.legend(loc="lower right",fontsize=9)
    fig.text(.02,-.025,"Analisis deskriptif pada fitur tunggal; jumlah kelompok berbeda. Bukan uji signifikansi atau ukuran akurasi CL.",fontsize=9)
    fig.tight_layout()
    fig.savefig(folder/"recording_and_difficulty_variance.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    sources = sorted(p for p in OUTPUTS.rglob("*") if p.is_file())
    sources += sorted(DATA.glob("*"))
    sources = [p for p in sources if p.is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    dataset_sha = sha(DATA/"good_acceptable.csv")
    archived_identity = json.loads((next((OUTPUTS/"exp12").iterdir())/"identity.json").read_text())
    assert dataset_sha == archived_identity["dataset_sha256"], "Dataset differs from archived batch-01; do not mix versions"
    frame = pd.read_csv(DATA / "good_acceptable.csv")
    all_data = pd.read_csv(DATA / "all_data.csv")
    # Explicit epoch deltas avoid dependence on pandas datetime storage resolution.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for prefix in ["start", "end"]:
        frame[prefix+"_s"] = (pd.to_datetime(frame["window_"+prefix], utc=True)-epoch).dt.total_seconds()
    assert np.allclose(frame.end_s-frame.start_s, 25)
    assert frame.window_id.nunique() == len(frame) == 2294
    people = sorted(frame.participant_id.unique())
    aliases = {p: f"P{i+1:02d}" for i,p in enumerate(people)}
    actual_labels = frame.set_index("window_id").difficulty_level.map(dict(zip(LEVELS, range(3))))
    for row in frame.itertuples():
        mix = json.loads(row.window_phase_mix)
        assert all(p["study_phase"] == "active_task" and p["difficulty_level"] == row.difficulty_level for p in mix)
        assert abs(sum(p["duration_seconds"] for p in mix)-25) <= .01
    eligible_ratio = frame.baseline_usable.astype(bool)
    assert np.allclose(frame.loc[eligible_ratio,"rmssd_baseline_ratio"],
                       frame.loc[eligible_ratio,"rmssd_ms"]/frame.loc[eligible_ratio,"baseline_rmssd_ms"])
    assert frame.loc[~eligible_ratio,"rmssd_baseline_ratio"].isna().all()
    runs = {}
    per_class, fold_selection, fold_checks = [], [], []
    verified = 0
    reference_manifest = None
    for i in range(1,13):
        run = next((OUTPUTS / f"exp{i:02d}").iterdir())
        runs[i] = run
        state = json.loads((run / "summary.json").read_text())
        assert state["status"] == "complete" and state["mode"] == "full"
        manifest = json.loads((run / "fold_manifest.json").read_text())
        assert len(manifest) == 13
        if reference_manifest is None:
            reference_manifest = manifest
        assert manifest == reference_manifest, f"Different fold assignment in exp{i:02d}"
        identities = frame.set_index("window_id").participant_id
        for fold in manifest:
            train_ids, test_ids = set(fold["train_ids"]), set(fold["test_ids"])
            assert not train_ids & test_ids
            train_people = set(identities.loc[list(train_ids)])
            test_people = set(identities.loc[list(test_ids)])
            assert not train_people & test_people and len(test_people) == 1
            for inner in fold["inner"]:
                a,b = set(inner["train_ids"]), set(inner["test_ids"])
                assert not a & b and a | b == train_ids
                assert not set(identities.loc[list(a)]) & set(identities.loc[list(b)])
            fold_checks.append({"experiment":f"exp{i:02d}","fold":fold["fold_id"],"participant_disjoint":True,"inner_disjoint":True})
        predictions = pd.read_csv(run / "predictions.csv")
        saved = pd.read_csv(run / "metrics.csv").set_index("model")
        for model, pred in predictions.groupby("model"):
            assert len(pred) == 2294 and pred.window_id.nunique() == 2294
            assert np.array_equal(pred.y_true.to_numpy(), actual_labels.loc[pred.window_id].to_numpy())
            mat = confusion(pred.y_true, pred.y_pred)
            measured = metrics(mat)
            for key,value in measured.items():
                assert np.isclose(value, saved.loc[model,key], atol=1e-12)
            for cls,label in enumerate(LABELS):
                support, predicted = mat[cls].sum(), mat[:,cls].sum()
                recall = mat[cls,cls]/support
                precision = mat[cls,cls]/predicted if predicted else 0
                per_class.append({"experiment":f"exp{i:02d}","model":model,"label":label,
                                  "correct":mat[cls,cls],"support":support,"predicted":predicted,
                                  "precision":precision,"recall":recall,
                                  "f1":2*precision*recall/(precision+recall) if precision+recall else 0,
                                  "medium_high_true_accuracy":(mat[1,1]+mat[2,2])/mat[1:].sum()})
            verified += 1
        if i == 12:
            selections = json.loads((run / "selected_candidates.json").read_text())
            reference = pd.read_csv(runs[5]/"predictions.csv").query("model == 'logistic_control'")
            for selection in selections:
                pred = predictions[predictions.fold_id == selection["fold_id"]]
                reference_fold = reference[reference.fold_id == selection["fold_id"]]
                a,b = metrics(confusion(pred.y_true,pred.y_pred)), metrics(confusion(reference_fold.y_true,reference_fold.y_pred))
                fold_selection.append({"fold":selection["fold_id"],"participant_alias":aliases[pred.participant_id.iloc[0]],
                                       "inner_selected_macro_f1":selection["score"],"outer_macro_f1_three_classes":a["macro_f1"],
                                       "control_outer_macro_f1_three_classes":b["macro_f1"],"selected_model":selection["candidate"]["pipeline"]["model"]})
    write_table("fold_integrity.csv",fold_checks)
    write_table("class_errors.csv",per_class)
    write_table("selection_transfer.csv",fold_selection)
    retained_all = all_data[all_data.participant_id.isin(people)]
    duration_rows, span_rows, candidates, sensor_rows, set_rows, assembly_checks = [], [], [], [], [], []
    for person in people:
        subset = frame[frame.participant_id == person]
        directory = ROOT/"sessions/saved_session"/str(subset.session_folder.iloc[0])
        events_path, hrv_path = directory/"task_events.csv", directory/"hrv.csv"
        for source in [events_path,hrv_path,directory/"hrv_calib_status.json",directory/"eye_summary_calibration.json",directory/"eye_tracking.csv"]:
            before[str(source.relative_to(ROOT))] = sha(source)
        events = pd.read_csv(events_path,keep_default_na=False)
        events = events[events.session_id.eq(subset.session_id.iloc[0])].sort_values("event_time_ms",kind="stable")
        records = events.to_dict("records")
        hrv = pd.read_csv(hrv_path)
        task = hrv[hrv.phase.eq("task")]
        source_hrv = task.assign(end_key=pd.to_datetime(task.window_end,utc=True)).set_index("end_key")
        matching = source_hrv.loc[pd.to_datetime(subset.window_end,utc=True)]
        for feature in ["rmssd_ms","sdnn_ms"]:
            residual=np.abs(matching[feature].to_numpy()-subset[feature].to_numpy())
            assert residual.max() < 1e-9
            assembly_checks.append({"participant_alias":aliases[person],"feature":feature,
                                    "checked_windows":len(subset),"max_absolute_residual":residual.max()})
        # Independent raw-eye reaggregation at first/middle/last retained windows.
        eye=pd.read_csv(directory/"eye_tracking.csv",usecols=["timestamp_ms","face_detected","EAR_Left","EAR_Right","total_blinks"])
        face=eye.face_detected.astype(str).str.lower().isin(["true","1"])
        increments=eye.total_blinks.diff()
        left=eye.EAR_Left.where(face & eye.EAR_Left.between(0,1))
        right=eye.EAR_Right.where(face & eye.EAR_Right.between(0,1))
        probes=subset.iloc[sorted(set([0,len(subset)//2,len(subset)-1]))]
        error_by_feature={k:[] for k in ["ear_left_mean","ear_left_std","ear_right_mean","ear_right_std","blink_count"]}
        for probe_row in probes.itertuples():
            mask=(eye.timestamp_ms > probe_row.start_s*1000)&(eye.timestamp_ms <= probe_row.end_s*1000)
            expected={"ear_left_mean":left[mask].mean(),"ear_left_std":left[mask].std(ddof=1),
                      "ear_right_mean":right[mask].mean(),"ear_right_std":right[mask].std(ddof=1),
                      "blink_count":increments[mask].sum()}
            for name,value in expected.items():
                difference=abs(value-getattr(probe_row,name))
                assert difference<1e-9,(person,name,difference)
                error_by_feature[name].append(difference)
        for name,errors in error_by_feature.items():
            assembly_checks.append({"participant_alias":aliases[person],"feature":name,
                                    "checked_windows":len(probes),"max_absolute_residual":max(errors)})
        start = pd.Timestamp(task.window_start.iloc[0]).timestamp()*1000
        end = pd.Timestamp(task.window_end.iloc[-1]).timestamp()*1000
        ready = calibration_ready_ms(directory)
        bounds = sorted({start,end,*[float(r["event_time_ms"]) for r in records if start < float(r["event_time_ms"]) < end]})
        spans = []
        for left,right in zip(bounds,bounds[1:]):
            context = context_at((left+right)/2,records,ready)
            if context["study_phase"] == "active_task" and context["difficulty_level"] in LEVELS:
                key=context["difficulty_level"]
                if spans and spans[-1][2] == key and abs(spans[-1][1]-left)<1e-6:
                    spans[-1]=(spans[-1][0],right,key)
                else:
                    spans.append((left,right,key))
        for left,right,key in spans:
            span_rows.append({"participant_alias":aliases[person],"difficulty_level":key,
                              "duration_seconds":(right-left)/1000,"start_ms":left,"end_ms":right})
        for key in LEVELS:
            active = [(a,b) for a,b,k in spans if k==key]
            active_seconds = sum((b-a)/1000 for a,b in active)
            original = retained_all[(retained_all.participant_id == person)&(retained_all.difficulty_level == key)]
            chosen = subset[subset.difficulty_level == key]
            seconds = union_seconds(zip(chosen.start_s,chosen.end_s))
            short_seconds = sum((b-a)/1000 for a,b in active if b-a < 25000)
            duration_rows.append({"participant_alias":aliases[person],"participant_id":person,"difficulty_level":key,
                                  "active_seconds":active_seconds,"active_spans":len(active),
                                  "shorter_than_25s_spans":sum(b-a<25000 for a,b in active),
                                  "shorter_than_25s_seconds":short_seconds,
                                  "pure_25s_windows_before_quality":len(original),"retained_windows":len(chosen),
                                  "retained_union_seconds":seconds,"poor_windows_removed":len(original)-len(chosen)})
            for length in [10,15,25,35,45,60]:
                grid = np.arange(start+length*1000,end+0.01,5000)
                count=sum(int(np.sum((grid>=a+length*1000-0.01)&(grid<=b+0.01))) for a,b in active)
                if length==25:
                    assert count == len(original),(person,key,count,len(original))
                candidates.append({"participant_alias":aliases[person],"difficulty_level":key,
                                   "window_seconds":length,"step_seconds":5,
                                   "label_eligible_windows_before_sensor_checks":count})
        calibration = json.loads((directory/"eye_summary_calibration.json").read_text())
        validations = [a for a in calibration["attempts"] if a.get("phase")=="held_out_validation"]
        validation=validations[-1] if validations else {}
        row = subset.iloc[0]
        sensor_rows.append({"participant_alias":aliases[person],"participant_id":person,
                            "windows":len(subset),"gaze_validation_passed":row.gaze_validation_passed,
                            "gaze_validation_decision":validation.get("decision"),
                            "median_error_px":row.gaze_validation_median_px,"p95_error_px":row.gaze_validation_p95_px,
                            "baseline_usable":row.baseline_usable,"baseline_coverage":row.baseline_coverage,
                            "missing_baseline_ratio_windows":int(subset.rmssd_baseline_ratio.isna().sum()),
                            "mixed_puzzle_windows":int(subset.puzzle_id.eq("mixed").sum()),
                            "observed_fps_median":subset.observed_eye_fps.median()})
        starts=events[events.event_type.eq("set_started") & events.difficulty_level.isin(LEVELS)]
        order=starts.difficulty_level.tolist()
        set_rows.append({"participant_alias":aliases[person],"recorded_set_order":" > ".join(order),
                         "first_order_easy_medium_hard":list(dict.fromkeys(order))==LEVELS})
    durations=write_table("active_duration_and_retention.csv",duration_rows)
    write_table("active_spans.csv",span_rows)
    write_table("window_geometry.csv",candidates)
    write_table("sensor_admission.csv",sensor_rows)
    write_table("task_order.csv",set_rows)
    write_table("sensor_feature_reaggregation.csv",assembly_checks)
    retention = durations.groupby("difficulty_level",sort=False).sum(numeric_only=True).reset_index()
    retention["retained_fraction_of_active_time"]=retention.retained_union_seconds/retention.active_seconds
    write_table("retention_by_class.csv",retention)

    effects, relations = [], []
    design_p = pd.get_dummies(frame.participant_id,dtype=float).to_numpy()
    design_c = pd.get_dummies(frame.difficulty_level,dtype=float).to_numpy()
    design_pc=np.column_stack([design_p,design_c[:,1:]])
    for feature in FEATURES:
        good=frame[feature].notna().to_numpy()
        y=frame.loc[good,feature].to_numpy(float)
        total=float(np.sum((y-y.mean())**2))
        if total<1e-20:
            effects.append({"feature":feature,"constant":True,"participant_eta2":None,
                            "difficulty_eta2":None,"difficulty_partial_eta2_after_participant":None})
            continue
        resid_p=y-design_p[good]@np.linalg.lstsq(design_p[good],y,rcond=None)[0]
        resid_c=y-design_c[good]@np.linalg.lstsq(design_c[good],y,rcond=None)[0]
        resid_pc=y-design_pc[good]@np.linalg.lstsq(design_pc[good],y,rcond=None)[0]
        sse=float(resid_p@resid_p)
        effects.append({"feature":feature,"constant":False,"participant_eta2":1-sse/total,
                        "difficulty_eta2":1-float(resid_c@resid_c)/total,
                        "difficulty_partial_eta2_after_participant":max(0,1-float(resid_pc@resid_pc)/sse)})
    write_table("feature_variance.csv",effects)
    for a,b in [("blink_count","blink_rate_per_minute"),("iris_size_estimate_std","iris_size_delta_std")]:
        valid=frame[[a,b]].dropna()
        multiplier=2.4 if a=="blink_count" else 1.0
        relations.append({"feature_a":a,"feature_b":b,"expected_multiplier":multiplier,
                          "max_absolute_residual":float(np.max(np.abs(valid[b]-multiplier*valid[a]))),
                          "pearson_r":valid[a].corr(valid[b])})
    write_table("redundant_features.csv",relations)
    probe=[]
    for name,features in [("all",FEATURES),("hrv",FEATURES[:3]),("eye",FEATURES[3:])]:
        probe.append({"feature_group":name,**temporal_centroid_probe(frame,features)})
    write_table("recording_identity_probe.csv",probe)
    direction=[]
    for person,group in frame.groupby("participant_id",sort=True):
        for feature in ["rmssd_ms","sdnn_ms","blink_rate_per_minute","ear_left_mean"]:
            medians=group.groupby("difficulty_level")[feature].median()
            if "medium" in medians and "hard" in medians:
                direction.append({"participant_alias":aliases[person],"feature":feature,
                                  "medium_median":medians["medium"],"high_median":medians["hard"],
                                  "high_minus_medium":medians["hard"]-medians["medium"]})
    write_table("within_recording_class_direction.csv",direction)

    code_checks=[]
    expected=json.loads((runs[12]/"identity.json").read_text())["runtime"]["code"]
    for name,digest in expected.items():
        path=ROOT/"models/shared"/name
        actual=sha(path)
        code_checks.append({"path":name,"archived_sha256":digest,"current_sha256":actual,"matches":actual==digest})
    write_table("runtime_source_check.csv",code_checks)
    make_figures()
    # Provenance and source immutability check. Raw sensor files are read only.
    for rel,digest in before.items():
        assert sha(ROOT/rel)==digest, f"Source changed during audit: {rel}"
    result={"audit":"batch01_diagnostic_v1","dataset_sha256":sha(DATA/"good_acceptable.csv"),
            "python":platform.python_version(),"packages":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__},
            "verified_model_metric_rows":verified,"verified_outer_folds":len(fold_checks),
            "source_files_verified":len(before),"source_files_unchanged":True,
            "identity_probe_note":"Recording identity, one session per participant; not CL prediction or participant-independent validation.",
            "window_geometry_note":"Label eligibility only, no new RR/eye features computed; not a new dataset or model score.",
            "source_sha256":before,"script_sha256":sha(Path(__file__))}
    (DEST/"audit_manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="source_sha256"},indent=2))
    print("\nRetention:\n"+retention.to_string(index=False))
    print("\nIdentity probe:\n"+pd.DataFrame(probe).to_string(index=False))
    print("\nTop recording variance:\n"+pd.DataFrame(effects).sort_values("participant_eta2",ascending=False).head(8).to_string(index=False))


if __name__=="__main__":
    main()
