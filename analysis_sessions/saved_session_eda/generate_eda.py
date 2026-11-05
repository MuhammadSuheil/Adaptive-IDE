"""EDA mata/HRV terhadap waktu, fase, dan soal; sumber tidak pernah ditulis ulang.

python analysis_sessions/saved_session_eda/generate_eda.py
python analysis_sessions/saved_session_eda/generate_eda.py --subjects ikhsan Shakilla
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Only pure phase helpers are imported. Never call fuse_session/annotation writers.
from fuse_and_analyze import context_at, interval_segments, summarize_segments
from analysis_sessions.session_inventory import select_session_directories, participant_group

LEVELS = ["easy", "medium", "hard"]
COLORS = {"easy": "#30a46c", "medium": "#e5a000", "hard": "#d64f64",
          "practice": "#8e6ad2", "pre_task": "#a0a8b5", "post_task": "#9099a7",
          "feedback": "#47a5c5", "rating": "#9b77cc", "rest": "#8996a6",
          "interrupted": "#49566a", "baseline": "#bcc3ce", "unlabeled": "#b0b0b0"}
PHASE_NAMES = {"active_task": "Coding aktif", "feedback": "Feedback", "rating": "Rating",
               "rest": "Jeda", "pre_task": "Pra-tugas", "post_task": "Pasca-tugas",
               "interrupted": "Interupsi", "baseline": "Baseline", "unlabeled": "Tanpa label"}
NUMERIC_EYE = ["timestamp_ms", "gaze_x_raw", "gaze_y_raw", "gaze_x_smooth", "gaze_y_smooth",
               "EAR_Left", "EAR_Right", "iris_size_delta", "iris_size", "head_pitch", "head_yaw",
               "total_blinks", "fps_actual", "capture_fps", "confidence"]
MEAN_FEATURES = ["ear_left", "ear_right", "iris_delta", "pitch", "yaw"]
CSS = """body{font:15px/1.55 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 22px;color:#233146;background:#f4f6fa}
a{color:#145a9c}h1,h2{line-height:1.2}.note{padding:16px;background:#fff2d8;border-left:4px solid #df9d24}
table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #dce2eb;text-align:left}
.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.card{background:white;border:1px solid #dce2eb;border-radius:10px;padding:17px;margin:18px 0}
img{width:100%;height:auto}summary{cursor:pointer;font-weight:600}code{background:#e8edf4;padding:2px 5px}small{color:#5b687b}input{padding:10px;font:inherit;min-width:260px}footer{margin:30px 0;color:#5b687b}
"""


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def number(frame, name):
    values = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def truth(frame, name):
    return frame[name].astype(str).str.lower().isin(["true", "1"]) if name in frame else pd.Series(False, index=frame.index)


def epoch_ms(series):
    parsed = pd.to_datetime(series, utc=True, format="mixed", errors="raise")
    return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() * 1000


def wib(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("Asia/Jakarta").isoformat()


def percent(series):
    return float(series.mean() * 100) if len(series) else np.nan


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(paths, root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def sample_context(times, events, ready_ms):
    """Final ledger labels; independent of renamed archive directory names."""
    if not len(times):
        return pd.DataFrame(columns=["study_phase", "difficulty_level", "puzzle_set", "puzzle_id", "puzzle_title"])
    boundaries = sorted(set([float(np.min(times))] + [float(e["event_time_ms"]) for e in events]
                            + ([ready_ms] if ready_ms is not None else [])))
    contexts = [context_at(t, events, ready_ms) for t in boundaries]
    indices = np.searchsorted(boundaries, times, side="right") - 1
    return pd.DataFrame([contexts[i] for i in indices])


def phase_segments(start, end, events, ready_ms):
    """Attach exact boundaries (not rounded durations) to context segments."""
    boundaries = sorted(set([start, end] + [float(e["event_time_ms"]) for e in events if start < e["event_time_ms"] < end]
                            + ([ready_ms] if ready_ms is not None and start < ready_ms < end else [])))
    rows = []
    for left, right in zip(boundaries, boundaries[1:]):
        rows.append({"start_ms": left, "end_ms": right, "duration_seconds": (right-left)/1000,
                     **context_at((left+right)/2, events, ready_ms)})
    return pd.DataFrame(rows)


def prepare_eye(eye, screen):
    eye = eye.copy()
    for column in NUMERIC_EYE:
        eye[column] = number(eye, column)
    if eye.timestamp_ms.isna().any() or (eye.timestamp_ms.diff().dropna() < 0).any():
        raise ValueError("Timestamp mata hilang/mundur; periksa sumber sebelum EDA.")
    face = truth(eye, "face_detected")
    valid = eye.gaze_status.isin(["on_screen", "gaze_outside_screen"])
    finite_gaze = eye.gaze_x_raw.notna() & eye.gaze_y_raw.notna()
    eye["valid_gaze"] = valid & finite_gaze
    for column in ["gaze_x_smooth", "gaze_y_smooth"]:
        eye[column] = eye[column].where(eye.valid_gaze)
    w, h = screen["width"], screen["height"]
    clipped = (np.isclose(eye.gaze_x_raw, -.1*w) | np.isclose(eye.gaze_x_raw, 1.1*w)
               | np.isclose(eye.gaze_y_raw, -.1*h) | np.isclose(eye.gaze_y_raw, 1.1*h))
    eye["clipped"] = clipped & eye.valid_gaze
    eye["on_screen"] = eye.gaze_status.eq("on_screen")
    eye["face"] = face
    eye["ear_left"] = eye.EAR_Left.where(face)
    eye["ear_right"] = eye.EAR_Right.where(face)
    iris_mask = face & truth(eye, "iris_valid") & ~truth(eye, "is_blinking")
    eye["iris_delta"] = eye.iris_size_delta.where(iris_mask)
    eye["pitch"] = eye.head_pitch.where(face)
    eye["yaw"] = eye.head_yaw.where(face)
    # Counter increments are assigned to reopening timestamps; do not differentiate
    # the recorded 60-second rolling rate as though it were a window count.
    eye["blink_increment"] = eye.total_blinks.diff().clip(lower=0).fillna(0)
    return eye


def window_features(eye, starts, ends):
    """Inclusive endpoints match the existing fusion; NaN stays unavailable."""
    times = eye.timestamp_ms.to_numpy()
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, ends, side="right")
    counts = hi - lo
    result = {"eye_frames_count": counts}
    for name in MEAN_FEATURES + ["on_screen", "face", "valid_gaze", "clipped", "blink_increment"]:
        values = pd.to_numeric(eye[name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(values)
        totals = np.r_[0., np.cumsum(np.where(ok, values, 0.))]
        amounts = np.r_[0, np.cumsum(ok)]
        sums, n = totals[hi]-totals[lo], amounts[hi]-amounts[lo]
        if name == "blink_increment":
            result["blink_count"] = np.where(counts > 0, sums, np.nan)
        else:
            result[name + "_mean"] = np.divide(sums, n, out=np.full(len(n), np.nan), where=n>0)
    valid_count = result["valid_gaze_mean"] * counts
    clipped_count = result["clipped_mean"] * counts
    result["clipping_ratio_valid"] = np.divide(clipped_count, valid_count, out=np.full(len(counts), np.nan), where=valid_count>0)
    return pd.DataFrame(result)


def hrv_eligibility(frame):
    ready = frame.status.eq("ready") & frame.rmssd_ms.notna()
    level = ready & frame.study_phase.eq("active_task") & frame.difficulty_level.isin(LEVELS)
    question = level & ~frame.puzzle_id.isin(["", "mixed"]) & ~frame.puzzle_set.isin(["", "mixed"])
    return level, question


@dataclass
class Session:
    name: str
    original_id: str
    directory: Path
    origin: float
    ready_ms: float | None
    eye: pd.DataFrame
    hrv: pd.DataFrame
    events: pd.DataFrame
    rr: pd.DataFrame
    segments: pd.DataFrame
    binned: pd.DataFrame
    questions: pd.DataFrame
    levels: pd.DataFrame
    calibration: dict
    baseline_rmssd: float
    screen: dict
    xmin: float
    xmax: float
    notes: list[str] = field(default_factory=list)
    bin_seconds: float = 5.
    charts: list[tuple[str, str]] = field(default_factory=list)


def context_key(row):
    return (row["difficulty_level"], row["puzzle_set"], row["puzzle_id"])


def question_summaries(eye, hrv, segments, events):
    keys, titles = [], {}
    for row in events.to_dict("records"):
        if row["event_type"] == "puzzle_started" and row.get("puzzle_id"):
            key = context_key(row)
            if key not in keys:
                keys.append(key)
            titles[key] = row.get("puzzle_title") or key[-1]
    active = eye[eye.study_phase.eq("active_task")]
    for key in active.groupby(["difficulty_level", "puzzle_set", "puzzle_id"], sort=False).groups:
        if key not in keys and key[-1]:
            keys.append(key)
    rows = []
    eye_start, eye_end = eye.timestamp_ms.min(), eye.timestamp_ms.max()
    for level, puzzle_set, puzzle_id in keys:
        group = active[(active.difficulty_level == level) & (active.puzzle_set == puzzle_set) & (active.puzzle_id == puzzle_id)]
        spans = segments[(segments.study_phase == "active_task") & (segments.difficulty_level == level)
                         & (segments.puzzle_set == puzzle_set) & (segments.puzzle_id == puzzle_id)]
        recorded_seconds = sum(max(0, min(r.end_ms, eye_end)-max(r.start_ms, eye_start))/1000 for r in spans.itertuples())
        hw = hrv[hrv.eligible_question & (hrv.difficulty_level == level) & (hrv.puzzle_set == puzzle_set) & (hrv.puzzle_id == puzzle_id)]
        attempts = events[(events.event_type == "attempt_submitted") & (events.difficulty_level == level)
                          & (events.puzzle_set == puzzle_set) & (events.puzzle_id == puzzle_id)]
        clip = group.loc[group.valid_gaze, "clipped"]
        rows.append({"difficulty_level": level, "puzzle_set": puzzle_set, "puzzle_id": puzzle_id,
                     "puzzle_title": titles.get((level,puzzle_set,puzzle_id), puzzle_id),
                     "active_ledger_seconds": spans.duration_seconds.sum(),
                     "active_seconds_in_eye_range": recorded_seconds, "eye_frames": len(group),
                     "ear_left_median": group.ear_left.median(), "ear_right_median": group.ear_right.median(),
                     "iris_delta_median": group.iris_delta.median(),
                     "blink_count": group.blink_increment.sum() if len(group) else np.nan,
                     "blink_rate_per_active_minute": group.blink_increment.sum()*60/recorded_seconds if len(group) and recorded_seconds>0 else np.nan,
                     "on_screen_pct": percent(group.on_screen), "clipping_pct_valid": percent(clip),
                     "face_pct": percent(group.face), "iris_available_pct": percent(group.iris_delta.notna()),
                     "ready_single_question_hrv_windows": len(hw), "rmssd_median_ms": hw.rmssd_ms.median(),
                     "rmssd_ratio_median": hw.rmssd_ratio.median(),
                     "submitted_attempts": len(attempts), "correct_attempts": int(attempts.outcome.eq("correct").sum()),
                     "incorrect_attempts": int(attempts.outcome.eq("incorrect").sum())})
    return pd.DataFrame(rows)


def level_summaries(eye, hrv, segments):
    rows=[]
    for level in LEVELS:
        group=eye[(eye.study_phase == "active_task") & (eye.difficulty_level == level)]
        spans=segments[(segments.study_phase == "active_task") & (segments.difficulty_level == level)]
        seconds=sum(max(0,min(r.end_ms,eye.timestamp_ms.max())-max(r.start_ms,eye.timestamp_ms.min()))/1000 for r in spans.itertuples())
        hw=hrv[hrv.eligible_level & (hrv.difficulty_level == level)]
        rows.append({"difficulty_level":level, "eye_frames":len(group), "active_seconds_in_eye_range":seconds,
                     "ear_mean_median": ((group.ear_left+group.ear_right)/2).median(),
                     "iris_delta_median":group.iris_delta.median(),
                     "on_screen_pct":percent(group.on_screen),
                     "clipping_pct_valid":percent(group.loc[group.valid_gaze,"clipped"]),
                     "blink_rate_per_active_minute":group.blink_increment.sum()*60/seconds if len(group) and seconds>0 else np.nan,
                     "ready_active_hrv_windows":len(hw), "rmssd_median_ms":hw.rmssd_ms.median(),
                     "rmssd_ratio_median":hw.rmssd_ratio.median()})
    return pd.DataFrame(rows)


def load_session(directory, bin_seconds):
    eye=read_csv(directory/"eye_tracking.csv")
    raw_hrv=read_csv(directory/"hrv.csv")
    if eye.empty or raw_hrv.empty:
        raise ValueError(f"{directory.name}: eye_tracking.csv atau hrv.csv kosong/tidak ada.")
    calibration=read_json(directory/"eye_summary_calibration.json")
    screen=calibration.get("config",{}).get("screen",{"width":2880,"height":1800})
    ids=set(eye.session_id.astype(str)) | set(raw_hrv.session_id.astype(str))
    if len(ids)!=1:
        raise ValueError(f"{directory.name}: ID sensor tidak konsisten: {ids}")
    original_id=next(iter(ids))
    events=read_csv(directory/"task_events.csv")
    notes=[]
    if events.empty:
        events=pd.DataFrame(columns=["event_time_ms","event_type","difficulty_level","puzzle_set","puzzle_id","puzzle_title","outcome","session_id"])
        notes.append("Ledger tidak tersedia: kaitan fase/soal tidak dapat ditentukan.")
    else:
        events=events[events.session_id.eq(original_id)].copy()
        events["event_time_ms"]=number(events,"event_time_ms")
        events=events.dropna(subset=["event_time_ms"]).sort_values("event_time_ms",kind="stable").reset_index(drop=True)
        if events.empty:
            notes.append("Tidak ada kejadian yang cocok dengan ID asli sensor.")
    status=read_json(directory/"hrv_calib_status.json")
    ready_ms=pd.Timestamp(status["both_calibrated_at_utc"]).timestamp()*1000 if status.get("both_calibrated_at_utc") else None
    eye=prepare_eye(eye,screen).reset_index(drop=True)
    origin=ready_ms if ready_ms is not None else float(eye.timestamp_ms.min())
    labels=sample_context(eye.timestamp_ms.to_numpy(),events.to_dict("records"),ready_ms)
    for col in labels:
        eye[col]=labels[col].to_numpy()
    eye["t_seconds"]=(eye.timestamp_ms-origin)/1000
    eye["_bin"]=np.floor(eye.t_seconds/bin_seconds).astype(int)
    eye["state_fixation"]=eye.Eye_State.eq("Fixation")
    eye["state_saccade"]=eye.Eye_State.eq("Saccade")
    eye["state_transition"]=eye.Eye_State.eq("Transition")
    eye["state_unclassified"]=eye.Eye_State.eq("Unclassified")
    aggregation={name:(name,"mean") for name in MEAN_FEATURES+["on_screen","face","valid_gaze","clipped","state_fixation","state_saccade","state_transition","state_unclassified"]}
    aggregation.update(frames=("timestamp_ms","size"),blink_count=("blink_increment","sum"),
                       gaze_x=("gaze_x_smooth","median"),gaze_y=("gaze_y_smooth","median"),
                       gaze_x_p10=("gaze_x_smooth",lambda x:x.quantile(.1)),gaze_x_p90=("gaze_x_smooth",lambda x:x.quantile(.9)),
                       gaze_y_p10=("gaze_y_smooth",lambda x:x.quantile(.1)),gaze_y_p90=("gaze_y_smooth",lambda x:x.quantile(.9)),
                       iris_available=("iris_delta",lambda x:x.notna().mean()),
                       fps_processing=("fps_actual","mean"),fps_capture=("capture_fps","mean"))
    binned=eye.groupby("_bin",sort=True).agg(**aggregation).reset_index()
    binned["t_minutes"]=(binned._bin+.5)*bin_seconds/60
    binned["clipping_ratio_valid"]=binned.clipped.div(binned.valid_gaze.where(binned.valid_gaze>0))
    baseline=raw_hrv[raw_hrv.phase.eq("baseline")]
    baseline_rmssd=(float(number(baseline,"rmssd_ms").iloc[-1])
                    if len(baseline) and baseline.iloc[-1]["status"]=="ready" else np.nan)
    hrv=raw_hrv[raw_hrv.phase.eq("task")].copy().reset_index(drop=True)
    starts,ends=epoch_ms(hrv.window_start),epoch_ms(hrv.window_end)
    for col in ["rmssd_ms","sdnn_ms","coverage","rr_count","rr_pairs","data_age_seconds","window_seconds"]:
        hrv[col]=number(hrv,col)
    # Mask unavailable HRV even when an older export happens to contain a number.
    hrv.loc[~hrv.status.eq("ready"),["rmssd_ms","sdnn_ms"]]=np.nan
    hrv["rmssd_ratio"]=hrv.rmssd_ms/baseline_rmssd if baseline_rmssd>0 else np.nan
    contexts=[summarize_segments(interval_segments(a,b,events.to_dict("records"),ready_ms)) for a,b in zip(starts,ends)]
    for col in ["study_phase","difficulty_level","puzzle_set","puzzle_id","window_phase_mix"]:
        hrv[col]=[r[col] for r in contexts]
    hrv["t_minutes"]=(ends-origin)/60000
    hrv["start_ms"],hrv["end_ms"]=starts,ends
    features=window_features(eye,starts,ends)
    for col in features:
        hrv[col]=features[col]
    hrv["eligible_level"],hrv["eligible_question"]=hrv_eligibility(hrv)
    rr=read_csv(directory/"rr_raw.csv")
    if not rr.empty:
        rr["t_minutes"]=(epoch_ms(rr.received_at)-origin)/60000
        rr["rr_original_ms"]=number(rr,"rr_original_ms")
    if not events.empty:
        events["t_minutes"]=(events.event_time_ms-origin)/60000
    all_starts=[eye.timestamp_ms.min(),origin]+starts.tolist()
    all_ends=[eye.timestamp_ms.max()]+ends.tolist()
    if not events.empty:
        all_starts.append(events.event_time_ms.min())
        all_ends.append(events.event_time_ms.max())
    xmin,xmax=float(min(all_starts)),float(max(all_ends))
    segments=phase_segments(xmin,xmax,events.to_dict("records"),ready_ms)
    questions=question_summaries(eye,hrv,segments,events)
    levels=level_summaries(eye,hrv,segments)
    validation=[a for a in calibration.get("attempts",[]) if a.get("phase")=="held_out_validation"]
    if validation and not validation[-1].get("passed_thresholds",False):
        notes.append("Validasi gaze tidak lulus aturan program; lokasi/heatmap/Eye_State adalah keluaran estimasi.")
    if not hrv.status.eq("ready").any():
        notes.append("Tidak ada HRV tugas berstatus ready. Grafik/analisis HRV siap sengaja kosong.")
    if not (directory/"multimodal_timeline.csv").exists():
        notes.append("Timeline gabungan tidak tersedia; EDA ini menyelaraskan sumber secara read-only.")
    if not (directory/"eye_crops").exists():
        notes.append("Gambar potongan mata tidak tersedia dalam arsip.")
    if directory.name in ["Ramadhan","Ramadhan-2"]:
        notes.append("Satu peserta dengan dua rekaman; referensi dan interupsi dipertahankan terpisah.")
    return Session(directory.name,original_id,directory,origin,ready_ms,eye,hrv,events,rr,segments,binned,
                   questions,levels,calibration,baseline_rmssd,screen,(xmin-origin)/60000,(xmax-origin)/60000,notes,bin_seconds)


def empty(ax, message="Data tidak tersedia"):
    ax.text(.5,.5,message,ha="center",va="center",transform=ax.transAxes,color="#65758a",wrap=True)


def shade(ax, session):
    for row in session.segments.itertuples():
        if row.study_phase=="active_task":
            ax.axvspan((row.start_ms-session.origin)/60000,(row.end_ms-session.origin)/60000,
                       color=COLORS.get(row.difficulty_level,"#999999"),alpha=.075,linewidth=0)
    ax.set_xlim(session.xmin,session.xmax)
    ax.grid(alpha=.2)


def marks(ax,session):
    for r in session.events[session.events.event_type.eq("set_started")].itertuples():
        ax.axvline(r.t_minutes,color=COLORS.get(r.difficulty_level,"#888888"),lw=.8,ls="--",alpha=.7)


def footer(fig,text,bottom=.045):
    # Reserve real canvas space; bbox_inches='tight' alone does not prevent
    # figure-level footnotes from overlapping the bottom axis label.
    engine=fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0,bottom,1,1-bottom))
    fig.text(.01,.005,text,fontsize=8,color="#5b687b",ha="left",va="bottom")


def save(fig,folder,name,title,session=None,dpi=115):
    fig.savefig(folder/name,dpi=dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    if session is not None:
        session.charts.append((name,title))


def line(ax,x,y,label=None,color=None,**kwargs):
    ax.plot(np.asarray(x,dtype=float),np.asarray(y,dtype=float),label=label,color=color,**kwargs)


def plot_overview(s,out,dpi):
    fig,axes=plt.subplots(7,1,figsize=(17,17),sharex=True,layout="constrained")
    b,h=s.binned,s.hrv
    line(axes[0],h.t_minutes,h.rmssd_ms,"RMSSD siap", "#196da4")
    line(axes[0],h.t_minutes,h.sdnn_ms,"SDNN siap","#8c62aa")
    if np.isfinite(s.baseline_rmssd): axes[0].axhline(s.baseline_rmssd,color="#196da4",ls=":",label="Baseline RMSSD")
    axes[0].set_ylabel("HRV (ms)"); axes[0].legend(loc="upper right",fontsize=8)
    line(axes[1],h.t_minutes,h.coverage*100,"Cakupan HRV","#196da4")
    line(axes[1],b.t_minutes,b.face*100,"Wajah tersedia","#30a46c")
    line(axes[1],b.t_minutes,b.clipping_ratio_valid*100,"Gaze clipping / valid","#d64f64")
    axes[1].axhline(80,color="#777777",ls=":",lw=.8); axes[1].set_ylabel("Kualitas (%)");axes[1].legend(loc="upper right",fontsize=8,ncol=3)
    line(axes[2],b.t_minutes,b.ear_left,"EAR kiri","#196da4");line(axes[2],b.t_minutes,b.ear_right,"EAR kanan","#e09925")
    axes[2].set_ylabel("Rata-rata EAR");axes[2].legend(loc="upper right",fontsize=8)
    axes[3].bar(b.t_minutes,b.blink_count,width=s.bin_seconds/60*.85,color="#7c63b0")
    axes[3].set_ylabel("Kedipan/bin")
    line(axes[4],b.t_minutes,b.iris_delta,color="#238e81");axes[4].set_ylabel("Delta iris\n(proxy landmark)")
    line(axes[5],b.t_minutes,b.gaze_x,"x median","#196da4");line(axes[5],b.t_minutes,b.gaze_y,"y median","#e09925")
    axes[5].set_ylabel("Gaze estimasi\n(px)");axes[5].legend(loc="upper right",fontsize=8)
    for phase,group in s.segments.groupby("study_phase",sort=False):
        for r in group.itertuples():
            color=COLORS.get(r.difficulty_level if phase=="active_task" else phase,"#999999")
            y=0 if phase=="active_task" else 1
            axes[6].broken_barh([((r.start_ms-s.origin)/60000,r.duration_seconds/60)],(y-.35,.7),facecolors=color)
    attempts=s.events[s.events.event_type.eq("attempt_submitted")]
    if len(attempts): axes[6].scatter(attempts.t_minutes,np.full(len(attempts),-.6),marker="|",color="#27364b",label="Submit")
    axes[6].set_yticks([0,1],["Coding", "Fase lain"]);axes[6].set_ylim(-.9,1.6)
    axes[6].legend(handles=[Patch(color=COLORS[k],label=k) for k in LEVELS]+[Patch(color=COLORS["rating"],label="rating"),Patch(color=COLORS["feedback"],label="feedback")],ncol=5,fontsize=8,loc="upper right")
    for ax in axes: shade(ax,s);marks(ax,s)
    axes[-1].set_xlabel(f"Menit sejak kedua sensor siap / awal mata | nol: {wib(s.origin)}")
    fig.suptitle(f"{s.name} — sensor, fase coding, dan submit",fontsize=16)
    footer(fig,"Gaze/iris/Eye_State adalah estimasi; HRV tak siap tidak diisi nol. Warna latar: easy/medium/hard. Batas waktu mencakup ledger, termasuk bagian tanpa sensor.")
    save(fig,out,"01_overview.png","Ringkasan sensor terhadap waktu",s,dpi)


def plot_gantt(s,out,dpi):
    q=s.questions
    fig,ax=plt.subplots(figsize=(17,max(5,len(q)*.38+2)),layout="constrained")
    if q.empty:
        empty(ax,"Tidak ada soal pada ledger")
    else:
        mapping={context_key(r):i for i,r in enumerate(q.to_dict("records"))}
        for r in s.segments[s.segments.study_phase.eq("active_task")].to_dict("records"):
            key=context_key(r)
            if key in mapping:
                ax.broken_barh([((r["start_ms"]-s.origin)/60000,r["duration_seconds"]/60)],(mapping[key]-.3,.6),facecolors=COLORS.get(key[0],"#999999"))
        for r in s.events[s.events.event_type.eq("attempt_submitted")].to_dict("records"):
            key=context_key(r)
            if key in mapping:
                ax.scatter(r["t_minutes"],mapping[key],color="#18794e" if r["outcome"]=="correct" else "#bd263b",marker="o" if r["outcome"]=="correct" else "x",s=28,zorder=4)
        ax.set_yticks(range(len(q)),[f"{r.difficulty_level} | {r.puzzle_id}" for r in q.itertuples()],fontsize=8)
        ax.invert_yaxis()
        eye_min,eye_max=s.eye.t_seconds.min()/60,s.eye.t_seconds.max()/60
        if s.xmin<eye_min: ax.axvspan(s.xmin,eye_min,color="#555555",alpha=.14)
        if s.xmax>eye_max: ax.axvspan(eye_max,s.xmax,color="#555555",alpha=.14)
    ax.set_xlim(s.xmin,s.xmax);ax.grid(axis="x",alpha=.25);ax.set_xlabel("Menit relatif; area abu-abu di luar rentang rekaman mata")
    ax.set_title(f"{s.name} — kapan setiap soal dikerjakan (percobaan ulang tetap terlihat)")
    fig.legend(handles=[Patch(color=COLORS[k],label=k) for k in LEVELS]+[plt.Line2D([],[],marker="o",ls="",color="#18794e",label="Submit benar"),plt.Line2D([],[],marker="x",ls="",color="#bd263b",label="Submit salah")],fontsize=8,ncol=5,loc="outside upper center")
    footer(fig,"Batang hanya fase active_task menurut ledger akhir; celah dapat berarti feedback/rating/jeda. Durasi ledger tidak menjamin ada sensor.")
    save(fig,out,"02_soal_timeline.png","Urutan soal dan hasil submit",s,dpi)


def plot_eye(s,out,dpi):
    b=s.binned
    fig,axes=plt.subplots(5,1,figsize=(17,14),sharex=True,layout="constrained")
    for i,(axis,limit,color) in enumerate([("x",s.screen["width"],"#196da4"),("y",s.screen["height"],"#dd8c21")]):
        axes[i].fill_between(b.t_minutes.to_numpy(),b[f"gaze_{axis}_p10"].to_numpy(),b[f"gaze_{axis}_p90"].to_numpy(),color=color,alpha=.18,label="P10–P90/bin")
        line(axes[i],b.t_minutes,b[f"gaze_{axis}"],"Median/bin",color)
        for boundary in [0,limit]:axes[i].axhline(boundary,color="#30a46c",ls="--",lw=.8)
        for boundary in [-.1*limit,1.1*limit]:axes[i].axhline(boundary,color="#d64f64",ls=":",lw=.8)
        axes[i].set_ylabel(f"Gaze {axis} (px)");axes[i].legend(fontsize=8,loc="upper right")
    line(axes[2],b.t_minutes,b.on_screen*100,"Pada layar","#30a46c")
    line(axes[2],b.t_minutes,b.clipping_ratio_valid*100,"Clipping/valid","#d64f64")
    line(axes[2],b.t_minutes,(1-b.valid_gaze)*100,"Tidak valid","#8490a3")
    axes[2].set_ylabel("Frame (%)");axes[2].legend(fontsize=8,ncol=3,loc="upper right")
    axes[3].stackplot(b.t_minutes,*[b["state_"+key]*100 for key in ["fixation","saccade","transition","unclassified"]],labels=["Fixation","Saccade","Transition","Unclassified"],colors=["#30a46c","#d64f64","#e5a000","#a7b1c0"],alpha=.8)
    axes[3].set_ylabel("Eye_State (%)");axes[3].legend(fontsize=8,ncol=4,loc="upper right");axes[3].set_ylim(0,100)
    line(axes[4],b.t_minutes,b.pitch,"Pitch","#196da4");line(axes[4],b.t_minutes,b.yaw,"Yaw","#8e6ad2")
    axes[4].set_ylabel("Pose estimasi (°)");axes[4].legend(fontsize=8,loc="upper right")
    for ax in axes:shade(ax,s);marks(ax,s)
    axes[-1].set_xlabel("Menit sejak nol sesi");fig.suptitle(f"{s.name} — gaze, clipping, label gerak mata, dan pose",fontsize=15)
    footer(fig,"Garis hijau: layar; merah: pembatas mapper. P10/P90 adalah sebaran prediksi, bukan interval kepercayaan. Eye_State merupakan label algoritme.")
    save(fig,out,"03_eye_time.png","Gaze dan Eye_State terhadap waktu",s,dpi)


def plot_hrv(s,out,dpi):
    h=s.hrv
    fig,axes=plt.subplots(4,1,figsize=(17,12),sharex=True,layout="constrained")
    if not s.rr.empty: axes[0].scatter(s.rr.t_minutes,s.rr.rr_original_ms,s=3,alpha=.45,color="#196da4",rasterized=True)
    else:empty(axes[0],"RR mentah tidak tersedia")
    axes[0].set_ylabel("RR mentah (ms)")
    line(axes[1],h.t_minutes,h.rmssd_ms,"RMSSD siap","#196da4");line(axes[1],h.t_minutes,h.sdnn_ms,"SDNN siap","#8e6ad2")
    axes[1].set_ylabel("HRV (ms)");axes[1].legend(fontsize=8,loc="upper right")
    if not h.rmssd_ms.notna().any():empty(axes[1],"Tidak ada HRV tugas siap; tidak diimputasi")
    line(axes[2],h.t_minutes,h.rmssd_ratio,color="#196da4");axes[2].axhline(1,color="#777777",ls="--",lw=.8);axes[2].set_ylabel("RMSSD / baseline")
    statuses=list(dict.fromkeys(["ready","insufficient_data","gap","recovering"]+h.status.tolist()))
    status_map={v:i for i,v in enumerate(statuses)}
    for status,i in status_map.items():
        mask=h.status.eq(status);axes[3].scatter(h.loc[mask,"t_minutes"],np.full(mask.sum(),i),s=10,label=status)
    axes[3].set_yticks(range(len(statuses)),statuses,fontsize=8);axes[3].set_ylabel("Status HRV")
    for ax in axes:shade(ax,s);marks(ax,s)
    if not s.rr.empty:
        # Raw RR may begin before the first linked web event (e.g. Tio).
        # Keep the entire raw stream visible on this diagnostic chart.
        for ax in axes:
            ax.set_xlim(min(s.xmin,float(s.rr.t_minutes.min())),max(s.xmax,float(s.rr.t_minutes.max())))
    axes[-1].set_xlabel("Menit sejak nol sesi");fig.suptitle(f"{s.name} — RR mentah dan HRV (25 detik, langkah 5 detik)",fontsize=15)
    footer(fig,"Garis HRV hanya status ready; celah dipertahankan. RR mentah belum dikoreksi artefak. Rasio baseline bukan label keadaan kognitif.")
    save(fig,out,"04_hrv_time.png","RR, HRV, dan status perangkat",s,dpi)


def heatmap(ax,group,s,title):
    valid=group[group.valid_gaze & group.gaze_x_smooth.notna() & group.gaze_y_smooth.notna()]
    w,h=s.screen["width"],s.screen["height"]
    if valid.empty:empty(ax,"Tidak ada prediksi gaze valid")
    else:
        ax.hist2d(valid.gaze_x_smooth,valid.gaze_y_smooth,bins=[55,35],range=[[-.1*w,1.1*w],[-.1*h,1.1*h]],cmap="magma",norm=LogNorm(vmin=1),cmin=1)
    ax.add_patch(Rectangle((0,0),w,h,fill=False,edgecolor="#2fb17a",linewidth=1.4))
    ax.set_xlim(-.1*w,1.1*w);ax.set_ylim(1.1*h,-.1*h);ax.set_aspect("equal")
    ax.set_title(title,fontsize=9);ax.set_xlabel("x prediksi (px)",fontsize=8);ax.set_ylabel("y prediksi (px)",fontsize=8);ax.tick_params(labelsize=7)


def plot_heatmaps(s,out,dpi):
    active=s.eye[s.eye.study_phase.eq("active_task")]
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    heatmap(axes.flat[0],active,s,"Semua coding aktif")
    for ax,level in zip(list(axes.flat)[1:],LEVELS):heatmap(ax,active[active.difficulty_level==level],s,level)
    fig.suptitle(f"{s.name} — frekuensi prediksi gaze, bukan kebenaran lokasi",fontsize=14)
    footer(fig,"Warna logaritmik menghitung frame; bukan durasi fiksasi. Skala warna tiap panel mandiri. Kotak hijau adalah layar; area luar tetap ditampilkan.")
    save(fig,out,"05_gaze_heatmap_level.png","Heatmap gaze per level",s,dpi)
    records=s.questions.to_dict("records")
    for page,start in enumerate(range(0,len(records),12),1):
        batch=records[start:start+12];cols=min(3,len(batch));rows=math.ceil(len(batch)/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(cols*4.4,rows*3.5),squeeze=False,layout="constrained")
        for ax,r in zip(axes.flat,batch):
            group=active[(active.difficulty_level==r["difficulty_level"])&(active.puzzle_set==r["puzzle_set"])&(active.puzzle_id==r["puzzle_id"])]
            heatmap(ax,group,s,f'{r["difficulty_level"]} | {r["puzzle_id"]}\nn={len(group):,} frame aktif')
        for ax in list(axes.flat)[len(batch):]:ax.set_axis_off()
        fig.suptitle(f"{s.name} — estimasi gaze per soal | halaman {page}",fontsize=14)
        footer(fig,"Hanya frame active_task. Seluruh soal ledger disertakan; soal tanpa sensor ditampilkan kosong. Warna menghitung frame, skala per panel mandiri.")
        save(fig,out,f"06_gaze_per_soal_{page:02d}.png",f"Heatmap tiap soal ({page})",s,dpi)


def plot_questions(s,out,dpi):
    q=s.questions
    fig,axes=plt.subplots(2,3,figsize=(19,max(9,len(q)*.3+3)),layout="constrained")
    if q.empty:
        for ax in axes.flat:empty(ax,"Tidak ada soal")
    else:
        y=np.arange(len(q));labels=[f"{r.difficulty_level} | {r.puzzle_id}" for r in q.itertuples()]
        specs=[("active_seconds_in_eye_range","Durasi aktif dalam rentang mata (s)"),
               ("rmssd_ratio_median","Median RMSSD / baseline (siap, satu soal)"),
               ("ear_left_median","Median EAR kiri"),("iris_delta_median","Median delta iris (proxy)"),
               ("blink_rate_per_active_minute","Kedipan terdeteksi / menit aktif"),
               ("clipping_pct_valid","Clipping / gaze valid (%)")]
        for ax,(col,title) in zip(axes.flat,specs):
            vals=pd.to_numeric(q[col],errors="coerce");ax.barh(y,vals,color=[COLORS.get(v,"#999999") for v in q.difficulty_level])
            ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis="x",alpha=.2)
            for i,value in enumerate(vals):
                if not np.isfinite(value):ax.text(.02,i,"tidak tersedia",transform=ax.get_yaxis_transform(),fontsize=7,color="#778195",va="center")
    fig.suptitle(f"{s.name} — ringkasan tiap soal, percobaan ulang digabung",fontsize=15)
    footer(fig,"HRV per soal hanya jendela ready seluruhnya active_task dan satu soal; kosong bukan nol. Kedipan dibagi durasi aktif dalam rentang rekaman mata.")
    save(fig,out,"07_fitur_per_soal.png","Perbandingan fitur antarsoal",s,dpi)


def plot_distributions(s,out,dpi):
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout="constrained")
    specs=[("ear_left","EAR kiri"),("ear_right","EAR kanan"),("iris_delta","Delta iris (proxy)"),
           ("pitch","Pitch estimasi"),("yaw","Yaw estimasi")]
    active=s.eye[s.eye.study_phase.eq("active_task")]
    for ax,(col,title) in zip(axes.flat,specs):
        found=False
        for level in LEVELS:
            vals=active.loc[active.difficulty_level==level,col].dropna()
            if len(vals):ax.hist(vals,bins=45,density=True,histtype="step",linewidth=1.5,label=level,color=COLORS[level]);found=True
        ax.set_title(title);ax.set_ylabel("Densitas frame")
        if found:ax.legend(fontsize=8)
        else:empty(ax)
    ax=axes.flat[-1]
    for level in LEVELS:
        vals=s.hrv.loc[s.hrv.eligible_level & (s.hrv.difficulty_level==level),"rmssd_ratio"].dropna()
        if len(vals):ax.hist(vals,bins=22,density=True,histtype="step",linewidth=1.5,label=level,color=COLORS[level])
    ax.set_title("RMSSD / baseline (jendela aktif siap)");ax.set_ylabel("Densitas jendela")
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    else:empty(ax,"Tidak ada jendela HRV aktif siap")
    fig.suptitle(f"{s.name} — distribusi fitur per level",fontsize=15)
    footer(fig,"Distribusi frame/jendela bersifat deskriptif; sampel berurutan dan jendela overlap bukan pengamatan independen.")
    save(fig,out,"08_distribusi.png","Distribusi fitur per level",s,dpi)


def plot_correlation(s,out,dpi):
    columns=["rmssd_ratio","sdnn_ms","ear_left_mean","ear_right_mean","iris_delta_mean","blink_count","clipping_ratio_valid","on_screen_mean","pitch_mean","yaw_mean"]
    subset=s.hrv[s.hrv.eligible_level & (s.hrv.eye_frames_count>0)][columns]
    corr=subset.corr(method="spearman",min_periods=5)
    pair_counts=subset.notna().astype(int).T.dot(subset.notna().astype(int))
    corr.to_csv(out/"correlation_spearman.csv");pair_counts.to_csv(out/"correlation_pair_counts.csv")
    fig,ax=plt.subplots(figsize=(11,9),layout="constrained")
    if corr.notna().any().any():
        plot=ax.imshow(np.ma.masked_invalid(corr.to_numpy()),vmin=-1,vmax=1,cmap="coolwarm")
        fig.colorbar(plot,ax=ax,label="Spearman rho")
        for i in range(len(columns)):
            for j in range(len(columns)):
                if np.isfinite(corr.iloc[i,j]):ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
    else:empty(ax,"Korelasi tidak tersedia: kurang data siap/bervariasi")
    ax.set_xticks(range(len(columns)),columns,rotation=55,ha="right",fontsize=8);ax.set_yticks(range(len(columns)),columns,fontsize=8)
    ax.set_title(f"{s.name} — korelasi eksploratif | {len(subset)} jendela aktif siap")
    footer(fig,"Minimum 5 pasangan lengkap, per pasangan; lihat correlation_pair_counts.csv. Tanpa p-value: overlap/ketergantungan temporal dan kualitas sensor membatasi interpretasi.")
    save(fig,out,"09_korelasi.png","Korelasi fitur pada jendela HRV siap",s,dpi)


def plot_calibration(s,out,dpi):
    attempts=[a for a in s.calibration.get("attempts",[]) if a.get("phase")=="held_out_validation"]
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    if not attempts:
        for ax in axes:empty(ax,"Validasi tidak tersedia")
    else:
        points=attempts[-1].get("points",[]);errors=[]
        for i,p in enumerate(points):
            target=p["target"];samples=np.array([r["prediction"] for r in p.get("samples",[])])
            axes[0].scatter(*target,marker="+",s=110,color="#30a46c")
            if len(samples):
                axes[0].scatter(samples[:,0],samples[:,1],s=12,alpha=.6,color="#196da4")
                avg=samples.mean(axis=0);axes[0].annotate("",xy=avg,xytext=target,arrowprops={"arrowstyle":"->","color":"#d64f64"})
            axes[0].annotate(str(i+1),target,xytext=(5,5),textcoords="offset points")
            errors.append(p.get("mean_prediction_error_px",np.nan))
        w,h=s.screen["width"],s.screen["height"]
        axes[0].add_patch(Rectangle((0,0),w,h,fill=False,edgecolor="#30a46c"));axes[0].set_xlim(-.1*w,1.1*w);axes[0].set_ylim(1.1*h,-.1*h)
        axes[0].set_title("Hijau: target | biru: prediksi | panah: rata-rata")
        axes[1].bar(np.arange(len(errors))+1,errors,color="#196da4");axes[1].set_xlabel("Titik validasi (1 = tengah)");axes[1].set_ylabel("Error rata-rata target (px)")
        axes[1].set_title("Kesalahan tiap titik; tidak disembunyikan oleh median")
        for value,color,label in [(180,"#8490a0","Median asli 180"),(450,"#df9d24","Median audit 450"),(900,"#d64f64","P95 audit 900")]:axes[1].axhline(value,color=color,ls="--",lw=.8,label=label)
        axes[1].legend(fontsize=8)
    fig.suptitle(f"{s.name} — validasi gaze sebelum tugas",fontsize=15)
    footer(fig,"Garis 450/900 adalah kriteria penyaringan eksploratif audit untuk statistik median/P95, bukan batas per target yang disahkan ilmiah.")
    save(fig,out,"10_validasi.png","Target dan kesalahan kalibrasi/validasi",s,dpi)


def plot_device(s,out,dpi):
    b,h=s.binned,s.hrv
    fig,axes=plt.subplots(4,1,figsize=(17,11),sharex=True,layout="constrained")
    line(axes[0],b.t_minutes,b.fps_capture,"FPS capture teramati","#196da4")
    line(axes[0],b.t_minutes,b.fps_processing,"FPS pemrosesan tercatat","#e09925")
    axes[0].set_ylabel("FPS");axes[0].legend(fontsize=8,loc="upper right")
    line(axes[1],b.t_minutes,b.iris_available*100,"Delta iris tersedia","#238e81")
    line(axes[1],b.t_minutes,b.face*100,"Wajah tersedia","#30a46c")
    axes[1].set_ylabel("Frame (%)");axes[1].legend(fontsize=8,loc="upper right")
    line(axes[2],h.t_minutes,h.rr_pairs,color="#196da4");axes[2].axhline(10,color="#d64f64",ls=":",lw=.8);axes[2].set_ylabel("Pasangan RR")
    line(axes[3],h.t_minutes,h.data_age_seconds,color="#8e6ad2");axes[3].axhline(3,color="#d64f64",ls=":",lw=.8);axes[3].set_ylabel("Usia data HRV (s)")
    for ax in axes:shade(ax,s);marks(ax,s)
    axes[-1].set_xlabel("Menit sejak nol sesi");fig.suptitle(f"{s.name} — ketersediaan fitur dan diagnostik perangkat",fontsize=15)
    footer(fig,"Garis acuan pasangan/usia adalah syarat kecukupan HRV, bukan ambang kognitif. Wajah ditemukan tidak menjamin gaze akurat.")
    save(fig,out,"11_kualitas_perangkat.png","FPS, data hilang, dan kecukupan HRV",s,dpi)


def plot_heuristic(s,out,dpi):
    timeline=read_csv(s.directory/"multimodal_timeline.csv")
    fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True,layout="constrained")
    if timeline.empty or "cognitive_state" not in timeline:
        for ax in axes:empty(ax,"Keluaran heuristik tersimpan tidak tersedia; tidak dibuat ulang oleh EDA")
    else:
        t=(epoch_ms(timeline.window_end)-s.origin)/60000
        score=number(timeline,"cognitive_load_score")
        line(axes[0],t,score,color="#8e6ad2");axes[0].set_ylabel("Skor aturan tersimpan");axes[0].set_ylim(0,100)
        states=list(dict.fromkeys(timeline.cognitive_state.tolist()))
        for i,state in enumerate(states):
            mask=timeline.cognitive_state.eq(state);axes[1].scatter(t[mask],np.full(mask.sum(),i),s=11,label=state)
        axes[1].set_yticks(range(len(states)),states,fontsize=8)
    for ax in axes:shade(ax,s);marks(ax,s)
    axes[-1].set_xlabel("Menit sejak nol sesi");fig.suptitle(f"{s.name} — keluaran HEURISTIK tersimpan, bukan ground truth",fontsize=15)
    footer(fig,"Tidak digunakan sebagai bukti kondisi mental. Aturan lama memberi label bahkan pada sebagian jendela dengan sensor bermasalah.")
    save(fig,out,"12_heuristik_tersimpan.png","Keluaran heuristik (bukan label kebenaran)",s,dpi)


def plot_question_time(s,out,dpi):
    """Small multiples with a local time axis, keeping retry gaps and NaNs."""
    records=s.questions.to_dict("records")
    for page,start in enumerate(range(0,len(records),6),1):
        batch=records[start:start+6]
        fig,axes=plt.subplots(len(batch),4,figsize=(19,3*len(batch)+1.4),squeeze=False,layout="constrained")
        for row,r in zip(axes,batch):
            mask=(s.segments.study_phase.eq("active_task") & s.segments.difficulty_level.eq(r["difficulty_level"])
                  & s.segments.puzzle_set.eq(r["puzzle_set"]) & s.segments.puzzle_id.eq(r["puzzle_id"]))
            spans=s.segments[mask]
            group=s.eye[(s.eye.study_phase=="active_task") & (s.eye.difficulty_level==r["difficulty_level"])
                        & (s.eye.puzzle_set==r["puzzle_set"]) & (s.eye.puzzle_id==r["puzzle_id"])]
            label=f'{r["difficulty_level"]} | {r["puzzle_id"]}'
            row[0].set_ylabel(label,fontsize=8)
            for ax,title in zip(row,["RMSSD / baseline (siap)","Median EAR per bin","Delta iris (proxy)","Kedipan terdeteksi per bin"]):
                ax.set_title(title,fontsize=9);ax.grid(alpha=.2);ax.set_xlabel("Menit sejak soal mulai",fontsize=8)
            if spans.empty:
                for ax in row:empty(ax,"Tidak ada interval coding aktif")
                continue
            first=float(spans.start_ms.min());last=float(spans.end_ms.max())
            duration=max((last-first)/60000,s.bin_seconds/60)
            hw=s.hrv[s.hrv.eligible_question & s.hrv.difficulty_level.eq(r["difficulty_level"])
                     & s.hrv.puzzle_set.eq(r["puzzle_set"]) & s.hrv.puzzle_id.eq(r["puzzle_id"])]
            if len(hw):row[0].scatter((hw.end_ms-first)/60000,hw.rmssd_ratio,s=9,color="#196da4")
            else:empty(row[0],"Tidak ada HRV siap satu soal")
            row[0].axhline(1,color="#8490a0",ls=":",lw=.7)
            if len(group):
                g=group.groupby("_bin").agg(ear_left=("ear_left","median"),ear_right=("ear_right","median"),
                                             iris_delta=("iris_delta","median"),blinks=("blink_increment","sum"))
                # Reindex restores gaps when a participant leaves and revisits a question.
                g=g.reindex(range(int(g.index.min()),int(g.index.max())+1))
                t=((g.index.to_numpy()+.5)*s.bin_seconds/60)-(first-s.origin)/60000
                line(row[1],t,g.ear_left,"kiri","#196da4");line(row[1],t,g.ear_right,"kanan","#e09925")
                line(row[2],t,g.iris_delta,color="#238e81")
                row[3].bar(t,g.blinks,width=s.bin_seconds/60*.8,color="#8e6ad2")
            else:
                for ax in row[1:]:empty(ax,"Tidak ada frame mata soal ini")
            for ax in row:
                ax.set_xlim(0,duration)
                for span in spans.itertuples():
                    ax.axvspan((span.start_ms-first)/60000,(span.end_ms-first)/60000,
                               color=COLORS.get(r["difficulty_level"],"#999999"),alpha=.07,lw=0)
        fig.suptitle(f"{s.name} — perubahan sensor selama tiap soal | halaman {page}",fontsize=14)
        footer(fig,"Waktu lokal dimulai dari interval coding aktif pertama soal. Kunjungan ulang tetap mengikuti waktu; celah tidak diisi. Titik HRV adalah nilai jendela di waktu akhirnya.")
        save(fig,out,f"13_sensor_per_soal_{page:02d}.png",f"Sensor terhadap waktu tiap soal ({page})",s,dpi)


def table_html(frame):
    return '<div class="scroll">'+frame.to_html(index=False,escape=True,na_rep="—",float_format=lambda x:f"{x:.3f}",border=0)+'</div>'


def document(title,body):
    return '<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'+body+'<footer>EDA deskriptif. Data mentah tidak diubah. File HTML/PNG/CSV dapat dibuka secara offline.</footer></body></html>'


def export_session(s,out):
    s.binned.to_csv(out/"eye_time_bins.csv",index=False)
    s.hrv.to_csv(out/"aligned_hrv_eye_windows.csv",index=False)
    s.questions.to_csv(out/"question_summary.csv",index=False)
    s.levels.to_csv(out/"level_summary.csv",index=False)
    seg=s.segments.copy()
    seg["start_wib"]=[wib(t) for t in seg.start_ms];seg["end_wib"]=[wib(t) for t in seg.end_ms]
    seg["start_minutes"]=(seg.start_ms-s.origin)/60000;seg["end_minutes"]=(seg.end_ms-s.origin)/60000
    seg.to_csv(out/"phase_intervals.csv",index=False)
    s.events.to_csv(out/"events_for_eda.csv",index=False)
    metadata={"subject_session":s.name,"participant_group":participant_group(s.name),
              "original_session_id":s.original_id,"origin_wib":wib(s.origin),
              "origin_definition":"both_calibrated_at_utc" if s.ready_ms is not None else "first_eye_frame",
              "eye_bin_seconds":s.bin_seconds,
              "screen":s.screen,"notes":s.notes,"eye_frames":len(s.eye),"hrv_task_windows":len(s.hrv),
              "baseline_rmssd_ms":s.baseline_rmssd if np.isfinite(s.baseline_rmssd) else None}
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    body='<p><a href="../../index.html">← Semua peserta</a></p>'
    body+=f'<h1>{html.escape(s.name)} — EDA waktu dan soal</h1><p>ID asli: <code>{html.escape(s.original_id)}</code><br>Nol waktu: {html.escape(wib(s.origin))}</p>'
    body+='<div class="note"><strong>Baca sebelum menafsirkan:</strong><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in s.notes)
    body+='<li>Heatmap menunjukkan frekuensi prediksi per frame, bukan kebenaran perhatian atau durasi fiksasi.</li><li>HRV per soal memakai jendela siap yang seluruhnya aktif pada satu soal. Bagian kosong bukan nol.</li><li>EAR/iris/rate mata bersifat estimasi; korelasi bukan sebab-akibat dan jendela tumpang tindih.</li></ul></div>'
    body+='<h2>Ringkasan level</h2>'+table_html(s.levels)
    columns=["difficulty_level","puzzle_id","active_ledger_seconds","active_seconds_in_eye_range","eye_frames","ready_single_question_hrv_windows","submitted_attempts","correct_attempts"]
    body+='<h2>Cakupan setiap soal</h2>'+table_html(s.questions[columns] if len(s.questions) else s.questions)
    body+='<p>CSV: '+ ' · '.join(f'<a href="{name}">{name}</a>' for name in ["question_summary.csv","level_summary.csv","eye_time_bins.csv","aligned_hrv_eye_windows.csv","phase_intervals.csv","events_for_eda.csv","correlation_spearman.csv","correlation_pair_counts.csv"])+'</p>'
    body+='<h2>Grafik</h2>'
    for filename,title in s.charts:
        body+=f'<section class="card"><h2>{html.escape(title)}</h2><a href="{quote(filename)}"><img loading="lazy" src="{quote(filename)}" alt="{html.escape(title)}"></a><small>Klik grafik untuk gambar penuh.</small></section>'
    (out/"index.html").write_text(document(f"{s.name} — EDA",body),encoding="utf-8")


def session_summary(s):
    attempts=[a for a in s.calibration.get("attempts",[]) if a.get("phase")=="held_out_validation"]
    v=attempts[-1] if attempts else {}
    group=s.eye[s.eye.study_phase.eq("active_task")]
    return {"subject_session":s.name,"participant_group":participant_group(s.name),
            "original_session_id":s.original_id,"eye_frames":len(s.eye),"active_eye_frames":len(group),
            "on_screen_active_pct":percent(group.on_screen),"clipping_active_pct_valid":percent(group.loc[group.valid_gaze,"clipped"]),
            "ear_pair_active_available_pct":percent(group.ear_left.notna() & group.ear_right.notna()),
            "iris_active_available_pct":percent(group.iris_delta.notna()),
            "hrv_task_windows":len(s.hrv),"hrv_ready_windows":int(s.hrv.status.eq("ready").sum()),
            "hrv_ready_pct":percent(s.hrv.status.eq("ready")),
            "ready_coding_level_windows":int(s.hrv.eligible_level.sum()),
            "ready_single_question_windows":int(s.hrv.eligible_question.sum()),
            "held_out_median_px":v.get("mean_target_error_median_px",np.nan),
            "held_out_p95_px":v.get("mean_target_error_p95_px",np.nan),
            "held_out_max_corner_px":v.get("max_corner_error_px",np.nan),
            "held_out_original_passed":v.get("passed_thresholds",False),
            "baseline_rmssd_ms":s.baseline_rmssd,"questions_in_ledger":len(s.questions),"plot_count":len(s.charts)}


def plot_cohort(summary,levels,questions,out,dpi):
    charts=[];y=np.arange(len(summary));names=summary.subject_session.tolist()
    fig,axes=plt.subplots(1,3,figsize=(17,max(6,len(names)*.37)),layout="constrained")
    for ax,col,title,color in zip(axes,["ear_pair_active_available_pct","clipping_active_pct_valid","hrv_ready_pct"],
                                  ["EAR tersedia saat coding (%)","Clipping / gaze valid saat coding (%)","HRV tugas siap (%)"],["#30a46c","#d64f64","#196da4"]):
        ax.barh(y,summary[col],color=color);ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_title(title,fontsize=10);ax.grid(axis="x",alpha=.2)
        for i,value in enumerate(summary[col]):
            if np.isfinite(value):ax.text(min(value+1,99),i,f"{value:.1f}",va="center",fontsize=8)
    fig.suptitle("Perbandingan kualitas antarrekaman — ketersediaan bukan akurasi",fontsize=15)
    footer(fig,"Clipping pada grafik ini hanya coding aktif, sehingga dapat berbeda dari persentase seluruh rekaman di laporan audit. Dua sesi Ramadhan adalah satu peserta.")
    save(fig,out,"cohort_quality.png","",dpi=dpi);charts.append(("cohort_quality.png","Kualitas data antarrekaman"))
    fig,ax=plt.subplots(figsize=(17,max(6,len(names)*.4)),layout="constrained")
    for offset,col,label,color in [(-.24,"held_out_median_px","Median","#196da4"),(0,"held_out_p95_px","P95","#e09925"),(.24,"held_out_max_corner_px","Sudut maksimum","#8e6ad2")]:
        ax.barh(y+offset,summary[col],height=.22,label=label,color=color)
    ax.axvline(450,color="#df9d24",ls="--",label="Median audit <450");ax.axvline(900,color="#d64f64",ls="--",label="P95 audit <900")
    ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis();ax.set_xlabel("Kesalahan target rata-rata (px)");ax.legend(fontsize=8,ncol=3);ax.grid(axis="x",alpha=.2);ax.set_title("Validasi gaze tiap rekaman")
    footer(fig,"450/900 hanya kriteria eksploratif audit. Aturan asli 180/350/300; batas sudut tetap dilaporkan. Ambang bukan bukti akurasi di seluruh tugas.")
    save(fig,out,"cohort_calibration.png","",dpi=dpi);charts.append(("cohort_calibration.png","Perbandingan validasi gaze"))
    fig,axes=plt.subplots(2,2,figsize=(16,11),layout="constrained")
    specs=[("rmssd_ratio_median","Median RMSSD / baseline: jendela aktif siap"),("ear_mean_median","Median rata-rata EAR: frame aktif"),
           ("iris_delta_median","Median delta iris (proxy landmark)"),("blink_rate_per_active_minute","Kedipan / menit aktif dalam rentang mata")]
    palette=plt.get_cmap("tab20")
    for ax,(col,title) in zip(axes.flat,specs):
        for i,name in enumerate(names):
            group=levels[levels.subject_session.eq(name)].set_index("difficulty_level").reindex(LEVELS)
            line(ax,np.arange(3),group[col],name,palette(i%20),marker="o",lw=1,alpha=.8)
        ax.set_xticks(range(3),LEVELS);ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    axes[0,0].axhline(1,color="#777777",ls=":",lw=.8)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.035),ncol=min(7,len(names)),fontsize=8)
    fig.suptitle("Perbandingan level: setiap garis adalah satu rekaman",fontsize=15)
    footer(fig,"Titik hilang = tidak tersedia. Ringkasan per rekaman, bukan pooling seluruh frame. Ramadhan dan Ramadhan-2 tidak dihitung sebagai dua peserta independen.",bottom=.15)
    save(fig,out,"cohort_levels.png","",dpi=dpi);charts.append(("cohort_levels.png","Pola fitur pada easy/medium/hard"))
    if not questions.empty:
        data=questions.copy();data["question"]=data.difficulty_level+" / "+data.puzzle_id
        matrix=data.pivot_table(index="subject_session",columns="question",values="rmssd_ratio_median",aggfunc="median",dropna=False).reindex(names)
        matrix.to_csv(out/"cohort_question_rmssd_matrix.csv")
        fig,ax=plt.subplots(figsize=(max(13,len(matrix.columns)*.43),max(6,len(names)*.38)),layout="constrained")
        if matrix.notna().any().any():
            graphic=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),aspect="auto",cmap="viridis");fig.colorbar(graphic,ax=ax,label="Median RMSSD / baseline")
        else:empty(ax,"Tidak ada HRV siap yang dapat diatribusikan ke satu soal")
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8);ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=65,ha="right",fontsize=7)
        ax.set_title("Perbandingan soal antarrekaman: HRV siap pada satu soal")
        footer(fig,"Sel kosong bukan nol; soal cepat dapat tidak memiliki jendela murni 25 detik. Nilai bukan skor kognitif. Soal/set berbeda jangan dianggap identik.")
        save(fig,out,"cohort_questions.png","",dpi=dpi);charts.append(("cohort_questions.png","Peta HRV per soal antarrekaman"))
    return charts


def write_index(summary,charts,out,bin_seconds,inventory=None):
    body='<h1>EDA sesi coding — mata, HRV, waktu, dan soal</h1>'
    body+=f'<p>{len(summary)} rekaman; {summary.participant_group.nunique()} peserta; {int(summary.eye_frames.sum()):,} frame mata; {int(summary.hrv_task_windows.sum()):,} jendela HRV tugas.</p>'
    body+='<div class="note">Grafik bersifat deskriptif. Gaze, iris, Eye_State, dan label kognitif tersimpan adalah estimasi/keluaran algoritme. HRV tidak siap dan bagian tugas tanpa sensor tetap kosong. Korelasi tidak membuktikan sebab-akibat.</div>'
    variants=[r for r in inventory or [] if r["is_variant"] and not r["selected"]]
    if variants:
        body+='<p><strong>Salinan rekaman tidak dihitung ulang:</strong> '+html.escape(", ".join(r["folder"]+" → "+r["canonical_folder"] for r in variants))+'. Lihat <a href="session_inventory.csv">inventaris sumber dan alasan pemilihan</a>.</p>'
    body+=f'<p>Ringkasan waktu mata memakai bin {bin_seconds:g} detik. Label dihitung dari ledger akhir menggunakan ID sesi asli. Waktu nol memakai kedua sensor siap, atau frame mata pertama bila waktu kesiapan tidak tersedia.</p>'
    body+='<h2>Buka peserta</h2><input id="search" placeholder="Cari nama peserta…" aria-label="Cari peserta"><div class="grid">'
    for row in summary.itertuples():
        path="sessions/"+quote(row.subject_session,safe="")+"/index.html"
        body+=f'<div class="card participant" data-name="{html.escape(row.subject_session.lower(),quote=True)}"><h2><a href="{path}">{html.escape(row.subject_session)}</a></h2><p>{row.eye_frames:,} frame · {row.hrv_task_windows} jendela HRV<br>HRV siap {row.hrv_ready_pct:.1f}% · {row.plot_count} grafik</p></div>'
    body+='</div><h2>Ringkasan kualitas</h2>'+table_html(summary)
    body+='<p>CSV: <a href="cohort_summary.csv">ringkasan peserta</a> · <a href="cohort_levels.csv">fitur per level</a> · <a href="cohort_questions.csv">fitur per soal</a> · <a href="run_manifest.json">manifest sumber</a></p>'
    for filename,title in charts:
        body+=f'<section class="card"><h2>{html.escape(title)}</h2><a href="{quote(filename)}"><img loading="lazy" src="{quote(filename)}" alt="{html.escape(title)}"></a></section>'
    body+='''<script>document.getElementById('search').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('.participant').forEach(e=>e.hidden=!e.dataset.name.includes(q));});</script>'''
    (out/"index.html").write_text(document("EDA seluruh saved_session",body),encoding="utf-8")


def discover_sessions(root,subjects=None):
    return select_session_directories(root,subjects)[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description="EDA read-only seluruh saved_session: grafik PNG, laporan HTML offline, dan CSV per soal/waktu.")
    parser.add_argument("--input",type=Path,default=ROOT/"sessions/saved_session",help="Folder semua sesi atau satu folder peserta")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"output",help="Folder keluaran; wajib terpisah dari sumber")
    parser.add_argument("--subjects",nargs="+",help="Nama folder, sama persis termasuk huruf besar/kecil")
    parser.add_argument("--bin-seconds",type=float,default=5.,help="Lebar ringkasan grafik waktu mata (default 5 detik)")
    parser.add_argument("--dpi",type=int,default=115,help="Resolusi PNG (default 115)")
    args=parser.parse_args(argv)
    source,out=args.input.resolve(),args.output.resolve()
    if not source.is_dir():parser.error(f"Folder sumber tidak ada: {source}")
    if not math.isfinite(args.bin_seconds) or args.bin_seconds<=0 or args.dpi<=0:parser.error("bin-seconds dan dpi harus positif dan berhingga")
    if out==source or source in out.parents or out in source.parents:parser.error("Folder output harus terpisah dari sumber, bukan leluhur/turunannya")
    try:sessions,inventory=select_session_directories(source,args.subjects)
    except ValueError as error:parser.error(str(error))
    source_dirs=(sessions if args.subjects else [source/r["folder"] for r in inventory]) if not (source/"eye_tracking.csv").exists() else sessions
    paths=[p for d in source_dirs for p in d.iterdir() if p.is_file() and p.suffix.lower() in [".csv",".json"]]
    before=fingerprint(paths,source)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(inventory).to_csv(out/"session_inventory.csv",index=False)
    for r in inventory:
        if r["is_variant"] and not r["selected"]:print(f"Varian tidak dihitung ulang: {r['folder']} -> {r['canonical_folder']}",flush=True)
    summaries=[];levels=[];questions=[]
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.max_open_warning":5})
    for i,directory in enumerate(sessions,1):
        print(f"[{i}/{len(sessions)}] Membaca dan menyelaraskan {directory.name}...",flush=True)
        s=load_session(directory,args.bin_seconds)
        dest=out/"sessions"/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for function in [plot_overview,plot_gantt,plot_eye,plot_hrv,plot_heatmaps,plot_questions,plot_distributions,plot_correlation,plot_calibration,plot_device,plot_heuristic,plot_question_time]:
            function(s,dest,args.dpi)
        export_session(s,dest)
        summaries.append(session_summary(s));levels.append(s.levels.assign(subject_session=s.name));questions.append(s.questions.assign(subject_session=s.name))
        print(f"  Selesai: {len(s.charts)} grafik, {len(s.questions)} soal, HRV siap {s.hrv.status.eq('ready').sum()}/{len(s.hrv)}",flush=True)
    summary=pd.DataFrame(summaries);all_levels=pd.concat(levels,ignore_index=True);all_questions=pd.concat(questions,ignore_index=True)
    summary.to_csv(out/"cohort_summary.csv",index=False);all_levels.to_csv(out/"cohort_levels.csv",index=False);all_questions.to_csv(out/"cohort_questions.csv",index=False)
    charts=plot_cohort(summary,all_levels,all_questions,out,args.dpi)
    after=fingerprint(paths,source)
    unchanged=before==after
    manifest={"input":str(source),"output":str(out),"subjects":[d.name for d in sessions],"bin_seconds":args.bin_seconds,"dpi":args.dpi,
              "source_inventory":inventory,"input_folder_count":len(inventory),
              "source_files_unchanged":unchanged,"source_file_count":len(before),"sha256":before,
              "charts_per_session":{r["subject_session"]:r["plot_count"] for r in summaries},"cohort_chart_count":len(charts),
              "method":"Final ledger, original IDs; HRV unready -> NaN; exact inclusive sensor windows; per-question HRV excludes mixed-question windows."}
    (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if not unchanged:raise RuntimeError("Berkas sumber berubah selama EDA; periksa manifest dan ulangi pada arsip stabil.")
    write_index(summary,charts,out,args.bin_seconds,inventory)
    print(f"\nSelesai. Buka {out/'index.html'}\n{len(before)} berkas sumber diverifikasi tidak berubah.",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
