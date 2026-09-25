# Adaptive IDE Interface based on Developer's Cognitive Load using Machine Learning and Multimodal Sensor Fusion

## Overview
This repository contains the software components and research context for developing an Adaptive Integrated Development Environment (IDE) extension. The core objective of this research is to apply Machine Learning (ML) and Artificial Intelligence (AI) to dynamically alter the IDE interface to reduce cognitive load and assist developers based on their current cognitive state.

## Research Team
**Head of Research:**
- Hadipurnawan Satria, Ph.D

**Researchers:**
1. Anggina Primanita, S.Kom., M.I.T., Ph.D.
2. Julian Supardi, S.Pd., M.T., Ph.D
3. Alvi Syahrini Utami, S. Si., M.Kom

**Student Researchers:**
- **Eye Tracking Sub-team:** M. Suheil Ichma Putra, M. Rabyndra Janitra Binello, Monica Amrina Rosyada
- **Heart Rate Sub-team:** Fitran Husein, Muhammad Alif Berri Rossi

## HRV Prototype: Current Implementation

The [HRV prototype guide](hrv_monitor_prototype/README.md) explains the current
RR-only workflow, configuration, filtering, and output files. Start a complete
session from the repository root:

```powershell
python -m pip install -r hrv_monitor_prototype/requirements.txt
python -m hrv_monitor_prototype.calibrate
```

Each session collects a seated personal baseline targeting 120 seconds of accepted
RR data, with a maximum calibration time of 300 seconds. After calibration succeeds,
recording, preprocessing, and HRV calculation continue automatically. With the
current configuration, task windows are 25 seconds long and advance every 5 seconds.
RMSSD and SDNN appear in the terminal when the data meets the prototype's recovery
and sufficiency checks.

Each session creates only `rr_raw.csv` and `hrv.csv`. Phase columns distinguish
calibration/baseline from task data. The prototype currently stops at HRV features;
cognitive-load estimation and automatic IDE adaptation remain research objectives.
BPM is not used as an input feature.

`record_rr.py` remains a required module: `calibrate.py` imports its BLE recording
function and session output directory. Run `calibrate` for the complete workflow;
run `record_rr` directly only when collecting raw RR without calibration or HRV.

## Multimodal session output

Eye calibration and the HRV baseline run concurrently. HRV text is not drawn over
the eye targets. After eye validation is accepted, a dedicated HRV screen shows
the remaining accepted RR needed, preserving the progress already collected.
If the baseline is already ready, this screen is skipped. The baseline targets
120 seconds of accepted RR within 300 seconds from HRV recording startup.
Task recording starts when both calibrations are ready. If HRV finishes first,
subsequent raw RR is marked `waiting_for_eye` until eye validation is accepted.
HRV-only mode does not wait for the eye tracker; eye-only mode does not wait for HRV.

Run `python run_multimodal.py --participant P01 --task coding_task` from the
repository root. Once either recorder exits (including pressing Q in the eye
tracker and dismissing its summary), the runner stops the other recorder, waits
for its files to close, and automatically combines the session data. Ctrl+C also
ends the session and triggers fusion.

A session with both recordings produces four CSV files in `sessions/<session>/`:

- `eye_tracking.csv`: eye tracking frames and timestamps.
- `hrv.csv`: baseline and task HRV windows.
- `rr_raw.csv`: original RR measurements.
- `multimodal_timeline.csv`: one row per task HRV window, containing HRV features
  and eye tracking features aggregated over the same timestamp range.

The raw CSV files are preserved. JSON summaries are additional outputs. If a
session has no task HRV windows yet, the combined CSV contains its header only;
missing source recordings are reported in the terminal. Single-sensor modes
(`--skip-eye` or `--skip-hrv`) do not produce a combined CSV.

`hrv_calib_status.json` carries live baseline progress, and `eye_calibration_ready`
signals that eye validation has been accepted so task recording can begin once
HRV is also ready. These are session coordination
files, not measurement data. The runner passes the required waiting flags to
both recorders automatically.

To combine an existing session, run:

```powershell
python fuse_and_analyze.py --session-dir "sessions/<session>"
```

## General Workflow & Methodology

The research utilizes a multimodal sensor fusion approach to detect the developer's cognitive state while they code. The system gathers data from two primary physiological sources:

1. **RR Intervals / Heart Rate Variability (HRV):**
   - Utilizing a PPG sensor (such as the Coospo HW9 HRM Arm Band) to collect beat intervals and derive RMSSD and SDNN.
   - A dedicated application communicates with the device via Bluetooth Low Energy (BLE) to gather physiological responses that correlate with stress or high cognitive load.

2. **Eye Movement (Eye Tracking):**
   - Utilizing standard or high-framerate (60fps) webcams to track iris movements.
   - Processing the video feed using Machine Learning (e.g., MediaPipe) to map eye gaze to specific coordinates on the screen and editor lines.

### System Architecture Concept
- **Data Collection:** Dedicated services run locally to capture RR intervals and Eye Tracking data in real-time.
- **Data Fusion & Processing:** The streams of data are synchronized (Multimodal Sensor Fusion). A machine learning model or algorithmic logic infers the cognitive state of the developer (e.g., *Focused, Confused, Scanning, Overloaded*).
- **Adaptive IDE Extension (VS Code):** A custom IDE extension acts as the client that receives the inferred state. It automatically triggers UI/UX changes. For instance:
  - Hiding the sidebar and entering Zen Mode during deep focus.
  - Enlarging font sizes or suggesting documentation when the developer appears confused or encounters errors.
  - Simplifying the interface to reduce visual noise when the developer is overloaded.
