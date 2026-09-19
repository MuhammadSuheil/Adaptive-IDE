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
