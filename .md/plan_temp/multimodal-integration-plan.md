# Multimodal Data Integration Plan: Eye Tracking & HRV Monitor
**Document Created:** 2026-09-19  
**Status:** Planning Completed & Aligned with Team Requirements  
**Target Goal:** Fuse Eye Tracking and BLE Heart Rate Variability (HRV) sensor streams into a single Python application to compute real-time Cognitive Load ($CLS \in [0, 100]$) in AdaptiveIDE.

---

## 1. Context & Objectives

The goal is to merge data from two independent sensors:
1. **Eye Tracking Prototype**: MediaPipe FaceLandmarker webcam tracking (~30–40 FPS), providing Gaze X/Y, Active IDE Region (`main_file`, `sidebar`, `ai_agent`, `terminal`), Eye Aspect Ratio (EAR), Blink Rate (BPM), and Head Pose.
2. **HRV Monitor Prototype**: BLE (Bleak) connection to Polar/HW9 heart rate monitor (~1 Hz RR interval stream), computing Low Frequency (LF) spectral power ($0.04 - 0.15\text{ Hz}$) and Z-scores relative to baseline.

### Primary Challenges Addressed
- **Timestamp Synchronization**: Eliminating drift from independent sensor start times by locking a unified UTC Epoch millisecond reference timestamp ($T_0$).
- **Unequal & Simultaneous Calibration**: Handling the 60-second HRV baseline requirement (user sitting still) while executing the eye tracking 13-point target calibration grid concurrently.
- **Asymmetric Frequency Resampling**: Resampling 30–40 Hz ocular data and 1 Hz cardiac data into synchronized 1-second time buckets (`fused_cognitive_load_1hz.csv`).
- **Cognitive Load Quantification**: Replacing vague qualitative states with a standardized Cognitive Load Score ($0.0 - 100.0$) and 4 range bands.

---

## 2. Technical Architecture & Execution Model

The system will run as a **single unified Python application** (`multimodal_app.py`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Single Python App (multimodal_app.py)                     │
├──────────────────────────────────────────┬──────────────────────────────────┤
│ Main Thread (OpenCV & MediaPipe)         │ Background Thread (Asyncio/Bleak)│
│ - WebCam Frame Capture (~30-40 FPS)      │ - BLE Device Scanner & Client    │
│ - Gaze & EAR Blink Pipeline              │ - HR Measurement Stream          │
│ - Live OpenCV HUD & Target Grid          │ - RR Interval Artifact Filter    │
└──────────────────────────────────────────┴──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Shared Synchronized Memory & Queue                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ - Simultaneous 60s Calibration Controller                                   │
│ - Unified Global Clock Reference (T0 UTC ms)                                │
│ - 1-Second Sliding Window Feature Aggregator                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Dual Output & Persistence                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ - fused_cognitive_load_1hz.csv (Synchronized 1Hz Time Buckets)              │
│ - eye_tracking_raw.csv & hrv_raw.csv (Full-resolution raw logs)             │
│ - session_summary.json (Metadata schema v2.0)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Simultaneous 60-Second Calibration Workflow

To accommodate the HRV requirement of **60 seconds of quiet sitting without arm movement**, calibration for both sensors takes place concurrently over a fixed 60-second window:

| Time Window | Eye Tracking Calibration Phase | HRV Calibration Phase | User Guidance / Instructions |
| :--- | :--- | :--- | :--- |
| **0s – 10s** | Phase 0 Head Pose Alignment Check (Yaw/Pitch/Roll) | Accumulating initial RR intervals | *"Sit still, align head in HUD box"* |
| **10s – 45s** | 13-Point Gaze Calibration Target Grid (moving dots) | Accumulating baseline RR intervals | *"Keep body & arms still; follow target dot with eyes only"* |
| **45s – 60s** | Center fixation validation & stability check | Finalizing baseline LF Mean & STD | *"Fixate on screen center; finalizing resting baseline..."* |
| **At t = 60s** | Lock gaze mapper model | Save $\mu_{\text{baseline\_LF}}, \sigma_{\text{baseline\_LF}}$ | **LOCK $T_0$ TIMESTAMP & START TRACKING** |

---

## 4. Cognitive Load Scoring & Range Classification

### Formula Structure
1. **Raw Composite Z-Score ($Z_{\text{load}}$)**:
   $$Z_{\text{load}}(t) = w_{\text{hrv}} \cdot Z_{\text{HRV\_LF}}(t) + w_{\text{blink}} \cdot Z_{\text{Blink\_Suppression}}(t) + w_{\text{dwell}} \cdot Z_{\text{Fixation\_Dwell}}(t)$$
   - Weights: $w_{\text{hrv}} = 0.45$, $w_{\text{blink}} = 0.35$, $w_{\text{dwell}} = 0.20$.
   - $Z_{\text{Blink\_Suppression}} = \frac{\text{BPM}_{\text{baseline}} - \text{BPM}(t)}{\sigma_{\text{BPM}}}$ (Blink rate drops during high focus).

2. **Normalized Cognitive Load Score ($CLS \in [0.0, 100.0]$)**:
   $$CLS(t) = \frac{100.0}{1.0 + e^{-1.2 \cdot Z_{\text{load}}(t)}}$$

### Cognitive Load Range Bands
| Range Band | $CLS$ Score Range | $Z_{\text{load}}$ Range | Label Key | Mental State / UX Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Band 1** | **0.0 – 25.0** | $Z < -1.0$ | `LOW_LOAD` | Deep rest, low engagement, idle reading |
| **Band 2** | **25.1 – 60.0** | $-1.0 \le Z \le +0.5$ | `OPTIMAL_LOAD` | Productive flow state, balanced focus, coding |
| **Band 3** | **61.0 – 85.0** | $+0.5 < Z \le +1.8$ | `HIGH_LOAD` | Heavy mental exertion, complex bug hunting |
| **Band 4** | **85.1 – 100.0** | $Z > +1.8$ | `VERY_HIGH_LOAD` | Cognitive overload, high mental fatigue risk |

---

## 5. Output Data Schema (`fused_cognitive_load_1hz.csv`)

| Header Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `timestamp_utc_ms` | uint64 | Unix Epoch ms | Absolute master clock timestamp |
| `elapsed_session_sec` | float | Seconds ($\ge 0.0$) | Time elapsed since $T_0$ session trigger lock |
| `eye_gaze_x` | float | Pixels ($0 - W$) | Screen X coordinate prediction |
| `eye_gaze_y` | float | Pixels ($0 - H$) | Screen Y coordinate prediction |
| `active_ide_region` | string | Categorical | `main_file`, `sidebar`, `ai_agent`, `terminal`, or `off_screen` |
| `blink_rate_bpm` | float | Blinks/min | Real-time 60s window blink rate |
| `hrv_rr_last_sec` | float | Seconds | Duration of latest RR interval |
| `hrv_lf_power` | float | $\text{ms}^2$ or normalized | Spectral power in $0.04 - 0.15\text{ Hz}$ band |
| `hrv_lf_zscore` | float | Z-score | HRV LF deviation from 60s baseline |
| `z_load_composite` | float | Z-score | Raw composite load Z-score |
| `cognitive_load_score` | float | $0.0 - 100.0$ | Sigmoid-normalized Cognitive Load Score ($CLS$) |
| `cognitive_load_band` | string | Categorical | `LOW_LOAD`, `OPTIMAL_LOAD`, `HIGH_LOAD`, or `VERY_HIGH_LOAD` |

---

## 6. Implementation Roadmap for Next Session

1. **Scaffold Entrypoint**: Create `multimodal_app.py` combining `eye_tracking_prototype.py` and `hrv_monitor.py` Bleak client.
2. **Implement Simultaneous Calibration**: Wire up the 60-second joint calibration loop with visual HUD timer.
3. **Build Fusion Engine**: Implement the 1-second time-bucket aggregator and composite score ($CLS$) calculator.
4. **Live HUD HUD Display**: Render live color-coded Cognitive Load Meter bar on the OpenCV window.
