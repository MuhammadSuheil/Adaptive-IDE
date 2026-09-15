# Eye Tracking Prototype: Handoff & Implementation Guide

**Date:** 14 September 2026  
**Project:** Adaptive IDE — Webcam Eye Tracking Prototype (`eye_tracking_prototype`)  
**Purpose:** Handoff document for developers, teammates, or future AI sessions to resume work, test the system, and implement the planned calibration overhauls.

---

## 1. Executive Summary & Diagnostic Findings

In recent sessions, we diagnosed major calibration failures and inaccurate gaze tracking (especially around screen edges and across different users). Our empirical diagnosis and analysis of 3 research papers (*Paper 1: ETRA '22*, *Paper 2: Frontiers Psych '24*, *Paper 3: Frontiers Robotics '24*) revealed three primary root causes:

1. **Unbounded Polynomial Extrapolation:** Polynomial regression models output wild screen coordinate predictions when gaze moves near screen corners.
2. **Head Pose Shift Post-Calibration:** Small yaw/pitch head movements change eye geometry relative to the screen, invalidating calibration mapping.
3. **Cross-User & Distance Variation:** Differences in user seating distance and head position create inconsistent pupil/iris feature ranges.

---

## 2. Key User Requirements & Architectural Decisions

Based on our interactive design review, the following decisions were locked in:

| # | Question / Requirement | Agreed Architecture & Design Choice |
|---|------------------------|------------------------------------|
| **1** | **Gate & Calibration UI** | **Fullscreen experience with Camera Preview Inset.** Calibration dots and head alignment bounds are shown fullscreen, with a live camera preview inset in the corner/center guide. |
| **2** | **Calibration Effort / Friction** | **Opt-In & Fast Calibration.** Keep calibration painless and fast (< 30-45 seconds). Avoid long 1-3 minute ordeals. Smooth pursuit calibration is **opt-in** (disabled by default in YAML). |
| **3** | **Point Layout & Modular Grid** | **Fully Modular N-Point Layout Engine.** Point configurations (3×3, 4×4, 13-point, or custom coordinates) are dynamically loaded from `config.yaml` rather than hardcoding point lists. |
| **4** | **Validation Screen Failure Strategy** | **Dual Retry Mode (YAML Configurable).** When held-out validation fails, allow choosing between `restart_full_calibration` OR `retry_validation_only` via YAML so both can be evaluated for better user experience. |
| **5** | **Head Pose Drift Feedback** | **Dual Warning (Text + CSV Logging).** Display non-intrusive text feedback on screen (*"⚠ Head pose shifted — please return to center"*) while simultaneously logging `head_pose_shifted` state in the CSV buffer. |

---

## 3. Immediate Adjustments & Removals (From Earlier Session Pain Points)

> [!IMPORTANT]
> The following two pain points caused major frustration during testing and MUST be updated immediately:

### A. Over-Sensitive Calibration Rejection (RELAX THRESHOLDS)
- **Problem:** Users taking great care to calibrate were still repeatedly rejected due to overly strict quality checks (`min_quality`, `max_sample_std`, `min_feature_span`).
- **Fix:** Relax rejection thresholds in `config.yaml`. Only reject calibration when data is completely corrupt or off-screen (e.g. eyes closed/out of frame).
- **Config changes:**
  - Lower `min_quality` threshold (e.g., from `0.60` to `0.35` or make rejection bypassable).
  - Increase `max_sample_std` and reduce `min_feature_span_x / y`.

### B. Long Calibration Delays (SHORTEN DELAYS)
- **Problem:** Wait time between calibration dots was too long, inflating total calibration duration.
- **Fix:** Make point transition delays significantly faster and fully configurable via YAML:
  - `move_delay_sec`: Reduce default from `2.0s` to `0.5s - 1.0s`.
  - `stability_required_frames`: Reduce default from `6` to `3 - 4`.

---

## 4. Things to Take a Better Look Later (Future Roadmap)

### 1. Modular N-Point Layout Engine
- Abstract calibration point generation. Support standard 3×3 (9-point), 4×4 (16-point), 13-point (3-3-3-2 layout), or custom point lists directly in `config.yaml`:
  ```yaml
  calibration:
    layout_mode: "13_point" # Options: "3x3", "4x4", "13_point", "custom"
  ```

### 2. Session Data Preprocessing & Cleaning
- Evaluate pre-mapping signal conditioning pipeline:
  - **Outlier Rejection:** MAD (Median Absolute Deviation) filtering on raw MediaPipe pupil/iris vectors.
  - **Blink Rejection:** Filter frames where EAR (Eye Aspect Ratio) drops below threshold.
  - **Noise Filtering:** Apply Savitzky-Golay or adaptive EMA smoothing on raw feature vectors before passing them to the RBF mapper.

---

## 5. Updated Configuration Schema (`config.yaml`)

Below is the complete proposed `config.yaml` additions and overrides to support Phase 0–4:

```yaml
# ─────────────────────────────────────────────
# Head Positioning Gate (Phase 0)
# ─────────────────────────────────────────────
head_positioning:
  enabled: true
  fullscreen: true
  target_face_width_ratio: 0.35
  target_center_x_ratio: 0.5
  target_center_y_ratio: 0.45
  alignment_tolerance_ratio: 0.08
  size_tolerance_ratio: 0.10
  max_yaw_deg: 10.0
  max_pitch_deg: 8.0
  stability_frames_required: 15
  countdown_seconds: 3

# ─────────────────────────────────────────────
# Head Pose Monitoring (Phase 3)
# ─────────────────────────────────────────────
head_pose:
  max_calib_yaw_deg: 8.0
  max_calib_pitch_deg: 6.0
  tracking_warn_yaw_deg: 15.0
  tracking_warn_pitch_deg: 12.0
  show_text_warning: true  # Displays "⚠ Head pose shifted" text on screen

# ─────────────────────────────────────────────
# Calibration & Gaze Mapping (Phase 1 & 2)
# ─────────────────────────────────────────────
calibration:
  layout_mode: "13_point"    # "3x3" | "4x4" | "13_point" | "custom"
  n_samples_per_point: 20    # reduced for speed
  move_delay_sec: 0.8        # SHORTER DELAY (was 2.0s)
  target_margin: 0.08
  dot_radius: 20
  stability_threshold: 0.025 # RELAXED (was 0.015)
  stability_required_frames: 4
  mapping_method: "rbf"      # Bounded Thin-Plate Spline RBF Mapper
  rbf_kernel: "thin_plate_spline"
  rbf_smoothing: 0.0
  output_clamp: true         # Prevents gaze coordinate from leaving screen bounds
  min_quality: 0.35          # RELAXED (was 0.60)
  smooth_pursuit_enabled: false # OPT-IN (default off for speed)

# ─────────────────────────────────────────────
# Post-Calibration Validation (Phase 4)
# ─────────────────────────────────────────────
validation:
  enabled: true
  failure_strategy: "retry_validation_only" # Options: "full_restart" | "retry_validation_only"
  max_median_error_px: 180
  max_p95_error_px: 350
  max_corner_error_px: 300
```

---

## 6. Implementation Architecture & Code Roadmap

### Phase 0: Head Positioning Gate (`run_head_positioning_gate()`)
- Shows a camera preview inset on fullscreen overlay.
- Computes face bounding box, inter-ocular distance (IOD), and head yaw/pitch.
- Guides the user into the neutral position before calibration begins.

### Phase 1: Modular Calibration Grid & Shorter Delays
- Uses `layout_mode` to select dot coordinates dynamically.
- Uses shorter delays (`move_delay_sec: 0.8`) and relaxed stability metrics.

### Phase 2: Bounded RBF Mapper (`modules/mapper.py`)
- Replaces unbounded high-degree polynomial regression with `scipy.interpolate.RBFInterpolator` (`thin_plate_spline`).
- Clamps output coordinates strictly to `[0, screen_width]` and `[0, screen_height]`.

### Phase 3: Head Pose Gating & Warning Text
- Captures baseline yaw/pitch during Phase 0.
- Rejects individual calibration samples if user turns head.
- Displays text warning on tracking HUD if head shifts beyond `tracking_warn_yaw_deg`.

### Phase 4: Held-Out Validation Screen
- Displays 5 validation dots after calibration.
- Evaluates median and corner pixel error.
- Executes `failure_strategy` (`full_restart` or `retry_validation_only`) if validation threshold is exceeded.

---

## 7. How to Resume Work / Quickstart

1. **Test Existing Prototype:**
   ```bash
   python eye_tracking_prototype/eye_tracking_prototype.py
   ```
2. **Apply Config Updates:** Edit `eye_tracking_prototype/config.yaml` with the relaxed thresholds and shorter delays shown in Section 5.
3. **Implement RBF Mapper:** Update `eye_tracking_prototype/modules/mapper.py` to add `RBFInterpolator`.
4. **Implement Gate & Validation:** Add `run_head_positioning_gate()` and `run_validation_screen()` to `eye_tracking_prototype/eye_tracking_prototype.py`.

---
*Document prepared for handoff & team collaboration.*
