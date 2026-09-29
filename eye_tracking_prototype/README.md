# Eye Tracking Prototype

Set the calibration/tracking screen dimensions manually in `config.yaml`:

```yaml
screen:
  width: 2880
  height: 1800
```

These required positive integer values define the screen coordinate system;
they are independent of webcam capture resolution. Match the fullscreen monitor
resolution to avoid display scaling affecting target positions. Restart the
application and recalibrate after changing them. Custom config files must also
include this section. Automatic screen-size detection is no longer used.

The head alignment step displays a live camera preview with a circle centered on
the screen. Forehead, chin and cheek landmarks must fit and fill that circle,
with the face centered and upright. A continuous five-second green countdown
is required; losing alignment or the camera resets it. Space/Enter cannot skip it.

Eye calibration defaults to nine targets (`layout_mode: "3x3"`,
`target_margin: 0.0`): four physical screen corners, four side midpoints, and
the center. Targets on the pixel boundary are intentionally clipped; look at
the point where the target meets the screen edge. Calibration has no padding.
The tracking display retains its visual off-screen padding.

This is the standalone Python prototype for the Eye Tracking Module of the Adaptive IDE Extension. It uses MediaPipe to extract facial landmarks (specifically the iris) and an OpenCV grid to map your gaze to different sections of the IDE.

## Requirements

Ensure you have Python 3.10+ installed.

Dependencies are managed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

*(Note: The `filterpy` package is optional but required if you want to use the Kalman filter instead of the default EMA filter).*

You also need the MediaPipe Face Landmarker model. It should already be inside the `models/` folder as `face_landmarker.task`. If it is missing, you can download it from [Google's Storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task).

## Running the Prototype

To start the eye tracking prototype, simply run:
```bash
python eye_tracking_prototype.py
```

You can optionally specify a custom configuration file:
```bash
python eye_tracking_prototype.py --config custom_config.yaml
```

## User Flow

### 1. Calibration Phase
When you launch the script, it will first enter the calibration phase.
- The screen displays nine red dots sequentially, kept safely inside the physical screen edges.
- **Look at the center of the visible red dot itself.** Do not look at the purple padding border.
- Keep your head still and move only your eyes. Blink normally between points where possible.
- The colored padding is visualization-only; it does not control calibration or grid classification.
- Calibration uses only unique camera frames; repeated reads of the latest frame do not count as new samples.
- Closed-eye, unstable, and noisy samples are rejected. A timed-out point or weak overall calibration must be retried.
- The final score uses leave-one-point-out validation. Tracking cannot start below `calibration.min_quality`.

### 2. Tracking Phase
After successful calibration, two windows will open:
1. **Eye Tracking - Camera Feed**: Shows your raw webcam feed, facial mesh, highlighted iris centers, and a live HUD of tracking metrics.
2. **Eye Tracking - Gaze Grid**: Shows a full-screen grid mapping. The cell you are currently looking at will be highlighted in color, and your exact estimated gaze point is shown as a red dot.

## Controls

While the tracking windows are active, you can use the following keyboard controls:
- `q` : **Quit** the application. This will flush the final CSV data and write the JSON session summary.
- `c` : **Recalibrate**. Restarts the calibration flow if tracking feels inaccurate.
- `p` : **Pause/Resume** tracking. Useful if you need to step away without ending the session.
- `g` : **Toggle Grid**. Shows or hides the Gaze Grid window.
- `s` : **Snapshot**. Saves the current camera frame to your disk as a PNG image.
- `d` : **Debug Mode**. Toggles the full facial mesh visualization on the camera feed.

## Configuration (`config.yaml`)

All parameters are tunable without touching the Python code. You can adjust:
- **Grid Size**: Number of rows and columns, and the label mapped to each cell.
- **Smoothing Filter**: Choose between `ema`, `kalman`, `median`, or `none`.
- **Webcam Options**: Target FPS, resolution, and device index.
- **Inference Resolution**: MediaPipe can process a smaller frame than the displayed camera feed.
- **Calibration Validation**: Target margin, point timeout, noise limits, feature separation, and minimum quality.
- **Dwell Threshold**: The minimum time (in ms) before a gaze is counted as intentional.

## Data Output

All session data is saved into the `sessions/` directory. For each session, two files are created:
1. `session_<uuid>_<timestamp>.csv`: Contains per-frame gaze data plus processing FPS, capture FPS, inference time, and an explicit gaze-validity state.
2. `session_<uuid>_<timestamp>_summary.json`: Includes camera diagnostics, held-out calibration errors, average processing FPS, dwell times, and visit counts.

Possible gaze states are `on_screen`, `gaze_outside_screen`, `face_missing`, and
`eyes_invalid_or_blink`. `gaze_outside_screen` now requires a predicted coordinate
outside the full physical screen; missing/invalid landmark states are recorded separately.

The black grid is a scaled representation of the **entire physical screen**.
Colored padding remains visible around that representation, but it is not removed
from logical screen coordinates and never changes grid-cell classification.

`webcam.width` and `webcam.height` are requested capture settings rather than
code constants; the negotiated values are recorded under `summary.camera`.
`inference_width` and `inference_height` intentionally control a separate
MediaPipe input copy so accuracy/performance can be tuned without changing the
screen-coordinate mapping, which uses normalized landmarks.

## Performance Pipeline

Tracking uses three decoupled stages: webcam capture, MediaPipe inference, and
OpenCV display. The inference worker always consumes the newest unique camera
frame, so a slow fullscreen window cannot build a stale-frame queue or reduce
the gaze-processing rate. The camera preview and grid default to 15 FPS while
tracking runs at the maximum rate supplied by the camera and processor.

Performance values have distinct meanings:

- `capture_fps`: host-delivered frames per second over a rolling 2-second window.
- `fps_actual`: processed frame intervals / monotonic elapsed time over a rolling
  2-second window (not the previous EMA of instantaneous reciprocals).
- `ui_fps`: display refresh rate, recorded in the JSON summary.
- `dropped_frame_ratio`: camera frames intentionally skipped to keep latency low.
- `stage_timings_ms`: mean, P50, and P95 timings for every processing stage.

### Camera configuration and benchmarks

`webcam.backend` selects `auto` (DirectShow on Windows), `dshow`, `msmf`, or `any`.
`webcam.fourcc` selects a requested format; `null` leaves the backend default.
The supplied configuration uses MJPG. DirectShow can reset FourCC when resolution
or FPS changes, so the opener checks and reapplies the requested format last.
Diagnostics retain every set result and final readback, including mismatches.
`buffer_size` is optional: unsupported values are reported, not assumed effective.
Exposure is read for diagnostics but is never changed automatically.

Run from the repository root (close other camera consumers first):

```powershell
python eye_tracking_prototype/benchmark_camera.py --fourcc default --seconds 60
python eye_tracking_prototype/benchmark_camera.py --fourcc MJPG --seconds 60
python eye_tracking_prototype/benchmark_camera.py --stage preprocess --threaded --seconds 60
python eye_tracking_prototype/benchmark_camera.py --stage inference --threaded --seconds 60
```

The first two measure raw synchronous reads. `--threaded` uses the application's
latest-frame capture thread and reports frames skipped by the consumer. Without
this flag, processing is synchronous, so that throughput is not directly equivalent
to the application. A 3-second warmup is excluded from consumer statistics;
`capture_thread` statistics include warmup and use up to the last 36,000 timings.
Inference results include `face_frames`: face-absent performance is not evidence
of sustained face tracking performance. No images are saved. JSON reports are
created under `tmp/` with unique filenames, or use `--output path.json` (no overwrite).
Optional `--timer-ms 1` tests a Windows timer request scoped to the benchmark.

For full calibrated tracking (iris, mapping, CSV), with or without display:

```powershell
python eye_tracking_prototype/eye_tracking_prototype.py --tracking-seconds 300 --no-tracking-ui
python eye_tracking_prototype/eye_tracking_prototype.py --tracking-seconds 300
```

Calibration remains interactive. The duration starts after calibration/HRV readiness.
Compare `summary.avg_fps`, frame drops, processing P95 and camera read timings.
Aim for capture and processing >=58 FPS and <1% consumer skips, and separately
check capture jitter (P95 <=20 ms). Meeting average FPS alone does not imply smooth
delivery or prove distinct sensor exposures. Frame IDs count successful host reads;
they cannot detect frames lost inside the driver or repeated images from the device.

### Timestamp and iris schema (version 3)

CSV `timestamp_ms` now uses host camera-read completion, converted from a monotonic
clock to an epoch anchor. `processing_timestamp_ms` records worker start separately.
This is not a hardware exposure timestamp and does not measure driver buffering.
In synchronous mode capture time follows the read/flip helper. Summary `avg_fps`
uses the full first-to-last processing span, including any pause/recalibration;
rolling FPS restarts after a pause. `total_processing_ms` in CSV excludes logging;
the JSON stage summary includes logging but excludes capture wait and UI work.

Iris baseline requires 1.5 seconds of contiguous valid observations; invalid
samples or gaps >250 ms restart an unfinished baseline. `baseline_frames` is kept
only for old configuration compatibility. Blank `iris_size_delta` means invalid
or baseline not ready, never a repeated previous measurement. CSV includes
`iris_valid`, `iris_baseline_ready`, `iris_size`, and `iris_size_mode`; fusion excludes
invalid, missing, and blinking samples. Head-pose warnings also invalidate new iris
samples. Baseline is fixed after collection.

The default `eye_width_ratio` measures average iris landmark radius divided by eye
width using aspect-correct geometry. Values are NOT numerically comparable to old
`image_radius` deltas. Choose `iris.size_mode: image_radius` to retain the old units.
Neither mode measures pupil diameter; iris landmark geometry is not validated
pupillometry. FPS changes also affect existing frame-based blink/fixation/smoothing
parameters; those classifiers have not been retuned by this performance change.

### Calibration and gaze stability update (2026-09-29)

Head alignment now smooths guide measurements over 0.12 seconds and uses a
small exit tolerance (`hysteresis_ratio`) after entering the accepted region.
Being inside the oval alone is insufficient: face fill, centering, angle and
measured movement must pass. Movement uses the P10-P90 spread of raw anchors
and angles over `stability_seconds` (0.5 seconds), so smoothing cannot hide
sustained movement. The five-second hold starts after stability is established.
Brief invalid intervals pause the hold; intervals longer than 0.35 seconds
reset it. The baseline is the median of accepted stable pose samples, rather
than the final frame. The old `stability_frames_required` and
`invalid_grace_frames` settings no longer control this gate.

Calibration, validation and tracking require BOTH eyes to have finite EAR within
`max(eye_validity.min_ear, blink.ear_threshold)` and `eye_validity.max_ear`.
Closed-eye samples do not update gaze mapping or smoothing, even when blink
counting is disabled. Invalid gaze coordinates are blank in CSV and the grid
shows `GAZE UNAVAILABLE`. A gap longer than 0.30 seconds resets filter history.

EMA alpha now accounts for elapsed capture time, with `reference_fps: 40`
preserving the nominal response of the previous alpha settings. Adaptive EMA
uses velocity in pixels/second. Large jumps exceeding BOTH 250 pixels and
12,000 pixels/second require a consistent candidate for 0.035 seconds before
acceptance (about 50 ms with evenly spaced 60 FPS samples). Pending candidates
have status `gaze_unconfirmed`, blank gaze coordinates and no valid iris delta.
This guard applies to all filter modes, including `none`; the median window
remains frame-based. A genuine large saccade can incur this confirmation delay.

Each session writes `<summary_stem>_calibration.json` beside its summary
(`eye_summary_calibration.json` in shared multimodal sessions). It includes
configuration, gate rejection durations/resets, the median pose baseline,
accepted calibration samples and retained features, model/LOO diagnostics,
held-out target predictions/errors, and automatic/manual/retry decisions.
Rejection durations can overlap because several checks can fail together.
Audits are saved at phase boundaries and cleanup, including failed attempts;
there is no additional per-frame tracking file write. An interrupted attempt
may remain `incomplete`.

The summary's legacy `validation_*` model diagnostics are LOO estimates.
Actual held-out results are recorded separately in `held_out_validation` and
the audit. Existing pass thresholds use errors of each target's mean prediction;
new per-sample median/P95 errors expose jitter that averaging can conceal.
Manual acceptance is recorded separately from `passed_thresholds`.

Capture format/resolution, inference size and the decoupled pipeline remain
MJPG 640x480, 320x240 and target 60 FPS. This update does not change RBF features
or implement head-motion compensation. Those require the new calibration data
and a separate model comparison; smoothing cannot correct a systematically
inaccurate mapping. See [verification results](PERFORMANCE_RESULTS.md) for
automated checks and the remaining interactive acceptance run.
