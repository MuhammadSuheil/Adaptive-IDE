# Eye Tracking Prototype

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
- The purple padding is only a tracking classification margin; it does not control dot placement.
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
`eyes_invalid_or_blink`. The padding-based state does not by itself prove that a
person looked away; missing/invalid landmark states are recorded separately.
