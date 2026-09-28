# Camera performance verification — 2026-09-28

Environment: Windows, Python 3.10.11, OpenCV 5.0.0, configured device index 1,
640×480 capture and 320×240 MediaPipe input. All runs exclude a 3-second warmup
from consumer results. Images were not saved. Measurements describe host delivery,
not sensor exposure timestamps or guaranteed distinct sensor images.

| Experiment | Measured FPS | Observation |
| --- | ---: | --- |
| DirectShow, default format, 20 s | 39.57 | Negotiated YUY2, reported 60 FPS |
| MJPG request before resolution/FPS, 20 s | 39.57 | Final format reverted to YUY2 despite successful set |
| Default format with Windows timer 1 ms, 20 s | 39.58 | No material improvement |
| MSMF at index 1, 20 s | 29.89 | Reported 30 FPS; device index identity across backends not independently verified |
| MJPG reapplied after resolution/FPS, 20 s | 60.00 | Final format MJPG; all requested mode properties matched |
| MJPG + threaded MediaPipe, final 60 s run | 59.18 processing / 59.96 capture | 46 skipped frames, 1.28%; 2,306 face frames out of 3,551 |

The controlled DirectShow comparison isolates a material configuration issue:
the default capture format and property-setting order limited host delivery to
about 40 FPS on this setup. Reapplying MJPG after other mode settings produced
60 FPS without reducing resolution or changing exposure. This does not establish
the underlying driver/USB mechanism; CPU and disk optimization was not needed
to remove this particular limit.

Final inference processing P95 was 14.38 ms and P99 21.61 ms. Capture interval
P95 remained 32.06 ms, so average 60 FPS does not mean evenly spaced delivery.
The final run passed >=58 FPS but did NOT pass the proposed <1% skipped-frame
and <=20 ms capture-interval P95 criteria. Face presence varied during the run.

An earlier 60 s inference run measured 59.60 FPS with 0.64% skips and 3,485/3,576
face frames, but encountered a report-variable shadowing bug during cleanup
after saving results. That bug was fixed and regression-tested; the final run
above exited successfully and closed capture/model resources.

## Raw reports

- [Default YUY2](../tmp/camera_benchmark_20260928_184508_986446.json)
- [MJPG request reset to YUY2](../tmp/camera_benchmark_20260928_184703_021313.json)
- [Timer experiment](../tmp/camera_benchmark_20260928_184805_071605.json)
- [MSMF experiment](../tmp/camera_benchmark_20260928_184922_786486.json)
- [MJPG after order correction](../tmp/camera_benchmark_20260928_185019_102674.json)
- [Final threaded inference run](../tmp/camera_benchmark_20260928_185445_235516.json)

## Remaining acceptance check

Run normal calibrated tracking for five minutes, once without tracking UI and
once with it (commands in README). This exercises mapping, normalized iris,
CSV and display in addition to the live inference path tested here. Calibration
requires participant interaction and was not bypassed or fabricated for this test.
New iris deltas use eye-width-normalized units and must not be compared numerically
with old image-radius deltas. Frame-based fixation/blink/smoothing thresholds
still need evaluation at the increased sampling rate.
