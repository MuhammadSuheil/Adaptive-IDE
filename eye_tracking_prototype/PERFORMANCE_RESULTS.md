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

## Stability update verification - 2026-09-29

Implemented time-based head alignment with hysteresis, raw-motion stability,
median pose baseline, per-eye blink/validity rejection, capture-time EMA,
large-jump confirmation and calibration/held-out validation audit records.
MJPG negotiation, capture/inference resolutions and worker architecture were
not changed. This is stages 1-4 of the stability work; EAR feature ablation and
head-motion compensation remain separate follow-up work.

Automated verification:

| Check | Result |
| --- | --- |
| Repository tests: `python -m unittest discover -s tests -q` from root | 28 passed |
| Eye module tests: same command from `eye_tracking_prototype` | 24 passed |
| Synthetic tracking filter + eye-validity check, 10,000 samples | Mean 0.00335 ms; P95 0.00470 ms; P99 0.0119 ms |
| Synthetic head gate, 1,000 samples (calibration only) | Mean 0.247 ms; P95 0.531 ms; P99 0.648 ms |

Tests cover gate timing at 20/40/60 updates per second, median baseline,
dropout pause/reset, movement rejection, oval fill/hysteresis, EMA elapsed-time
response, isolated-spike rejection versus sustained steps, long-gap reset,
one-eye closure, worker invalid-frame output and audit serialization.
Synthetic timings are local microbenchmarks, not a full camera FPS guarantee;
no new interactive camera session has been measured for this stability update.
The live results at the top of this document predate this update.

Next acceptance run: use the normal multimodal launcher or the five-minute
calibrated tracking commands in README. Record time to pass the head gate,
its rejection reasons/resets, and whether validation required manual override.
During tracking, look at known center/edge targets, blink normally, then make
small comfortable head shifts. Compare capture/processing FPS and skipped
frames against the prior runs. Inspect held-out per-sample error and valid-gaze
off-screen rate together with `eyes_invalid_or_blink` / `gaze_unconfirmed`
fractions: fewer reported off-screen frames alone is not evidence of improved
accuracy when invalid samples are now excluded. Evaluate iris baseline readiness
and valid delta coverage as well. Preserve the generated calibration audit to
separate mapping error, movement sensitivity and sample rejection in the next
diagnosis.
