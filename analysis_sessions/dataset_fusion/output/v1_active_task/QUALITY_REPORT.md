# Fused dataset quality report

Target: easy/medium/hard task condition, a proxy for cognitive demand. No independent cognitive-state labels were collected.

Participant eye-quality admission uses the team's configured name lists: good_only admits proposed_good; good_acceptable admits proposed_good and questionable; everyone else appears only in all_data. Passing window checks never promotes an unlisted participant into a curated dataset.

Window tiers describe completeness and clipping, not spatial accuracy. Original gaze validation is reported separately and is not used to assign tiers.

All features are exported together. Iris estimates are not measured pupil diameter; Eye_State is an algorithmic estimate. Missing values remain blank.

| Dataset | Windows | Participants |
|---|---:|---:|
| good_only.csv | 639 | 5 |
| good_acceptable.csv | 2294 | 13 |
| all_data.csv | 3546 | 18 |

## Session review

Counts below are actual dataset membership after both the team's eye-quality admission gate and the window checks. The CSV summary also retains raw window-tier counts for diagnostics.

| Session | Eye review group | good_only rows | good_acceptable rows | all_data rows | Original gaze validation |
|---|---|---:|---:|---:|---|
| Andrean | proposed_good | 227 | 247 | 248 | fail |
| Andreas | questionable | 0 | 130 | 180 | fail |
| Athaya | questionable | 0 | 166 | 174 | fail |
| Baithar | unreviewed | 0 | 0 | 185 | pass |
| Fabian | proposed_good | 208 | 243 | 248 | pass |
| faqih | proposed_good | 136 | 226 | 238 | fail |
| ikhsan | proposed_good | 11 | 212 | 283 | fail |
| irfan | unreviewed | 0 | 0 | 218 | fail |
| kaka | proposed_good | 57 | 106 | 196 | fail |
| Rafi | questionable | 0 | 272 | 283 | fail |
| raina | questionable | 0 | 71 | 134 | fail |
| Ramadhan | unreviewed | 0 | 0 | 6 | fail |
| Ramadhan-2 | unreviewed | 0 | 0 | 136 | fail |
| sam | questionable | 0 | 145 | 181 | fail |
| Savitri | questionable | 0 | 195 | 206 | fail |
| Shakilla | questionable | 0 | 88 | 88 | fail |
| Siti | unreviewed | 0 | 0 | 232 | fail |
| Sullivan | unreviewed | 0 | 0 | 105 | fail |
| Tio | questionable | 0 | 193 | 205 | fail |

## Comparison contract

Use evaluation_folds.csv and the identical evaluation_test_windows.csv for every training variant. For each feasible fold, test only its listed good windows; train the chosen dataset on participant_id != held_out_participant_id. Both Ramadhan recordings must be excluded together. Zero-test and missing-training-class folds are explicitly marked infeasible.

Fit imputers, scalers, feature selection and model tuning on training participants only. Handle entirely missing features explicitly; do not treat blanks as zeros. Do not feed metadata, task IDs or heuristic cognitive labels into the model. Report macro-F1, balanced accuracy, per-class recall, confusion matrices, and participant-level results in subsequent training work.

Training variants differ in both quality and quantity; scores alone cannot separate those effects. Difficulty may also be confounded with task order, experience, and device conditions. Thousands of overlapping windows are not thousands of independent participants.

## Rules used

```json
{
  "schema_version": 1,
  "window_seconds": 25,
  "step_seconds": 5,
  "eye_coverage_bin_ms": 100,
  "gaze_clip_margin_fraction": 0.1,
  "baseline_min_coverage": 0.8,
  "quality": {
    "good": {
      "min_eye_temporal_coverage": 0.95,
      "min_face_ear_ratio": 0.95,
      "min_valid_gaze_ratio": 0.9,
      "min_iris_available_ratio": 0.9,
      "max_clipping_ratio": 0.05
    },
    "acceptable": {
      "min_eye_temporal_coverage": 0.8,
      "min_face_ear_ratio": 0.8,
      "min_valid_gaze_ratio": 0.7,
      "min_iris_available_ratio": 0.7,
      "max_clipping_ratio": 0.2
    }
  },
  "participant_review_groups": {
    "proposed_good": [
      "Andrean",
      "Fabian",
      "Faqih",
      "Ikhsan",
      "Kaka"
    ],
    "questionable": [
      "Andreas",
      "Athaya",
      "Rafi",
      "Raina",
      "Sam",
      "Savitri",
      "Shakilla",
      "Tio"
    ]
  },
  "participant_notes": {
    "Ikhsan": "Team reports glasses; do not infer that glasses caused measurement errors.",
    "Shakilla": "Team reports a laptop camera at 30 FPS; medium only. Use observed camera metadata, not the requested FPS.",
    "Ramadhan": "Two recordings of one participant; interrupted first recording and repeated medium in the second. Keep both in the same evaluation group."
  },
  "repeated_difficulties_by_session": {
    "Ramadhan-2": [
      "medium"
    ]
  }
}
```
