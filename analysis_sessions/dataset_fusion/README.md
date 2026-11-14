# HRV + eye-tracking dataset builder

Build the three nested training-data candidates from the archived student recordings:

```powershell
python analysis_sessions/dataset_fusion/build_dataset.py
```

The default input is `sessions/saved_session`. The default output is `analysis_sessions/dataset_fusion/output`. Only derived outputs are written; all source files, including duplicate folder variants, are hashed before/after computation. A build fails before publishing results if the hashes change. Requires Python 3.10+, NumPy and pandas; see `requirements.txt`.

```powershell
python analysis_sessions/dataset_fusion/build_dataset.py --config analysis_sessions/dataset_fusion/config.json --output analysis_sessions/dataset_fusion/output
python analysis_sessions/dataset_fusion/build_dataset.py --subjects Andrean Fabian faqih ikhsan kaka --output analysis_sessions/dataset_fusion/output_team_good_review
python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py" -v
```

`--subjects` explicitly limits the source cohort and uses exact folder names. It does not change window-quality rules. The default build includes every unique recording. Output must be separate from the input tree. Rebuilding replaces generated files of the same names, so keep manual work outside the output directory. Each file is replaced atomically and `manifest.json` is published last; interrupted multi-file publication is detectable through its output hashes.

On Windows, published files inherit their destination folder's normal permissions. This prevents private temporary-build permissions from blocking your desktop Excel/CSV editor. If an older export gives an access error, rebuilding repairs the generated files; source session permissions are untouched.

## What the CSVs mean

| File | Rows |
|---|---|
| `good_only.csv` | Good windows from Andrean, Fabian, Faqih, Ikhsan, and Kaka only |
| `good_acceptable.csv` | Good + acceptable windows from those five and the eight listed questionable participants only |
| `all_data.csv` | All pure active-task windows, including poor/missing-sensor windows |

The target column is **`difficulty_level`: easy / medium / hard**. Training maps these values to Low / Medium / High as the study's cognitive-load proxy. Keep the original task-condition values in the source CSV and document this mapping; it does not create independently measured cognitive-state ground truth. Do not use the old sensor-rule cognitive labels as ground truth. Task order, experience and camera conditions may confound prediction.

## Research context and column roles

The intended application is one cognitive-load classifier using eye-tracking and HRV features per window during programming, not participant identification or one classifier per person. CodeArrange supplies the acquisition tasks and event/label provenance. Participant, session, question and quality columns can remain in the same CSV without becoming predictors. The authoritative context is recorded in [models/README.md](../../models/README.md#konteks-penelitian-dan-peran-kolom).

The protocol deliberately allocates **5 minutes to easy, 12 to medium and 15 to hard**. These are set time limits, not mandatory actual working durations. The operator also confirms that some participants intentionally skipped easy because of collection-time constraints. Unequal duration and absent classes are therefore not automatically recorder or cleaning failures. A zero class count still needs session-level interpretation: an omitted task, a short active interval or quality filtering can each produce no retained windows. Do not invent rows, relabel rest as Low or drop participants merely because their protocol omitted easy.

Keep four roles separate: sensor features as `X`; `difficulty_level` mapped to CL as `y`; IDs for grouping/deduplication; context/quality metadata for alignment, filtering and audit. Use `feature_columns.json` as an explicit predictor allowlist rather than selecting all numeric columns. Grouping by participant protects evaluation of the sensor model; participant ID is not a prediction feature. CodeArrange set/question IDs, difficulty and configured time limits must not reveal the answer through model inputs.

No source CSV/schema change is needed for this clarification. Preserve all three datasets, their metadata, labels and shared-row provenance. Any future temporal predictor must distinguish offline question-based history resets from generic time/stream boundaries available outside CodeArrange. Baseline-based sensor normalization, when valid and available before prediction, does not imply a separate personal classifier.

One row combines recorded HRV with eye measurements over exactly the same **25-second interval**, anchored to the source HRV windows with a **5-second step** (80% overlap). Eye endpoints are `(start, end]`; timestamps are absolute and timezone-aware. Skipped HRV grid points are allowed, but duplicates, reversed times, off-grid steps and incompatible durations are rejected. No new HRV rows are invented and no sensor values are interpolated. Changing the configuration does not recompute windows from RR data.

Labels are recalculated from the complete task-event ledger using the original sensor session ID, including for renamed participant folders and Rafi's missing timeline. Only intervals entirely in active coding at one difficulty qualify. Same-difficulty question transitions may be retained with mixed question provenance. Practice, feedback, rating, baseline, interruptions, mixed difficulty, and other phases are logged in `excluded_windows.csv`. Exact end-of-window ledger changes have zero duration inside that interval; an eye sample exactly at the end is included by the stated sensor endpoint convention.

All three CSVs have identical columns and identical shared rows. They contain HRV, EAR, blink, iris estimate, head pose, gaze and Eye_State features together. `feature_columns.json` lists **only sensor predictors**; `data_dictionary.csv` documents every column and distinguishes features, metadata and target. Sample standard deviations use `ddof=1`; a singleton's standard deviation stays blank. Eye_State proportions are frame-based among valid-gaze rows, not event counts or measured fixation durations. No LF/HF, pupillary wavelet indices or microsaccade estimates are fabricated from these short windows/webcam measurements.

## Configuration and your participant lists

Edit `participant_review_groups` in `config.json` to set the participant eye-quality admission lists, and `quality` to set the additional window checks. Both admission and window checks must pass. The actual configuration, dataset admission policy and source/code hashes are stored in every output manifest.

- **Proposed good:** Andrean, Fabian, Faqih, Ikhsan, Kaka.
- **Questionable:** Andreas, Athaya, Rafi, Raina, Sam, Savitri, Shakilla, Tio.

Review names match folders case-insensitively, so Faqih/faqih, Ikhsan/ikhsan and Kaka/kaka refer to the same people. These are the team's **eye-quality cohorts**, not a ranking of HRV or task performance. `proposed_good` permits admission to `good_only` and `good_acceptable`; `questionable` permits admission only to `good_acceptable`. Both datasets also apply their window checks. Everyone else is `unreviewed` and appears **only in `all_data`**, even when their numeric window checks pass. This excludes Baithar, Irfan, Ramadhan/Ramadhan-2, Siti and Sullivan from both curated datasets. A qualifying window cannot promote someone outside your lists.

Shakilla's reported laptop camera/30-FPS condition and Ikhsan's glasses are retained as context. Actual camera width/height, reported FPS and observed capture FPS are stored separately. Ramadhan and Ramadhan-2 share a participant ID, retain their own baselines, and the second medium recording is flagged as repeated. Baithar's duplicate acquisition is selected once through the existing inventory helper; sources are never deleted.

| Criterion | Good | Acceptable |
|---|---:|---:|
| HRV status | Ready, finite nonnegative RMSSD and SDNN | Same |
| Eye temporal coverage | ≥95% | ≥80% |
| Face + both EAR values within [0, 1] | ≥95% of observed rows | ≥80% |
| Valid finite gaze | ≥90% of observed rows | ≥70% |
| Valid nonblinking iris estimate | ≥90% of observed rows | ≥70% |
| Clipping among valid gaze | ≤5% | ≤20% |

Good rules are applied first; acceptable rules next; all other retained windows are poor. Every failed rule is recorded. Eye temporal coverage counts occupied 100-ms bins anchored at the window start; it never assumes nominal 30/60 FPS. Face/gaze/iris fractions use observed rows. Clipping uses the saved raw mapped coordinates, at the recorder's 10%-outside-screen clamps. The clipping margin must reflect the acquisition code; it is not an accuracy tolerance. The original gaze-validation pass/fail, median, P95 and maximum corner error are reported separately. **Good means operational completeness and limited clipping, not proven spatial accuracy.** These thresholds are exploratory, not scientifically validated cutoffs or copied audit benchmarks.

Unavailable or invalid values stay blank; zero is retained only when measured/derived as zero. Unready HRV stays blank even if an older source contains numeric values. Baseline normalization requires a positive ready baseline and coverage ≥80%; a failed baseline does not discard usable raw task HRV. EAR and head pose are masked without a face. Iris features require valid nonblinking frames in `eye_width_ratio` mode, and the delta additionally requires a ready iris baseline. Iris estimates are **not pupil diameter**. Blink features use cumulative-counter increments at their observed timestamps, not averages of cumulative counts or rolling rates. Counter resets/missing counts leave the window blink feature unavailable. Partial eye coverage can undercount blinks; coverage is always exported. Poor calibration/clipped gaze values remain represented and flagged for the intended poor-data comparison.

## Reports and future training

Start with `output/QUALITY_REPORT.md` and `session_quality_summary.csv`. They list every selected session, the team's eye-quality cohort, actual counts admitted to each dataset, and original gaze validation. The CSV summary also keeps numeric good/acceptable/poor window-tier counts as diagnostics; those counts alone do not grant dataset admission. `counts_by_session_difficulty_tier.csv` includes zero-count cells. The inventory explains duplicate selection. `manifest.json` contains the full configuration, admission policy, dataset counts, dependency versions and source/code/output SHA-256 hashes.

`evaluation_folds.csv` defines leave-one-participant-out comparisons, including folds with no good test windows or missing training classes as explicitly infeasible. `evaluation_test_windows.csv` fixes the same good test windows for every training variant. For a feasible fold, train the chosen CSV on `participant_id != held_out_participant_id` and evaluate only the fold's listed windows. Never move poorer windows from the held-out participant into training. Both Ramadhan recordings stay together. Some test participants only contribute one or two task classes; per-class counts are reported explicitly.

Fit imputation, scaling, feature selection and hyperparameter tuning on training participants only, including group separation within tuning. All-data training must explicitly handle missing features, including an entirely missing feature in a fold; do not silently fill blanks with zero. Use the feature allowlist to exclude IDs, target, puzzle identity, camera and quality diagnostics from predictors. Report macro-F1, balanced accuracy, per-class recall, confusion matrices and participant-level results. Quality and training-set size vary together, so raw score differences do not isolate quality alone. An experience/year subgroup analysis needs separately linked participant information; it cannot be inferred from these measurements. Model training is not performed by this builder.
