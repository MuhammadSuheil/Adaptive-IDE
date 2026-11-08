# Five-level HRV + eye load prototype

The output is **Low / Mid low / Mid / Mid high / High**. Task difficulty (`easy/medium/hard`) is not a target or predictor. This directory is a new implementation; the previous task-difficulty models and scores were deleted.

Start with [train_and_compare.ipynb](train_and_compare.ipynb). It contains configuration, training/review controls, explanations, graphs, held-out predictions and a saved-model example. Select a Python kernel and run cells from top to bottom. The default reviews the newest completed run matching both your inputs and scoring configuration. Set `TRAIN_NEW_RUN = True` to train again.

```powershell
python -m pip install -r analysis_sessions/model_training/requirements-notebook.txt
```

## What the five levels mean

These are **provisional relative sensor-based levels**, not independently measured cognitive load. No independent workload ratings were supplied. We define a transparent sensor reference, then train models to approximate it. Agreement with that reference is not validated mental-state accuracy. The reference calculator itself can produce its own levels without ML; the trained models are prototype approximations, not new ground truth.

The default configuration in `config.json` makes these explicit assumptions:

1. Fit empirical percentile distributions using **good training windows only**, with each participant receiving equal total weight. Lower baseline-normalized RMSSD maps to a higher 0–100 HRV reference score.
2. Lower blink rate and higher estimated fixation proportion each map to a higher eye reference component. The eye reference is their equal-weight average. These directions are experimental assumptions, not universal physiological laws.
3. Combine the branches: **score = 0.60 × HRV score + 0.40 × eye score**. Thus the reference uses 60% HRV, 20% blink and 20% fixation components.
4. Fit boundaries at the participant-weighted 20th, 40th, 60th and 80th percentiles of training-reference scores. Scores below the first boundary are Low; scores exactly on a boundary enter the next band. Ties receive identical percentile values. Degenerate references that cannot separate five bands fail explicitly.
5. Freeze the distributions and boundaries with the exported models. Future windows are compared against that reference, not ranked against their own incoming batch.

Measured zeros are valid, including zero blink rate or zero baseline-normalized RMSSD. Missing, nonfinite, negative HRV ratios/blink rates or out-of-range fixation ratios cannot define a target. Such rows are retained in coverage/label reports as `insufficient_signal`, and excluded from supervised fitting. We do not fill missing target measurements with medians. No quiet eye baseline is invented from task measurements.

The alternate **50/50** configuration reuses the same fitted branches with its own saved reference boundaries. The report shows label sensitivity; it does not claim one weighting is scientifically correct.

## How all the useful eye columns enter

The models use separate branches, so the requested modality weights are enforced at prediction time rather than implied by column count or feature scaling.

| Branch/family | Candidate inputs | Role |
|---|---|---|
| HRV | RMSSD, SDNN, baseline-normalized RMSSD | Predict the HRV reference score |
| Blink | Count, rate per minute | Predict the eye reference score |
| EAR | Left/right mean and standard deviation | Eyelid-pattern candidate inputs |
| Iris geometry | Size estimate and delta, means and standard deviations | Experimental geometry/change inputs; **not pupil diameter** |
| Gaze | Normalized X/Y means and standard deviations | Spatial context and dispersion; position has no automatic load direction |
| Eye state | Estimated fixation, saccade and transition proportions | Candidate viewing-pattern inputs; not independently validated event durations |
| Head context | Pitch/yaw means and standard deviations | Optional geometry/context covariates, enabled by default |
| Quality only | Unclassified/other eye-state ratios; export quality diagnostics | Excluded from load predictors |

There are **3 HRV and 21 eye/context predictors** by default. EAR, raw iris geometry and gaze coordinates do not receive arbitrary high-load directions in the label formula. They can help the eye branch approximate its reference or compensate for correlated context; they may also have little incremental value. The report includes feature-family perturbation diagnostics. These describe dependence of this fitted model on its inputs, not physiological validity, causal importance or independent evidence that an extra feature measures load.

## Models and evaluation

Three model families estimate continuous branch scores: **RBF-SVM regression (SVR), Random Forest regression, and histogram gradient boosting regression**. Regression preserves a continuous signal before assigning the five ordered output levels. Every exported bundle has independent HRV and eye pipelines. Their predictions are bounded to 0–100, fused at 60/40, and converted to levels using the frozen boundaries. We do not export class probabilities or claim a calibrated confidence.

Outer evaluation holds out a complete participant, including all their sessions. All algorithms and dataset variants use the same eligible good-window tests and the **same reference within that outer fold**. The current export provides five people with good test windows out of 18 overall. Other participants can contribute training rows in broader variants, but do not become extra shared-test participants.

Inner tuning uses up to three grouped folds, spreading good-reference people across validation folds so each inner training reference retains at least two good participants. References are refitted inside each inner training fold, along with preprocessing. Validation people never influence their reference. If no valid two-/three-fold scheme exists, that outer comparison is skipped with its reason, never replaced by random rows. All five target levels must be represented in training.

Each branch chooses hyperparameters by participant-average absolute score error. Missing non-anchor features use medians fit within the current training fold; entirely missing training columns are dropped rather than filled with zero. Gradient boosting's internal random-row early stopping is disabled. The compact search tries 3 SVM C values, 4 forest depth/leaf settings and 2 boosting learning rates per branch; `quick` uses one setting per family.

The main comparison is **participant-average five-class macro-F1 agreement**, with participant-average score MAE as the tie-breaker. Fixed-five-class F1 assigns zero F1 to a class absent from that person's test; class support and full confusion matrices remain visible. Pooled exact agreement, within-one-level agreement, ordinal MAE and per-level recall add context. These are exploratory development comparisons, not independent accuracy measurements after selecting the best candidate.

Family diagnostics shift all columns of one family together within the held-out person, using three circular shifts. The change in reference-score MAE describes model dependence. Perturbation creates artificial combinations and is not a causal test; correlated features can substitute for one another. No causal feature-selection or scientific feature-validity claim is made.

## Train with Python

Run from the repository root, using Python 3.10+:

```powershell
python -m pip install -r analysis_sessions/model_training/requirements.txt
python -m analysis_sessions.model_training.train_models
```

The default input is `analysis_sessions/dataset_fusion/output`. Results go into a new timestamped directory under `analysis_sessions/model_training/runs/`, which is ignored by Git. Input exports are hashed before reading, after loading, and after fitting; they are not rebuilt or overwritten. Existing fusion manifests and shared test-window identities are checked. The builder's old task-difficulty feasibility flags are not used to decide five-level feasibility.

```powershell
python -m analysis_sessions.model_training.train_models --output analysis_sessions/model_training/runs/my_run
python -m analysis_sessions.model_training.train_models --search quick --output analysis_sessions/model_training/runs/smoke
```

Output must be a new directory separate from source data. Only `status: complete` in `run_manifest.json` identifies a finished run; failed/interrupted runs are not complete. `--no-export` evaluates without final-model refits. `--no-family-diagnostics` skips perturbation diagnostics for a faster run.

## Add other window sizes

```powershell
python -m analysis_sessions.model_training.train_models --dataset window25_step5=analysis_sessions/dataset_fusion/output --dataset window40_step5=path/to/friends/export --output analysis_sessions/model_training/runs/window_comparison
```

Each input directory supplies the same three nested CSVs, `feature_columns.json`, and the configured sensor columns. Required metadata are `window_id`, `participant_id`, `session_id`, `window_start`, `window_end` and `window_seconds`. One directory represents one real window length. Duplicate windows, inconsistent durations, or conflicting shared rows fail validation. Keep repeat recordings under the same participant ID. The optional fusion dictionary, manifest and evaluation contract are verified when present.

This runner never invents windows or recomputes HRV. Different sizes receive separate references and rankings. Their labels are relative to those references; comparing window sizes scientifically requires matched people and task/time coverage. Changing the input specification's existing difficulty target does not change our target: the new trainer explicitly generates `cognitive_load_level` from its own reference.

## Outputs, graphs and inference

The notebook and `<run>/<dataset-id>/visualizations/report.html` explain nine graphs: data coverage, score bands and weights, all predictor missingness, model agreement, five-level confusion matrices, per-person agreement, feature dependence, branch contributions over time and 50/50 sensitivity. The HTML embeds its images for sharing as a standalone file. Individual PNG files are available alongside it.

Important files are:

- `reference.json` and `reference_labels.csv`: final frozen reference and auditable derived labels, including unscorable rows. The same final reference is shared by all variants.
- `held_out_predictions.csv`: fold-specific reference labels, estimated branch scores, weighted contributions, output levels and sensitivity outputs. These are the evaluation predictions.
- `comparison.csv`, `summary.json`, `participant_metrics.csv/json`: aggregate and per-person reference agreement.
- `evaluation_folds.csv`, `data_coverage.csv`, `training_audit.json`: skipped folds, exclusion counts, grouped participant/reference provenance and tuning results.
- `feature_inventory.csv`, `family_sensitivity.csv`: predictor/diagnostic roles and family-dependence estimates.
- `models/<variant>__<algorithm>.joblib`: nine final bundles, each containing two independently trained branches. Companion JSON files expose the full feature/window contract, reference and training metadata.
- Parent `run_manifest.json`: status, versions, configuration and hashes of sources/artifacts.

Final bundles are retuned and refit on all eligible rows of their own training variant after evaluation. Their full-data reference differs from each held-out fold's training-only reference. Evaluation results never come from evaluating those final bundles on their own training data.

```powershell
python -m analysis_sessions.model_training.predict --model analysis_sessions/model_training/runs/five_level_60_40/window25_step5/models/good_acceptable__random_forest.joblib --input analysis_sessions/dataset_fusion/output/all_data.csv --output analysis_sessions/model_training/runs/five_level_60_40/inference_example.csv
```

Use only trusted model bundles and the recorded library versions. Python integration uses `predict_windows(bundle, frame)` from `predict.py` after `joblib.load(...)`. The output includes `cognitive_load_level`, `hrv_score`, `eye_score`, `combined_score`, explicit modality weights and signal status. Predictions at 60/40 satisfy the weighted formula exactly. An optional `--hrv-weight 0.5` uses the saved sensitivity bands.

Missing required feature columns or incompatible supplied window length fail. Missing/invalid anchors yield `insufficient_signal`; a supplied `quality_tier=poor` yields `poor_signal_quality`. Both abstain rather than invent a level. Without a recognized quality tier, predictions have `quality_unverified`; the application must enforce its quality policy. Other feature blanks use fitted medians. Feature-only callers must ensure correct window extraction because there are no timestamps to verify it against.

To refresh graphs from completed results without retraining:

```powershell
python -m analysis_sessions.model_training.visualize --run analysis_sessions/model_training/runs/five_level_60_40
```

The renderer verifies matching input/result hashes and records updated visualization hashes. Model files do not change. No model is installed in the adaptive application automatically. The application still needs an independently tested adaptation policy; high provisional scores are not validated overload thresholds.

## Verification

```powershell
python -m unittest discover -s analysis_sessions/model_training -p "test_*.py" -v
```

Tests exercise five-level semantics, ignored task difficulty, scoring directions, participant weighting, ties, missing/zero signals, grouped references, exact 60/40 and 50/50 fusion, poor-signal abstention, nested exports, alternative windows, all nine save/reload paths and graphs. Fused input hashes remain unchanged.
