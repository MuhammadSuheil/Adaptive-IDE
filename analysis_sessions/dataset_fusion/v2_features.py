"""Feature-aware quality and masking; never fit statistics or rewrite measurements."""

import copy
import math

import numpy as np

from hrv_monitor_prototype.config import (
    HRV_MAX_DATA_AGE_SECONDS,
    HRV_MIN_COVERAGE,
    HRV_MIN_PAIRS,
)

from . import build_dataset as shared

POLICY = 'feature_aware_quality_v1'
CORE = ['rmssd_ms', 'sdnn_ms', 'blink_rate_per_minute', 'ear_left_mean', 'ear_left_std',
        'ear_right_mean', 'ear_right_std', 'iris_size_estimate_mean', 'iris_size_estimate_std',
        'iris_size_delta_mean']
HEAD = ['head_pitch_mean', 'head_pitch_std', 'head_yaw_mean', 'head_yaw_std']
GAZE = ['gaze_x_norm_mean', 'gaze_x_norm_std', 'gaze_y_norm_mean', 'gaze_y_norm_std']
STATES = ['eye_state_fixation_ratio', 'eye_state_saccade_ratio', 'eye_state_transition_ratio',
          'eye_state_unclassified_ratio']
SPATIAL = [*GAZE, *STATES, 'eye_state_other_ratio']
PROFILES = {'core': CORE, 'core_head': [*CORE, *HEAD],
            'validated_fusion': [*CORE, *HEAD, *GAZE, *STATES]}
CORE_METRICS = {'min_eye_temporal_coverage': 'eye_temporal_coverage',
                'min_face_ear_ratio': 'face_ear_ratio', 'min_iris_available_ratio': 'iris_available_ratio'}


def true(value):
    return str(value).lower() in ('true', '1')


def core_quality(row, config):
    reasons = []
    if 'rr_local_median_jump' in str(row.get('hrv_reason', '')).split(';'):
        reasons.append('rr_local_median_jump')
    for column in CORE:
        value = shared.finite(row.get(column))
        valid = math.isfinite(value)
        if column.endswith('_std') or column in ('rmssd_ms', 'sdnn_ms', 'blink_rate_per_minute'):
            valid &= value >= 0
        if column in ('ear_left_mean', 'ear_right_mean'):
            valid &= 0 <= value <= 1
        if column == 'iris_size_estimate_mean':
            valid &= value > 0
        if not valid:
            reasons.append('invalid_or_missing:'+column)
    if not true(row.get('hrv_usable')) or row.get('hrv_status') != 'ready':
        reasons.append('hrv_unusable')
    for key, passes in (
        ('hrv_coverage', shared.finite(row.get('hrv_coverage')) >= HRV_MIN_COVERAGE-1e-9),
        ('hrv_rr_pairs', shared.finite(row.get('hrv_rr_pairs')) >= HRV_MIN_PAIRS),
        ('hrv_data_age_seconds', 0 <= shared.finite(row.get('hrv_data_age_seconds')) <= HRV_MAX_DATA_AGE_SECONDS+1e-9),
    ):
        if not passes:
            reasons.append(key)
    failed = {}
    for tier in ('good', 'acceptable'):
        tier_reasons = list(reasons)
        for rule, column in CORE_METRICS.items():
            value = shared.finite(row.get(column))
            if not math.isfinite(value) or not config['quality'][tier][rule]-1e-9 <= value <= 1:
                tier_reasons.append(rule)
        failed[f'{tier}_failed_criteria'] = ';'.join(tier_reasons)
    tier = 'good' if not failed['good_failed_criteria'] else 'acceptable' if not failed['acceptable_failed_criteria'] else 'poor'
    return {'quality_tier': tier, **failed}


def apply_policy(frame, config):
    """Select all eligible core windows using existing metadata columns only."""
    frame = frame.copy()
    if frame.window_id.duplicated().any():
        raise ValueError('Duplicate feature-aware window IDs.')
    evaluated = [core_quality(row, config) for row in frame.to_dict('records')]
    for column in ('quality_tier', 'good_failed_criteria', 'acceptable_failed_criteria'):
        frame[column] = [row[column] for row in evaluated]
    frame['curation_selected'] = frame.quality_tier.isin(('good', 'acceptable'))
    frame['curation_reason'] = np.where(frame.curation_selected, 'core_quality_selected', frame.acceptable_failed_criteria)
    return frame


def spatial_mask(frame, config, validation_checks=None):
    """Require source-verified validation plus window-level gaze quality."""
    allowed = set() if validation_checks is None else {
        session for session, audit in validation_checks.items() if audit.get('verified') is True}
    rules = config['quality']['acceptable']
    complete = np.isfinite(frame[SPATIAL].to_numpy(float)).all(axis=1)
    return (frame.session_id.isin(allowed) & frame.gaze_validation_passed.map(true)
            & frame.valid_gaze_ratio.ge(rules['min_valid_gaze_ratio']-1e-9)
            & frame.valid_gaze_ratio.le(1)
            & frame.clipping_ratio_valid.between(0, rules['max_clipping_ratio']+1e-9)
            & complete)


def curated_projection(frame, config, validation_checks=None):
    result = frame.copy()
    result.loc[~spatial_mask(frame, config, validation_checks), SPATIAL] = np.nan
    return result


def profile_frame(frame, profile, config, validation_checks=None):
    if profile not in PROFILES:
        raise ValueError(f'Unknown sensor profile: {profile}')
    mask = frame.quality_tier.isin(('good', 'acceptable'))
    mask &= np.isfinite(frame[PROFILES[profile]].to_numpy(float)).all(axis=1)
    if profile == 'validated_fusion':
        mask &= spatial_mask(frame, config, validation_checks)
    return frame.loc[mask].copy()


def pre_model_audit(frame):
    """Describe the admitted core without fitting transformations or changing labels."""
    selected = frame.loc[frame.quality_tier.isin(('good', 'acceptable'))]
    values = selected[CORE].to_numpy(float)
    variation: dict[str, dict[str, float | None]] = {}
    for column in ('rmssd_ms', 'ear_left_mean', 'iris_size_estimate_mean'):
        total = float(((selected[column]-selected[column].mean())**2).sum())
        variation[column] = {}
        for group, key in (('session_id', 'session'), ('cognitive_load_label', 'label')):
            between = sum(len(rows)*(rows.mean()-selected[column].mean())**2
                          for _, rows in selected.groupby(group)[column])
            variation[column][key] = round(float(between/total), 8) if total > 0 else None
    low = selected.cognitive_load_label.eq('Low')
    return {'windows': len(selected), 'missing_core_values': int(selected[CORE].isna().sum().sum()),
            'infinite_core_values': int(np.isinf(values).sum()),
            'constant_core_features': [name for name in CORE if selected[name].nunique() <= 1],
            'duplicate_window_ids': int(selected.window_id.duplicated().sum()),
            'duplicate_core_vectors': int(selected.duplicated(CORE, keep=False).sum()),
            'missing_optional_baseline_ratio': int(selected.rmssd_baseline_ratio.isna().sum()),
            'low_windows': int(low.sum()),
            'low_active_windows': int((low & selected.study_phase.eq('active_task')).sum()),
            'variation': variation}


def contract(config, validation_checks=None, *, curation_policy=None):
    checks = validation_checks or {}
    result = {
        'policy': POLICY, 'target': 'cognitive_load_label', 'group': 'participant_id',
        'features': list(CORE),
        'profiles': {name: {'features': list(columns),
                           'row_rule': 'all_core_eligible_rows' if name == 'core' else
                           'complete_finite_head_pose' if name == 'core_head' else
                           'complete_features_and_source_verified_validation_and_gaze_window_quality'}
                     for name, columns in PROFILES.items()},
        'validated_fusion_filter': {
            'verified_session_ids': sorted(session for session, audit in checks.items() if audit.get('verified') is True),
            'gaze_validation_passed': True,
            'min_valid_gaze_ratio': config['quality']['acceptable']['min_valid_gaze_ratio'],
            'max_clipping_ratio_valid': config['quality']['acceptable']['max_clipping_ratio'],
        },
        'core_quality_rules': {tier: {rule: config['quality'][tier][rule] for rule in CORE_METRICS}
                               for tier in ('good', 'acceptable')},
        'hrv_rules': {'min_coverage': HRV_MIN_COVERAGE, 'min_rr_pairs': HRV_MIN_PAIRS,
                      'max_data_age_seconds': HRV_MAX_DATA_AGE_SECONDS},
        'audit_only_features': [c for c in shared.FEATURE_COLUMNS if c not in PROFILES['validated_fusion']],
        'missing_value_policy': 'No imputation; curated spatial columns are missing unless spatial validation and window quality pass.',
    }
    if curation_policy:
        result['curation_policy'] = curation_policy
        for profile in result['profiles'].values():
            profile['row_rule'] = 'curation_selected_and_'+profile['row_rule']
    return result


def audit_validation(calibration):
    """Check the saved decision against the saved thresholds and five point means."""
    attempts = [a for a in calibration.get('attempts', []) if a.get('phase') == 'held_out_validation']
    source = {'config': {'validation': copy.deepcopy(calibration.get('config', {}).get('validation', {}))},
              'attempts': copy.deepcopy(attempts[-1:])}
    if not attempts:
        return {'verified': False, 'reason': 'validation_unavailable', 'source': source}
    last = attempts[-1]
    metrics = ('mean_target_error_median_px', 'mean_target_error_p95_px', 'max_corner_error_px')
    result = {'verified': False, 'reason': 'failed_or_unknown_validation', 'source': source,
              'passed_thresholds': last.get('passed_thresholds'), 'decision': last.get('decision'),
              'metrics': {k: last.get(k) for k in metrics},
              'validation_configuration': copy.deepcopy(calibration.get('config', {}).get('validation', {}))}
    if last.get('passed_thresholds') is not True:
        return result
    rules = result['validation_configuration']
    limits = [shared.finite(rules.get(k)) for k in ('max_median_error_px', 'max_p95_error_px', 'max_corner_error_px')]
    actual = [shared.finite(last.get(k)) for k in metrics]
    if rules.get('enabled') is not True or not all(math.isfinite(v) and v >= 0 for v in [*actual, *limits]):
        result['reason'] = 'validation_configuration_or_metrics_unavailable'
        return result
    points = last.get('points', [])
    if len(points) != 5:
        result['reason'] = 'validation_points_incomplete'
        return result
    errors = []
    try:
        for point in points:
            target = np.asarray(point['target'], dtype=float)
            samples = np.asarray([s['prediction'] for s in point['samples']], dtype=float)
            if target.shape != (2,) or samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
                raise ValueError('Incomplete validation samples.')
            if not np.isfinite(target).all() or not np.isfinite(samples).all():
                raise ValueError('Nonfinite validation samples.')
            error = float(np.linalg.norm(samples.mean(axis=0)-target))
            if not np.isclose(error, shared.finite(point.get('mean_prediction_error_px')), rtol=1e-7, atol=1e-6):
                raise ValueError('Validation point mean mismatch.')
            errors.append(error)
    except (KeyError, TypeError, ValueError):
        result['reason'] = 'validation_points_inconsistent'
        return result
    recomputed = [float(np.median(errors)), float(np.percentile(errors, 95)), max(errors[1:])]
    if not np.allclose(actual, recomputed, rtol=1e-7, atol=1e-6) or any(a > b for a, b in zip(actual, limits)):
        result['reason'] = 'validation_decision_or_metrics_inconsistent'
        return result
    result.update(verified=True, reason='saved_validation_verified')
    return result
