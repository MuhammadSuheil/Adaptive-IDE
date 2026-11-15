"""Validation and recoverable publication for complete v2 sensor bundles."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd

from . import build_dataset as shared
from hrv_monitor_prototype.config import (
    HRV_MIN_COVERAGE, HRV_MIN_PAIRS, HRV_MAX_DATA_AGE_SECONDS,
)

WINDOWS = (15, 25, 45)
LABELS = ('Low', 'Medium', 'High')
ADMISSION_POLICY = 'window_quality_only'


def quality_views(frame):
    if frame.window_id.duplicated().any():
        raise ValueError('Duplicate v2 window IDs.')
    return {name: frame[frame.quality_tier.isin(tiers)].copy()
            for name, tiers in shared.DATASETS.items()}


def _require(condition, message):
    if not bool(np.all(condition)):
        raise ValueError(message)


def _bool(series):
    text = series.astype(str).str.lower()
    _require(text.isin(('true', 'false', '1', '0')), f'Invalid boolean: {series.name}')
    return text.isin(('true', '1'))


def validate_frame(frame, config, seconds):
    """Fail rather than publish inconsistent measurements or invented values."""
    from .build_dataset_v2 import segment_label

    _require(seconds in WINDOWS, 'Unsupported window length.')
    _require(not frame.window_id.duplicated().any(), 'Duplicate window IDs.')
    _require(frame.cognitive_load_label.isin(LABELS), 'Invalid cognitive-load target.')
    _require(frame.quality_tier.isin(shared.TIERS), 'Invalid quality tier.')
    for column in ('window_id', 'session_id', 'participant_id'):
        _require(frame[column].notna() & frame[column].astype(str).str.strip().ne(''),
                 f'Missing identity: {column}')
    _require(frame.window_id.eq(frame.session_id.astype(str)+'|'+frame.window_start.astype(str)
                                +'|'+frame.window_end.astype(str)), 'Window identity does not match its source interval.')
    if frame.empty:
        return
    for column in ('window_start', 'window_end'):
        _require(frame[column].map(lambda v: pd.Timestamp(v).tzinfo is not None),
                 'Window timestamps must include a timezone.')
    start = pd.to_datetime(frame.window_start, utc=True)
    end = pd.to_datetime(frame.window_end, utc=True)
    _require(np.isclose((end-start).dt.total_seconds(), seconds, atol=1e-6),
             'Window duration mismatch.')
    _require(frame.window_seconds.eq(seconds), 'Window_seconds mismatch.')
    for _, rows in frame.assign(_start=start).groupby('session_id', sort=False):
        gaps = rows._start.diff().dropna().dt.total_seconds()
        _require((gaps > 0) & np.isclose(gaps/5, np.round(gaps/5), atol=1e-6),
                 'Window grid must be ordered with five-second increments.')
    features = frame[shared.FEATURE_COLUMNS].apply(pd.to_numeric, errors='raise')
    _require(~np.isinf(features.to_numpy(float)), 'Infinite sensor feature.')
    for name in features:
        if name.endswith('_std') or name in ('rmssd_ms', 'sdnn_ms', 'blink_count', 'blink_rate_per_minute'):
            _require(features[name].isna() | features[name].ge(0), f'Negative feature: {name}')
    for column in ('ear_left_mean', 'ear_right_mean', *[c for c in features if c.startswith('eye_state_')]):
        _require(features[column].isna() | features[column].between(0, 1), f'Invalid ratio: {column}')
    states = features[[c for c in features if c.startswith('eye_state_')]]
    _require(states.isna().all(axis=1) | states.notna().all(axis=1), 'Partial eye-state proportions.')
    _require(states.isna().all(axis=1) | np.isclose(states.sum(axis=1), 1), 'Eye-state sum is not one.')
    usable = _bool(frame.hrv_usable)
    ready = frame.hrv_status.eq('ready') & features.rmssd_ms.notna() & features.sdnn_ms.notna()
    _require(usable.eq(ready), 'HRV usability does not match status and measurements.')
    _require(features.loc[~usable, ['rmssd_ms', 'sdnn_ms']].isna(), 'Unusable HRV must remain missing.')
    _require(frame.loc[usable, 'hrv_coverage'] >= HRV_MIN_COVERAGE-1e-9, 'Insufficient HRV coverage.')
    _require(frame.loc[usable, 'hrv_rr_pairs'] >= HRV_MIN_PAIRS, 'Insufficient HRV pairs.')
    _require(frame.loc[usable, 'hrv_data_age_seconds'] <= HRV_MAX_DATA_AGE_SECONDS+1e-9, 'Stale HRV.')
    baseline = _bool(frame.baseline_usable)
    _require(~baseline | (_bool(frame.baseline_recorder_ready)
                          & frame.baseline_rmssd_ms.gt(0)
                          & frame.baseline_coverage.ge(config['baseline_min_coverage'])),
             'Baseline eligibility mismatch.')
    ratio_available = usable & baseline
    _require(features.loc[~ratio_available, 'rmssd_baseline_ratio'].isna(),
             'Invalid baseline ratio must remain missing, not zero.')
    _require(np.isclose(features.loc[ratio_available, 'rmssd_baseline_ratio'],
                        features.loc[ratio_available, 'rmssd_ms']/frame.loc[ratio_available, 'baseline_rmssd_ms']),
             'Baseline ratio mismatch.')
    _require(frame.loc[~baseline, 'baseline_rejection_reason'].notna()
             & frame.loc[~baseline, 'baseline_rejection_reason'].astype(str).str.strip().ne(''),
             'Missing baseline rejection reason.')
    bad_blinks = frame.blink_counter_discontinuities.gt(0) | frame.eye_frames_count.eq(0)
    _require(features.loc[bad_blinks, ['blink_count', 'blink_rate_per_minute']].isna(),
             'Unknown blink counts must remain missing.')
    _require(features.loc[~bad_blinks, 'blink_count'].notna(), 'Unexplained missing blink count.')
    _require(np.isclose(features.blink_rate_per_minute, features.blink_count*60/seconds, equal_nan=True),
             'Blink rate mismatch.')
    for row in frame.to_dict('records'):
        segments = json.loads(row['window_phase_mix'])
        _require(bool(segments), 'Missing phase segments.')
        durations = [float(s['duration_seconds']) for s in segments]
        _require(np.isfinite(durations) & (np.asarray(durations) >= 0), 'Invalid phase duration.')
        _require(np.isclose(sum(durations), seconds, rtol=0, atol=1e-5), 'Phase duration mismatch.')
        targets = {segment_label(s) for s in segments if s['duration_seconds'] > 0}
        _require(targets == {row['cognitive_load_label']}, 'Phase/target mismatch.')
        mix = json.loads(row['window_label_mix'])
        _require(set(mix) == {row['cognitive_load_label']}
                 and np.isclose(sum(mix.values()), seconds, rtol=0, atol=1e-5), 'Label duration mismatch.')
        expected = shared.quality_tier(row, config)
        _require(row['quality_tier'] == expected['quality_tier'], 'Quality tier mismatch.')


def _no_links(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError(f'Linked publication path is not allowed: {part}')
    return path


def verify_bundle(directory):
    """Verify hashes, row identity, admission, counts and sensor invariants."""
    directory = _no_links(directory)
    for child in directory.rglob('*'):
        if child.is_symlink() or child.is_junction():
            raise ValueError('A bundle cannot contain links.')
    try:
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        _require(manifest['schema_version'] == 2 and manifest['window_seconds'] == list(WINDOWS)
                 and manifest['step_seconds'] == 5 and manifest['target'] == 'cognitive_load_label',
                 'Not a complete v2 bundle.')
        actual = shared.snapshot(directory)
        del actual['manifest.json']
        _require(actual == manifest['output_sha256'], 'Bundle checksum mismatch.')
        policy = manifest.get('admission_policy', 'legacy_review_groups')
        _require(policy in (ADMISSION_POLICY, 'legacy_review_groups'), 'Unknown admission policy.')
        for length in WINDOWS:
            root = directory/f'window_{length}s_step_5s'
            frame = pd.read_csv(root/'all_data.csv')
            validate_frame(frame, manifest['quality_configuration'], length)
            expected = quality_views(frame) if policy == ADMISSION_POLICY else shared.dataset_views(frame)
            for name, view in expected.items():
                published = pd.read_csv(root/f'{name}.csv')
                pd.testing.assert_frame_equal(published, view.reset_index(drop=True),
                                              check_dtype=False, rtol=1e-10, atol=1e-10)
                counts = manifest['datasets'][root.name][name]
                _require(counts['windows'] == len(published)
                         and counts['participants'] == published.participant_id.nunique(), 'Bundle count mismatch.')
                for label in LABELS:
                    _require(counts[label] == published.cognitive_load_label.eq(label).sum(), 'Class count mismatch.')
            allowlist = json.loads((root/'feature_columns.json').read_text())
            _require(allowlist['target'] == 'cognitive_load_label'
                     and allowlist['features'] == shared.FEATURE_COLUMNS, 'Feature contract mismatch.')
        return manifest
    except (KeyError, OSError, AssertionError, TypeError) as error:
        raise ValueError(f'Incomplete or inconsistent v2 bundle: {directory}') from error


def guard_output(input_dir, output_dir, *, replace=False):
    output = _no_links(output_dir).resolve()
    source = Path(input_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    if (output.is_relative_to(source) or source.is_relative_to(output)
            or root.is_relative_to(output) or Path(__file__).resolve().is_relative_to(output)
            or output == Path(__file__).with_name('output')
            or 'v1_active_task' in output.parts):
        raise ValueError(f'Protected or overlapping output target: {output}')
    if output.exists():
        if not replace:
            raise FileExistsError(f'Output exists; use --replace for a verified v2 bundle: {output}')
        verify_bundle(output)
    return output


def publish_bundle(staged, output, *, replace=False):
    """Keep rollback outside temporary staging so recovery failure cannot erase it."""
    staged, output = _no_links(staged).resolve(), _no_links(output).resolve()
    manifest = verify_bundle(staged)
    guard_output(manifest['input_directory'], output, replace=replace)
    backup = output.with_name(f'.{output.name}.rollback-{uuid4().hex}') if output.exists() else None
    if backup is not None:
        os.rename(output, backup)
    try:
        os.rename(staged, output)
        if os.name == 'nt':
            subprocess.run(['icacls', str(output), '/reset', '/T', '/q'], check=True, capture_output=True)
        verify_bundle(output)
    except Exception:
        try:
            if output.exists():
                os.rename(output, staged)
            if backup is not None:
                os.rename(backup, output)
        except OSError as error:
            raise RuntimeError(f'Publication recovery failed; previous bundle preserved at {backup}') from error
        raise
    if backup is not None:
        # Exact sibling created by this invocation, never a caller-supplied delete target.
        _require(backup.resolve().parent == output.parent and not backup.is_symlink()
                 and not backup.is_junction() and backup.name.startswith(f'.{output.name}.rollback-'),
                 'Unsafe rollback cleanup path.')
        shutil.rmtree(backup)
