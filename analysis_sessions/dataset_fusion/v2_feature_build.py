"""Rebuild the aligned v2 bundle with explicit feature-aware admission."""

import json
import platform
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_sessions.session_inventory import select_session_directories

from . import build_dataset as shared
from . import v2_features as features
from . import v2_integrity as integrity
from . import v2_rr_quality as rr_quality
from .v2_documentation import build_update_summary
from .v2_feature_documentation import write_reports


def build_feature_dataset(input_dir, output_dir, config_path, subjects, previous, *, replace=False, reuse_verified=False):
    from . import build_dataset_v2 as builder

    config = shared.load_config(config_path)
    sources_before = shared.snapshot(input_dir)
    cached = {}
    cached_audits = {}
    if reuse_verified:
        if not previous or previous['admission_policy'] != features.POLICY or subjects:
            raise ValueError('RR refresh requires a complete verified feature-aware bundle.')
        if integrity.verify_bundle(output_dir) != previous:
            raise ValueError('Active manifest changed before refresh.')
        if sources_before != previous['source_sha256'] or shared.digest(config_path) != previous['quality_configuration_sha256']:
            raise ValueError('Raw sources or quality settings changed; run the full builder instead.')
        cache_code = ('build_dataset.py', 'build_dataset_v2.py', 'v2_timestamps.py', 'v2_sensor_checks.py',
                      'session_inventory.py', 'calculate_hrv.py', 'preprocess_rr.py', 'config.py', 'fuse_and_analyze.py')
        for name, digest in previous['code_sha256'].items():
            if Path(name).name in cache_code and shared.digest(builder.ROOT/name) != digest:
                raise ValueError(f'Sensor calculation code changed: {name}; run the full builder instead.')
        for seconds in builder.WINDOWS:
            directory = output_dir/f'window_{seconds}s_step_5s'
            cached[seconds] = (pd.read_csv(directory/'all_data.csv', float_precision='round_trip'),
                               pd.read_csv(directory/'excluded_windows.csv', keep_default_na=False, float_precision='round_trip'),
                               pd.read_csv(directory/'session_quality_summary.csv', keep_default_na=False, float_precision='round_trip'))
        cached_audits = {audit['session_folder']: audit for audit in
                         json.loads((output_dir/'replay_validation.json').read_text(encoding='utf-8'))['sessions']}
    directories, inventory = select_session_directories(
        input_dir, subjects, session_exclusions=config.get('session_exclusions', {}))
    if not directories:
        raise ValueError('No selected sensor sessions.')
    previous_policy = previous.get('admission_policy') if previous else None
    timestamps = previous.get('timestamp_policies', {}) if previous else {}
    if previous_policy in builder.history.POLICIES:
        historical = json.loads((output_dir/'replacement_selection.json').read_text(encoding='utf-8'))
        timestamps = historical.get('timestamp_policies', {})
    results: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {
        seconds: ([], [], []) for seconds in builder.WINDOWS}
    audits, validation_checks = [], {}
    rr_registry = {**rr_quality.settings(), 'sessions': {}}
    for directory in directories:
        if not (directory/'task_events.csv').is_file():
            raise ValueError(f'{directory.name}: missing task_events.csv; source is not explicitly excluded.')
        session_id = str(pd.read_csv(directory/'eye_tracking.csv', nrows=1, usecols=['session_id']).session_id.iloc[0])
        validation_checks[session_id] = features.audit_validation(shared.read_json(directory/'eye_summary_calibration.json'))
        rr_audit = rr_quality.audit_session(directory)
        if rr_audit['session_id'] != session_id:
            raise ValueError('Eye and RR source identity mismatch.')
        rr_registry['sessions'][session_id] = rr_audit
        if reuse_verified:
            session = {}
            for seconds, (rows, excluded, reports) in cached.items():
                report = reports.loc[reports.session_id.eq(session_id)]
                if len(report) != 1:
                    raise ValueError('Missing or duplicated cached session report.')
                session[seconds] = (rows.loc[rows.session_id.eq(session_id)].to_dict('records'),
                                    excluded.loc[excluded.session_id.eq(session_id)].to_dict('records'),
                                    report.iloc[0].to_dict())
            audit = cached_audits[directory.name]
        else:
            session, audit = builder.build_session(directory, config, timestamps.get(session_id))
        audits.append(audit)
        eligible_counts = {}
        for seconds, (rows, excluded, report) in session.items():
            frame = rr_quality.apply_window_policy(pd.DataFrame(rows, columns=builder.DATA_COLUMNS), rr_audit)
            frame = features.apply_policy(frame, config)
            eligible_counts[seconds] = int(frame.curation_selected.sum())
            results[seconds][0].extend(frame.to_dict('records'))
            results[seconds][1].extend(excluded)
            report = {**report, **{f'{tier}_windows': int(frame.quality_tier.eq(tier).sum()) for tier in shared.TIERS}}
            results[seconds][2].append(report)
        print(f'{directory.name}: core eligible ' + ', '.join(
            f'{s}s={eligible_counts[s]}' for s in builder.WINDOWS), flush=True)
    if not set(timestamps).issubset(validation_checks):
        raise ValueError('Pinned timestamp source missing from the selected source cohort.')
    # On unchanged intervals the migration must not silently alter sensor values or targets.
    if previous:
        for seconds, (rows, _, _) in results.items():
            old = pd.read_csv(output_dir/f'window_{seconds}s_step_5s/all_data.csv').set_index('window_id')
            new = pd.DataFrame(rows).set_index('window_id')
            common = old.index.intersection(new.index)
            cols = [*shared.FEATURE_COLUMNS, 'session_id', 'participant_id', 'window_start', 'window_end',
                    'cognitive_load_label', 'gaze_validation_passed']
            pd.testing.assert_frame_equal(old.loc[common, cols], new.loc[common, cols],
                                          check_dtype=False, rtol=1e-10, atol=1e-10)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-v2-', dir=output_dir.parent) as temporary:
        destination = Path(temporary)/'bundle'
        destination.mkdir()
        history_dir = destination/'history'
        if previous:
            if (output_dir/'history').is_dir():
                shutil.copytree(output_dir/'history', history_dir)
            else:
                history_dir.mkdir()
            if previous_policy != features.POLICY:
                shutil.copy2(output_dir/'manifest.json', history_dir/'pre_feature_aware_manifest.json')
                for name in ('replacement_selection.json', 'replacement_pairs.csv'):
                    if (output_dir/name).is_file():
                        shutil.copy2(output_dir/name, history_dir/name)
        counts = {f'window_{s}s_step_5s': builder.export_window(
            destination/f'window_{s}s_step_5s', *results[s], inventory, s, config=config,
            policy=features.POLICY, validation_checks=validation_checks) for s in builder.WINDOWS}
        update = build_update_summary(results, counts, inventory, previous, policy=features.POLICY)
        update['feature_policy'] = features.POLICY
        profile_counts = {
            f'window_{s}s_step_5s': {name: len(features.profile_frame(pd.DataFrame(rows), name, config, validation_checks))
                                    for name in features.PROFILES}
            for s, (rows, _, _) in results.items()}
        rr_registry['windows'] = {}
        for seconds, (rows, _, _) in results.items():
            rr_registry['windows'][f'window_{seconds}s_step_5s'] = rr_quality.window_summary(pd.DataFrame(rows), config)
        pre_model_audit = {f'window_{seconds}s_step_5s': features.pre_model_audit(pd.read_csv(
            destination/f'window_{seconds}s_step_5s/all_data.csv')) for seconds in results}
        replay = {'sessions': audits, 'unexplained_windows': sum(a['unexplained_windows'] for a in audits)}
        (destination/'replay_validation.json').write_text(json.dumps(replay, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        operator_guide = ''
        old_report = output_dir/'DATASET_CONDITION_REPORT.md'
        if old_report.is_file():
            text = old_report.read_text(encoding='utf-8')
            start = text.find('## 7. Panduan operator sebelum dan saat pengambilan')
            if start >= 0:
                stop = text.find('\n## ', start+4)
                operator_guide = text[start:stop if stop >= 0 else len(text)]
        write_reports(destination, results, counts, profile_counts, validation_checks, update, config,
                      previous_policy, operator_guide, rr_registry=rr_registry, pre_model_audit=pre_model_audit)
        if shared.snapshot(input_dir) != sources_before:
            raise RuntimeError('Raw sources changed during build; no bundle published.')
        code_paths = [Path(builder.__file__), Path(shared.__file__), Path(__file__),
                      Path(rr_quality.__file__),
                      Path(features.__file__), Path(integrity.__file__), Path(__file__).with_name('v2_feature_documentation.py'),
                      Path(__file__).with_name('v2_sensor_checks.py'), Path(__file__).with_name('v2_documentation.py'),
                      Path(builder.timing.__file__), builder.ROOT/'analysis_sessions/session_inventory.py',
                      builder.ROOT/'fuse_and_analyze.py', builder.ROOT/'hrv_monitor_prototype/calculate_hrv.py',
                      builder.ROOT/'hrv_monitor_prototype/preprocess_rr.py', builder.ROOT/'hrv_monitor_prototype/config.py']
        manifest = {
            'schema_version': 2, 'generated_at_utc': datetime.now(UTC).isoformat(),
            'input_directory': str(input_dir), 'selected_folders': [p.name for p in directories],
            'selected_session_ids': [r['session_id'] for r in results[25][2]],
            'excluded_sessions': [r for r in inventory if r['reason'].startswith('excluded_session:')],
            'update_summary': update, 'window_seconds': list(builder.WINDOWS), 'step_seconds': 5,
            'target': 'cognitive_load_label',
            'label_policy': {'active_task': builder.ACTIVE_LABELS, 'Low_phases': list(builder.LOW_PHASES),
                             'window_rule': 'unanimous_target'},
            'quality_configuration': config, 'quality_configuration_sha256': shared.digest(config_path),
            'endpoint_packet_policy': 'include_available_packets_at_endpoint',
            'admission_policy': features.POLICY,
            'previous_admission_policy': previous_policy,
            'previous_manifest_sha256': shared.digest(output_dir/'manifest.json') if previous else None,
            'spatial_validation': validation_checks, 'timestamp_policies': timestamps,
            'rr_quality_control': rr_registry,
            'pre_model_audit': pre_model_audit,
            'feature_profile_counts': profile_counts, 'feature_columns': features.CORE,
            'dataset_eye_review_groups': {name: None for name in shared.DATASETS},
            'source_integrity_verified': True, 'source_sha256': sources_before,
            'code_sha256': {str(p.resolve().relative_to(builder.ROOT)).replace('\\', '/'): shared.digest(p) for p in code_paths},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'datasets': counts, 'output_sha256': shared.snapshot(destination),
        }
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        index = Path(temporary)/'README.md'
        archive_csv = output_dir.parent/'v1_active_task/good_acceptable.csv'
        if output_dir.name == 'v2_session_phase' and archive_csv.exists():
            from .v2_feature_documentation import output_index
            index.write_text(output_index(archive_csv, counts), encoding='utf-8')
        integrity.publish_bundle(destination, output_dir, replace=replace)
        if index.exists():
            shared.publish_artifact(index, output_dir.parent/'README.md')
    print(json.dumps(counts, indent=2), flush=True)
    return manifest
