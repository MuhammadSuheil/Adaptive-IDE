"""Build and verify observed-rating datasets; no model fitting or synthetic labels."""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from . import build_dataset as shared
from . import v2_features as features
from . import v2_integrity as integrity
from . import v3_behavior as traces
from . import v4_ratings as ratings
from .build_dataset_v2 import unique_duration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name('output')/'v2_session_phase'
DEFAULT_LOGS = ROOT/'codearrange-web/logs/legacy-backup/rating_soal'
DEFAULT_OUTPUT = Path(__file__).with_name('output')/'v4_self_reported_rating'
WINDOW = 'window_15s_step_5s'
PACKAGE_FILES = ('episode_ratings.csv', WINDOW+'/good_acceptable.csv', WINDOW+'/window_assignments.csv')
PROTECTED = ('v1_active_task', 'v2_session_phase', 'v3_behavioral_proxy')


def read_frame(path):
    return pd.read_csv(path, low_memory=False, float_precision='round_trip')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def same_csv(path, frame):
    expected = pd.read_csv(StringIO(frame.to_csv(index=False)), low_memory=False, float_precision='round_trip')
    try:
        pd.testing.assert_frame_equal(read_frame(path), expected, check_dtype=False, rtol=1e-10, atol=1e-10)
    except AssertionError as error:
        raise ValueError(f'Inconsistent derived artifact: {path}') from error


def rating_hashes(directory):
    return {name: shared.digest(integrity._no_links(directory/name)) for name in ratings.RATING_FILES}


def derive(source, manifest, log_directory):
    metadata = pd.read_csv(source/'window_25s_step_5s/session_quality_summary.csv', keep_default_na=False)
    if set(metadata.session_id) != set(manifest['selected_session_ids']) or metadata.session_id.duplicated().any():
        raise ValueError('Source cohort mismatch.')
    episodes, attempts, audits, conflicts = [], {}, {}, set()
    fields = ('session_id', 'event_time_ms', 'puzzle_set', 'puzzle_id', 'submission_type', 'outcome')
    for row in metadata.to_dict('records'):
        relative = f'{row["session_folder"]}/task_events.csv'
        path = integrity._no_links(Path(manifest['input_directory'])/relative)
        if shared.digest(path) != manifest['source_sha256'][relative]:
            raise ValueError('Source ledger changed; rebuild v2 first.')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            events = list(csv.DictReader(stream))
        result, audit = traces.reconstruct(events, row['participant_id'], row['session_id'], row['session_folder'])
        episodes.extend(result)
        audits[row['session_id']] = dict(audit, session_folder=row['session_folder'])
        for e in events:
            aid = e.get('attempt_id')
            if e.get('event_type') != 'attempt_submitted' or not aid:
                continue
            if aid in attempts and any(e.get(k) != attempts[aid].get(k) for k in fields):
                conflicts.add(aid)
            else:
                attempts[aid] = e
        conflicts.update(audit['conflicting_attempt_ids'])
    for aid in conflicts:
        attempts.pop(aid, None)
    if len({ep['episode_id'] for ep in episodes}) != len(episodes):
        raise ValueError('Duplicate episode IDs.')
    episodes, matching = ratings.match_ratings(ratings.load_ratings(log_directory), attempts, episodes)
    rated = [e for e in episodes if e[ratings.TARGET] is not None]
    summary = {'episodes': len(episodes), 'valid_episodes': sum(e['valid'] for e in episodes),
        'rated_episodes': len(rated), 'rated_participants': len({e['participant_id'] for e in rated}),
        'episode_statuses': dict(Counter(e['rating_status'] for e in episodes)),
        'episode_rating_counts': {str(r): sum(e[ratings.TARGET] == r for e in rated) for r in ratings.RATINGS},
        'rating_log_rows': len(matching), 'matching_statuses': matching.status.value_counts().to_dict(),
        'rated_submission_types': dict(Counter(attempts[e['rated_attempt_id']]['submission_type'] for e in rated)),
        'source_sessions_without_ratings': sorted(set(metadata.session_folder)-{e['session_folder'] for e in rated}),
        'conflicting_attempt_ids': sorted(conflicts)}
    return episodes, matching, audits, summary


def distributions(views, mapping, seconds):
    assigned = mapping.set_index('window_id').episode_id
    counts, rows = {}, []
    for name, data in views.items():
        counts[name] = {str(r): int(data[ratings.TARGET].eq(r).sum()) for r in ratings.RATINGS}
        counts[name].update(windows=len(data), participants=int(data.participant_id.nunique()),
                            episodes=int(data.window_id.map(assigned).nunique()), unique_seconds=unique_duration(data))
        for rating in ratings.RATINGS:
            part = data.loc[data[ratings.TARGET].eq(rating)]
            rows.append({'window_seconds': seconds, 'dataset': name, ratings.TARGET: rating,
                'windows': len(part), 'episodes': int(part.window_id.map(assigned).nunique()),
                'participants': int(part.participant_id.nunique()), 'unique_seconds': unique_duration(part)})
    return counts, pd.DataFrame(rows)


def check_source_exclusions(source, seconds, episodes):
    frame = pd.read_csv(source/f'window_{seconds}s_step_5s/excluded_windows.csv', keep_default_na=False)
    index = traces.WindowIndex(episodes)
    for row in frame.itertuples():
        ep, _ = index.match(row.session_id, shared.epoch_ms(row.window_start), shared.epoch_ms(row.window_end))
        if ep and ep['valid'] and ep[ratings.TARGET] is not None and 'ambiguous_eye_timestamp' not in row.exclusion_flags:
            raise ValueError('Potential rated window absent from v2 sensor pool; full sensor reconstruction required.')
    return frame.exclusion_reason.value_counts().to_dict()


def feature_contract(manifest):
    result = features.contract(manifest['quality_configuration'], manifest['spatial_validation'])
    result.update(dataset_version=4, target=ratings.TARGET, target_values=list(ratings.RATINGS),
        label_kind='observed_participant_perceived_task_difficulty', label_policy=ratings.POLICY,
        episode_mapping='window_assignments.csv', episode_evidence='../episode_ratings.csv',
        requires_fixed_participant_splits=False,
        evaluation_note='Select splits in the modeling pipeline; keep participant and episode provenance and purge overlapping intervals.')
    return result


def data_dictionary(source):
    frame = pd.read_csv(source/WINDOW/'data_dictionary.csv', keep_default_na=False)
    frame = frame.loc[~frame.column.isin(('cognitive_load_label', 'window_label_mix'))].copy()
    frame.loc[frame.column.eq('curation_selected'), 'description'] = 'Core sensor-quality selection; no v2 class quota or v3 balancing.'
    return pd.concat([frame, pd.DataFrame([{'column': ratings.TARGET, 'role': 'target', 'unit': 'integer_1_to_10',
        'description': 'Observed participant difficulty rating for the question episode; not a CL class or an imputed score.'}])], ignore_index=True)


def validate_views(views, mapping):
    for name, data in views.items():
        if not data.window_id.is_unique or not data[ratings.TARGET].isin(ratings.RATINGS).all():
            raise ValueError('Duplicate windows or invalid v4 targets.')
        if any(c in data for c in ('cognitive_load_label', 'window_label_mix', 'split', 'sample_weight')):
            raise ValueError('Leaking legacy targets or training state into v4.')
        target = data[ratings.TARGET].to_numpy(float)
        if not np.isfinite(target).all() or not np.equal(target, np.floor(target)).all():
            raise ValueError('Non-integer v4 target.')
        if not data.window_id.isin(mapping.window_id).all():
            raise ValueError('Unassigned rated window.')
        if name != 'all_data' and not np.isfinite(data[features.CORE].to_numpy(float)).all():
            raise ValueError('Invalid core sensor values in curated v4.')
    if not mapping.window_id.is_unique:
        raise ValueError('Duplicate episode assignments.')


def verify_bundle(directory, *, check_sources=True):
    directory = integrity._no_links(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('dataset_version') != 4 or manifest.get('label_policy') != ratings.POLICY \
            or manifest.get('feature_columns') != features.CORE or manifest.get('target') != ratings.TARGET:
        raise ValueError('Unsupported v4 bundle.')
    actual = shared.snapshot(directory)
    actual.pop('manifest.json')
    if actual != manifest['output_sha256']:
        raise ValueError('V4 output checksum mismatch.')
    if not check_sources:
        return manifest
    source, log_dir = Path(manifest['source_v2_directory']), Path(manifest['rating_log_directory'])
    if shared.snapshot(source) != manifest['source_v2_sha256'] or rating_hashes(log_dir) != manifest['rating_source_sha256']:
        raise ValueError('V4 sources changed; an explicit rebuild is required.')
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_sha256'] != original['source_sha256'] or \
            shared.snapshot(Path(original['input_directory'])) != manifest['source_sha256']:
        raise ValueError('Raw sensor source integrity mismatch.')
    episodes, matches, audits, summary = derive(source, original, log_dir)
    if summary != manifest['summary'] or json.loads((directory/'event_audit.json').read_text(encoding='utf-8')) != audits:
        raise ValueError('Rating/event reconstruction mismatch.')
    same_csv(directory/'episode_ratings.csv', ratings.episode_frame(episodes))
    same_csv(directory/'rating_matches.csv', matches)
    coverage = []
    for seconds in (15, 25, 45):
        views, mapping, excluded = ratings.project(read_frame(source/f'window_{seconds}s_step_5s/all_data.csv'),
            episodes, original['quality_configuration'], original['spatial_validation'])
        validate_views(views, mapping)
        counts, classes = distributions(views, mapping, seconds)
        coverage.append(classes.loc[classes.dataset.eq('good_acceptable')])
        if manifest['coverage'][str(seconds)] != counts['good_acceptable']:
            raise ValueError('Window coverage mismatch.')
        if manifest['source_pool_exclusions'][str(seconds)] != check_source_exclusions(source, seconds, episodes):
            raise ValueError('Source exclusions mismatch.')
        if seconds == 15:
            if counts != manifest['datasets']:
                raise ValueError('V4 dataset count mismatch.')
            for name, data in [*views.items(), ('window_assignments', mapping), ('excluded_windows', excluded),
                               ('class_distribution', classes), ('data_dictionary', data_dictionary(source))]:
                same_csv(directory/WINDOW/f'{name}.csv', data)
    same_csv(directory/'coverage_by_window.csv', pd.concat(coverage, ignore_index=True))
    if json.loads((directory/WINDOW/'feature_columns.json').read_text(encoding='utf-8')) != feature_contract(original):
        raise ValueError('V4 feature contract mismatch.')
    return manifest


def guard_output(output, inputs=()):
    output = integrity._no_links(output).resolve()
    if ROOT.is_relative_to(output) or Path(__file__).resolve().is_relative_to(output) \
            or any(part in PROTECTED for part in output.parts) or output == DEFAULT_OUTPUT.parent.resolve():
        raise ValueError(f'Protected v4 output path: {output}')
    for value in inputs:
        value = integrity._no_links(value).resolve()
        if output.is_relative_to(value) or value.is_relative_to(output):
            raise ValueError('V4 output overlaps a protected input.')
    return output


def publish(staged, output, verifier, *, replace=False):
    staged, output = integrity._no_links(staged).resolve(), guard_output(output)
    if staged.is_relative_to(output) or output.is_relative_to(staged):
        raise ValueError('Overlapping publication paths.')
    verifier(staged)
    if output.exists():
        if not replace:
            raise FileExistsError(output)
        verifier(output)
    backup = output.with_name(f'.{output.name}.rollback-{uuid4().hex}') if output.exists() else None
    if backup:
        os.rename(output, backup)
    try:
        os.rename(staged, output)
        if os.name == 'nt':
            subprocess.run(['icacls', str(output), '/reset', '/T', '/q'], check=True, capture_output=True)
        verifier(output)
    except Exception:
        try:
            if output.exists():
                os.rename(output, staged)
            if backup:
                os.rename(backup, output)
        except OSError as error:
            raise RuntimeError(f'Publication recovery failed; old data preserved at {backup}') from error
        raise
    if backup:
        if backup.resolve().parent != output.parent or backup.is_symlink() or backup.is_junction() \
                or not backup.name.startswith(f'.{output.name}.rollback-'):
            raise ValueError('Unsafe rollback cleanup.')
        shutil.rmtree(backup)


def build_dataset(source=DEFAULT_SOURCE, log_directory=DEFAULT_LOGS, output=DEFAULT_OUTPUT, *, replace=False):
    from .v4_documentation import write_reports
    source, log_directory = integrity._no_links(source).resolve(), integrity._no_links(log_directory).resolve()
    output = guard_output(output, [source, log_directory])
    if output.exists():
        if not replace:
            raise FileExistsError(output)
        verify_bundle(output, check_sources=False)
    manifest = integrity.verify_bundle(source)
    if manifest['admission_policy'] != features.POLICY:
        raise ValueError('V4 requires verified feature-aware v2 input.')
    raw = Path(manifest['input_directory'])
    guard_output(output, [raw])
    source_hash, log_hash = shared.snapshot(source), rating_hashes(log_directory)
    if shared.snapshot(raw) != manifest['source_sha256']:
        raise ValueError('Raw sensors changed; rebuild v2 first.')
    episodes, matches, audits, summary = derive(source, manifest, log_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-v4-', dir=output.parent) as temp:
        dest = Path(temp)/'bundle'
        (dest/WINDOW).mkdir(parents=True)
        ratings.episode_frame(episodes).to_csv(dest/'episode_ratings.csv', index=False)
        matches.to_csv(dest/'rating_matches.csv', index=False)
        write_json(dest/'event_audit.json', audits)
        coverage, combined, counts, source_exclusions = {}, [], None, {}
        for seconds in (15, 25, 45):
            views, mapping, excluded = ratings.project(read_frame(source/f'window_{seconds}s_step_5s/all_data.csv'),
                episodes, manifest['quality_configuration'], manifest['spatial_validation'])
            validate_views(views, mapping)
            all_counts, classes = distributions(views, mapping, seconds)
            coverage[str(seconds)] = all_counts['good_acceptable']
            combined.append(classes.loc[classes.dataset.eq('good_acceptable')])
            source_exclusions[str(seconds)] = check_source_exclusions(source, seconds, episodes)
            if seconds == 15:
                counts = all_counts
                for name, data in [*views.items(), ('window_assignments', mapping), ('excluded_windows', excluded),
                                   ('class_distribution', classes), ('data_dictionary', data_dictionary(source))]:
                    data.to_csv(dest/WINDOW/f'{name}.csv', index=False)
                write_json(dest/WINDOW/'feature_columns.json', feature_contract(manifest))
        comparison = pd.concat(combined, ignore_index=True)
        comparison.to_csv(dest/'coverage_by_window.csv', index=False)
        write_reports(dest, counts, comparison, summary, episodes, matches, source_exclusions)
        if shared.snapshot(source) != source_hash or rating_hashes(log_directory) != log_hash \
                or shared.snapshot(raw) != manifest['source_sha256']:
            raise RuntimeError('Inputs changed during v4 build; nothing published.')
        code_paths = [Path(__file__), Path(ratings.__file__), Path(traces.__file__), Path(features.__file__),
                      Path(__file__).with_name('v4_documentation.py')]
        result = {'dataset_version': 4, 'label_policy': ratings.POLICY, 'target': ratings.TARGET,
            'target_values': list(ratings.RATINGS), 'feature_columns': features.CORE,
            'generated_at_utc': datetime.now(UTC).isoformat(), 'source_v2_directory': str(source),
            'source_v2_sha256': source_hash, 'input_directory': str(raw), 'source_sha256': manifest['source_sha256'],
            'rating_log_directory': str(log_directory), 'rating_source_sha256': log_hash,
            'sensor_admission_policy': manifest['admission_policy'], 'quality_configuration': manifest['quality_configuration'],
            'timestamp_policies': manifest.get('timestamp_policies', {}), 'spatial_validation': manifest['spatial_validation'],
            'window_seconds': [15], 'step_seconds': 5, 'datasets': counts, 'coverage': coverage,
            'source_pool_exclusions': source_exclusions, 'summary': summary,
            'readiness': {'all_ten_ratings_present': all(counts['good_acceptable'][str(r)] > 0 for r in ratings.RATINGS),
                'statistical_sufficiency_established': False, 'split_and_weighting': 'deferred_to_modeling_pipeline'},
            'code_sha256': {p.relative_to(ROOT).as_posix(): shared.digest(p) for p in code_paths},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'output_sha256': shared.snapshot(dest)}
        write_json(dest/'manifest.json', result)
        # Validate the old bundle before publication; full checks run on the new bundle in both locations.
        old_manifest_hash = shared.digest(output/'manifest.json') if output.exists() else None
        def check(path):
            archived = Path(path) == output and old_manifest_hash and shared.digest(Path(path)/'manifest.json') == old_manifest_hash
            return verify_bundle(path, check_sources=not archived)
        publish(dest, output, check, replace=replace)
    return result


def verify_package(directory, bundle=None):
    directory = integrity._no_links(directory).resolve()
    text = (directory/'README.md').read_text(encoding='utf-8')
    matches = re.findall(r'^\| `([^`]+)` \| [0-9]+ \| `([0-9a-f]{64})` \|$', text, flags=re.MULTILINE)
    checks = dict(matches)
    if set(checks) != set(PACKAGE_FILES) or len(matches) != len(PACKAGE_FILES):
        raise ValueError('Unrecognized v4 package checksums.')
    expected_files = {'README.md', *PACKAGE_FILES}
    if set(shared.snapshot(directory)) != expected_files:
        raise ValueError('Unexpected or missing package artifacts.')
    for name, digest in checks.items():
        path = integrity._no_links(directory/name)
        if shared.digest(path) != digest or bundle is not None and shared.digest(Path(bundle)/name) != digest:
            raise ValueError('V4 package checksum mismatch.')
    return checks


def export_package(bundle, output, *, replace=False):
    from .v4_documentation import package_readme
    bundle = integrity._no_links(bundle).resolve()
    manifest = verify_bundle(bundle)
    output = guard_output(output, [bundle, Path(manifest['source_v2_directory']), Path(manifest['input_directory']),
                                  Path(manifest['rating_log_directory'])])
    if output.exists():
        if not replace:
            raise FileExistsError(output)
        verify_package(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.package-v4-', dir=output.parent) as temp:
        stage = Path(temp)/'package'
        (stage/WINDOW).mkdir(parents=True)
        for name in PACKAGE_FILES:
            shutil.copyfile(bundle/name, stage/name)
        (stage/'README.md').write_text(package_readme(stage, manifest), encoding='utf-8')
        old_hashes = verify_package(output) if output.exists() else None
        def check(path):
            archived = Path(path) == output and old_hashes is not None and verify_package(path) == old_hashes
            return verify_package(path, None if archived else bundle)
        publish(stage, output, check, replace=replace)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--rating-logs', type=Path, default=DEFAULT_LOGS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--verify-package', type=Path)
    args = parser.parse_args()
    if args.verify_package:
        print(json.dumps(verify_package(args.verify_package), indent=2))
        return
    result = verify_bundle(args.verify) if args.verify else build_dataset(args.source, args.rating_logs, args.output, replace=args.replace)
    if args.package:
        export_package(args.verify or args.output, args.package, replace=args.replace)
    print(json.dumps({'summary': result['summary'], 'datasets': result['datasets'], 'readiness': result['readiness']}, indent=2))


if __name__ == '__main__':
    main()
