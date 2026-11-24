"""Build a separate, auditable behavioral-proxy bundle from verified v2 pools."""

import argparse
import csv
import json
import os
import platform
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
from . import v3_behavior as behavior
from . import v3_curation as curation
from .build_dataset_v2 import unique_duration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name('output')/'v2_session_phase'
DEFAULT_OUTPUT = Path(__file__).with_name('output')/'v3_behavioral_proxy'
WINDOWS = (15, 25, 45)
SPLITS = ('Train', 'validation', 'Test')
MAP_COLUMNS = ['window_id', 'participant_id', 'session_id', 'episode_id', 'split',
               'window_start', 'window_end', 'cognitive_load_label', 'reason', 'source_v2_label', 'quality_tier',
               'source_phase', 'puzzle_set', 'puzzle_id']


def policy_contract(version=behavior.POLICY):
    if version not in (behavior.LEGACY_POLICY, behavior.POLICY):
        raise ValueError('Unsupported v3 label policy version.')
    result = {'version': version, 'label_kind': 'exploratory_behavioral_proxy_not_measured_cognitive_load',
            'labels': list(behavior.LABELS), 'unit': 'question_episode_until_first_success_or_censored_end',
            'Low': 'manual_first_submission_correct_without_manual_error_and_time_le_lower_quantile',
            'Medium': 'completed_with_1_or_2_manual_errors_and_lower_lt_time_le_upper_quantile',
            'High': 'at_least_3_manual_errors_and_observed_active_time_gt_upper_quantile',
            'otherwise': 'unlabeled', 'reference_quantiles': [1/3,2/3], 'quantile_method': 'linear',
            'minimum_other_train_people': 5, 'reference_exposure': 'first_exposure_only_if_successful_manual',
            'reference_item': ['puzzle_set','puzzle_id','normalized_title'], 'self_reference': False,
            'reference_split': 'Train', 'window_rule': 'one_continuous_active_episode',
            'sensor_features_used_for_labels': False, 'difficulty_used_as_label': False,
            'missing_ratings_imputed': False, 'class_balancing': False,
            'split_policy': 'participant_sha256_v3_pipe_42_pipe_id_validation_test_each_round_15_percent'}
    if version != behavior.LEGACY_POLICY:
        result.update(reference_exposure='observed_first_exposure_excluding_documented_prior_exposure',
                      reference_item='ledger_identity_with_hash_pinned_verified_aliases',
                      split_policy='bounded_coverage_v1',
                      sampling_policy='train_class_person_episode_v1',
                      class_balancing='optional_train_only_views_without_relabeling')
    return result


def derive(source, manifest, *, policy=behavior.POLICY, registry=None):
    metadata = pd.read_csv(source/'window_25s_step_5s/session_quality_summary.csv', keep_default_na=False)
    if set(metadata.session_id) != set(manifest['selected_session_ids']):
        raise ValueError('Source cohort mismatch.')
    episodes, audits = [], {}
    for row in metadata.to_dict('records'):
        path = Path(manifest['input_directory'])/row['session_folder']/'task_events.csv'
        source_key = f"{row['session_folder']}/task_events.csv"
        if shared.digest(path) != manifest['source_sha256'][source_key]:
            raise ValueError('Source event ledger changed; rebuild v2 first.')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            events = list(csv.DictReader(stream))
        result, audit = behavior.reconstruct(events, row['participant_id'], row['session_id'], row['session_folder'])
        episodes.extend(result)
        audits[row['session_id']] = dict(session_folder=row['session_folder'], **audit)
    extras = {}
    if policy == behavior.LEGACY_POLICY:
        splits = behavior.partition(metadata.participant_id.tolist())
    else:
        registry = curation.load_aliases() if registry is None else curation.validate_aliases(registry)
        episodes = curation.prepare_episodes(episodes, registry,
            manifest['quality_configuration'].get('repeated_difficulties_by_session', {}))
        index = behavior.WindowIndex(episodes)
        support = {}
        for seconds in WINDOWS:
            count = Counter()
            for row in read_frame(source/f'window_{seconds}s_step_5s/all_data.csv').itertuples():
                if row.quality_tier not in ('good', 'acceptable'):
                    continue
                episode, _ = index.match(row.session_id, shared.epoch_ms(row.window_start), shared.epoch_ms(row.window_end))
                if episode and all(s['study_phase'] == 'active_task' for s in json.loads(row.window_phase_mix)
                                   if s['duration_seconds'] > 0):
                    count[episode['episode_id']] += 1
            support[seconds] = count
        splits, search = curation.select_partition(episodes, support, people=metadata.participant_id.tolist())
        extras = {'item_aliases': registry, 'partition_search': search}
    reference = behavior.make_reference(episodes, splits, policy=policy)
    seen = set()
    for episode in sorted(episodes, key=lambda r: (r['start_ms'],r['episode_id'])):
        key = episode['participant_id'], episode['item_key']
        if policy == behavior.LEGACY_POLICY:
            episode['first_exposure'] = key not in seen
        seen.add(key)
        episode['split'] = splits[episode['participant_id']]
        episode.update(behavior.label_episode(episode, reference))
    if len({r['episode_id'] for r in episodes}) != len(episodes):
        raise ValueError('Duplicate episode IDs.')
    if policy != behavior.LEGACY_POLICY:
        extras['reference_sensitivity'] = curation.reference_sensitivity(episodes, reference)
    return episodes, splits, reference, audits, extras


def episode_frame(episodes):
    rows = []
    for episode in episodes:
        row = episode.copy()
        for key in ('intervals','attempt_ids','invalid_reasons','reference_participants'):
            row[key] = json.dumps(row[key], ensure_ascii=False, separators=(',', ':'))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['participant_id','start_ms','episode_id']).reset_index(drop=True)


def read_frame(path):
    return pd.read_csv(path, low_memory=False, float_precision='round_trip')


def project(frame, index, config, checks):
    rows, mappings, excluded = [], [], []
    for row in frame.to_dict('records'):
        episode, reason = index.match(row['session_id'], shared.epoch_ms(row['window_start']), shared.epoch_ms(row['window_end']))
        if episode is not None:
            if episode['participant_id'] != row['participant_id']:
                raise ValueError('Window/episode participant mismatch.')
            if any(s['study_phase'] != 'active_task' for s in json.loads(row['window_phase_mix']) if s['duration_seconds'] > 0):
                reason = 'source_phase_disagreement'
            else:
                reason = '' if episode['cognitive_load_label'] else episode['label_reason']
        mapping = {'window_id': row['window_id'], 'participant_id': row['participant_id'], 'session_id': row['session_id'],
            'episode_id': episode['episode_id'] if episode else '', 'split': episode['split'] if episode else '',
            'window_start': row['window_start'], 'window_end': row['window_end'],
            'cognitive_load_label': episode['cognitive_load_label'] if episode and not reason else '',
            'reason': reason or 'concordant_behavioral_evidence', 'source_v2_label': row['cognitive_load_label'],
            'quality_tier': row['quality_tier'], 'source_phase': row['study_phase'],
            'puzzle_set': row['puzzle_set'], 'puzzle_id': row['puzzle_id']}
        if reason:
            excluded.append(mapping)
            continue
        mappings.append(mapping)
        row['cognitive_load_label'] = episode['cognitive_load_label']
        row['window_label_mix'] = json.dumps({episode['cognitive_load_label']:float(row['window_seconds'])}, separators=(',', ':'))
        rows.append(row)
    labeled = pd.DataFrame(rows, columns=frame.columns)
    original_quality = labeled.quality_tier.copy()
    labeled = features.apply_policy(labeled, config)
    if labeled.quality_tier.tolist() != original_quality.tolist():
        raise ValueError('V3 must not silently change source quality tiers.')
    views = {name: (labeled.copy() if name == 'all_data' else features.curated_projection(
        labeled.loc[labeled.quality_tier.isin(tiers)].copy(), config, checks)) for name,tiers in shared.DATASETS.items()}
    return views, pd.DataFrame(mappings, columns=MAP_COLUMNS), pd.DataFrame(excluded, columns=MAP_COLUMNS)


def distributions(views, mapping, *, revised=False):
    counts, classes, split_rows = {}, [], []
    for name, view in views.items():
        counts[name] = {label:int(view.cognitive_load_label.eq(label).sum()) for label in behavior.LABELS}
        counts[name].update(windows=len(view), participants=int(view.participant_id.nunique()), unique_seconds=unique_duration(view))
        for label in behavior.LABELS:
            part = view.loc[view.cognitive_load_label.eq(label)]
            classes.append({'dataset': name, 'label': label, 'windows': len(part), 'participants': part.participant_id.nunique(),
                                'unique_seconds': unique_duration(part)})
        assigned = mapping.set_index('window_id').split if len(mapping) else pd.Series(dtype=str)
        roles = view.window_id.map(assigned)
        for split in SPLITS:
            for label in behavior.LABELS:
                part = view.loc[roles.eq(split) & view.cognitive_load_label.eq(label)]
                split_rows.append({'dataset': name, 'split': split, 'label': label, 'windows': len(part),
                                       'participants': part.participant_id.nunique(), 'unique_seconds': unique_duration(part)})
    status = {'primary_has_all_classes':all(counts['good_acceptable'][c] > 0 for c in behavior.LABELS),
              'missing_classes_by_split':{s:[c for c in behavior.LABELS if not any(
                  r['dataset']=='good_acceptable' and r['split']==s and r['label']==c and r['windows'] > 0
                  for r in split_rows)] for s in SPLITS}}
    status['evaluation_has_all_classes_in_every_split'] = not any(status['missing_classes_by_split'].values())
    status['interpretation'] = 'exploratory_proxy_only_not_validated_mental_load'
    if revised:
        support = {s: {label: next(r['participants'] for r in split_rows
                    if r['dataset'] == 'good_acceptable' and r['split'] == s and r['label'] == label)
                      for label in behavior.LABELS} for s in SPLITS}
        status['participants_by_split_class'] = support
        status['minimum_people_met'] = all(support[s][label] >= (3 if s == 'Train' else 2)
                                           for s in SPLITS for label in behavior.LABELS)
        status['technical_status'] = ('supported_for_exploratory_evaluation' if status['minimum_people_met'] else
            'limited_support_secondary_only' if status['evaluation_has_all_classes_in_every_split'] else
            'missing_evaluation_classes')
    return counts, pd.DataFrame(classes), pd.DataFrame(split_rows), status


def feature_contract(source_manifest, *, policy=behavior.POLICY):
    result = features.contract(source_manifest['quality_configuration'], source_manifest['spatial_validation'])
    result.update(dataset_version=3, label_policy=policy,
                  label_kind='exploratory_behavioral_proxy_not_validated_cognitive_load',
                  requires_fixed_participant_splits=True, participant_splits='../participant_splits.csv',
                  episode_mapping='window_assignments.csv',
                  evaluation_note='Use frozen Train/validation/Test participant roles; a different split requires relabeling.')
    if policy != behavior.LEGACY_POLICY:
        result.update(train_sample_weights='train_sample_weights.csv', balanced_train='train_balanced.csv',
                      sampling_source='good_acceptable.csv', default_window_seconds=15,
                      sampling_note='Alternative Train treatments; never multiply full-pool weights onto the balanced view.')
    return result


def _same_csv(actual, expected):
    normalized = pd.read_csv(StringIO(expected.to_csv(index=False)), low_memory=False, float_precision='round_trip')
    try:
        pd.testing.assert_frame_equal(read_frame(actual), normalized, check_dtype=False, rtol=1e-10, atol=1e-10)
    except AssertionError as error:
        raise ValueError(f'Inconsistent derived artifact: {actual}') from error


def _check_source_exclusions(source, seconds, index):
    exclusions = pd.read_csv(source/f'window_{seconds}s_step_5s/excluded_windows.csv',keep_default_na=False,low_memory=False)
    for row in exclusions.itertuples():
        episode, _ = index.match(row.session_id, shared.epoch_ms(row.window_start), shared.epoch_ms(row.window_end))
        if episode and episode['cognitive_load_label'] and 'ambiguous_eye_timestamp' not in row.exclusion_flags:
            # A missing aggregate cannot be invented or silently omitted by reusing v2.
            raise ValueError(f'Potential v3 window absent from v2 sensor pool: {row.window_id}; full sensor reconstruction required.')
    return exclusions.exclusion_reason.value_counts().to_dict()


def verify_bundle(directory, *, check_sources=True):
    directory = integrity._no_links(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    policy = manifest.get('label_policy', {}).get('version')
    if manifest.get('dataset_version') != 3 or manifest.get('label_policy') != policy_contract(policy):
        raise ValueError('Unsupported v3 behavioral bundle.')
    actual = shared.snapshot(directory)
    actual.pop('manifest.json')
    if actual != manifest['output_sha256']:
        raise ValueError('V3 checksum mismatch.')
    if not check_sources:
        # Replacing a historical bundle must remain possible after its v2 source changed.
        # This checks the archived artifacts, not the now-unavailable old raw inputs.
        _same_csv(directory/'participant_splits.csv',pd.DataFrame(
            sorted(manifest['participant_splits'].items()),columns=['participant_id','split']))
        if policy == behavior.LEGACY_POLICY and manifest['participant_splits'] != behavior.partition(manifest['participant_splits']):
            raise ValueError('Archived partition mismatch.')
        if policy != behavior.LEGACY_POLICY:
            search = json.loads((directory/'partition_search.json').read_text(encoding='utf-8'))
            if search != manifest['partition_search'] or curation.content_hash(manifest['participant_splits']) != \
                    search['candidates'][search['selected_candidate_index']]['partition_sha256']:
                raise ValueError('Archived coverage partition mismatch.')
            curation.load_aliases(directory/'item_aliases.json')
        for seconds in WINDOWS:
            key=f'window_{seconds}s_step_5s'
            for name, expected in manifest['datasets'][key].items():
                frame=read_frame(directory/key/f'{name}.csv')
                actual_counts={label:int(frame.cognitive_load_label.eq(label).sum()) for label in behavior.LABELS}
                actual_counts.update(windows=len(frame),participants=int(frame.participant_id.nunique()),
                                     unique_seconds=unique_duration(frame))
                if actual_counts != expected:
                    raise ValueError('Archived counts mismatch.')
        return manifest
    source = Path(manifest['source_v2_directory'])
    if shared.snapshot(source) != manifest['source_v2_sha256']:
        raise ValueError('V2 source bundle changed; v3 requires an explicit rebuild.')
    source_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_sha256'] != source_manifest['source_sha256']:
        raise ValueError('Raw provenance mismatch.')
    registry = curation.load_aliases(directory/'item_aliases.json') if policy != behavior.LEGACY_POLICY else None
    episodes, splits, reference, audits, extras = derive(source, source_manifest, policy=policy, registry=registry)
    if json.loads((directory/'label_reference.json').read_text()) != reference:
        raise ValueError('Reference time reconstruction mismatch.')
    if json.loads((directory/'event_audit.json').read_text()) != audits:
        raise ValueError('Event audit mismatch.')
    _same_csv(directory/'episode_labels.csv', episode_frame(episodes))
    _same_csv(directory/'participant_splits.csv', pd.DataFrame(sorted(splits.items()),columns=['participant_id','split']))
    if manifest['participant_splits'] != splits or manifest['episode_summary'] != summarize_episodes(episodes):
        raise ValueError('Episode/partition manifest mismatch.')
    if manifest['feature_columns'] != features.CORE:
        raise ValueError('V3 feature allowlist mismatch.')
    if policy != behavior.LEGACY_POLICY:
        if json.loads((directory/'partition_search.json').read_text(encoding='utf-8')) != extras['partition_search'] \
                or manifest['partition_search'] != extras['partition_search']:
            raise ValueError('Coverage search reconstruction mismatch.')
        _same_csv(directory/'reference_sensitivity.csv', extras['reference_sensitivity'])
        if manifest['reference_audit'] != reference_audit(episodes, reference):
            raise ValueError('Reference audit mismatch.')
        verify_history(directory, manifest, episodes)
    index = behavior.WindowIndex(episodes)
    for seconds in WINDOWS:
        key = f'window_{seconds}s_step_5s'
        pool = read_frame(source/key/'all_data.csv')
        views, mappings, excluded = project(pool,index,source_manifest['quality_configuration'],source_manifest['spatial_validation'])
        counts, classes, split_counts, status = distributions(views,mappings, revised=policy != behavior.LEGACY_POLICY)
        if manifest['datasets'][key] != counts or manifest['readiness'][key] != status:
            raise ValueError('V3 counts or readiness mismatch.')
        if manifest['source_pool_exclusions'][key] != _check_source_exclusions(source,seconds,index):
            raise ValueError('Source-pool exclusion audit mismatch.')
        for name, frame in views.items():
            _same_csv(directory/key/f'{name}.csv',frame)
        for name, frame in [('window_assignments',mappings),('excluded_windows',excluded),
                            ('class_distribution',classes),('split_class_distribution',split_counts)]:
            _same_csv(directory/key/f'{name}.csv',frame)
        if policy != behavior.LEGACY_POLICY:
            weights, balanced = curation.training_views(views['good_acceptable'], mappings)
            _same_csv(directory/key/'train_sample_weights.csv', weights)
            _same_csv(directory/key/'train_balanced.csv', balanced)
        if json.loads((directory/key/'feature_columns.json').read_text()) != feature_contract(source_manifest, policy=policy):
            raise ValueError('V3 feature contract mismatch.')
    return manifest


def summarize_episodes(episodes):
    return {'total':len(episodes), 'valid':sum(r['valid'] for r in episodes),
            'labels':dict(Counter(r['cognitive_load_label'] or 'unlabeled' for r in episodes)),
            'label_reasons':dict(Counter(r['label_reason'] for r in episodes)),
            'invalid_reasons':dict(Counter(k for r in episodes for k in r['invalid_reasons'])),
            'by_split':{s:dict(Counter(r['cognitive_load_label'] or 'unlabeled' for r in episodes if r['split']==s))
                        for s in SPLITS}}


def reference_audit(episodes, reference):
    donors = {d['episode_id'] for values in reference['items'].values() for d in values}
    return {'eligible_episodes': sum(e['reference_eligible'] for e in episodes),
            'known_prior_exposure_episodes': sum(e['known_prior_exposure'] for e in episodes),
            'donor_episodes': len(donors),
            'excluded_reasons': dict(Counter(e['reference_exclusion_reason'] for e in episodes
                                             if e['reference_exclusion_reason'])),
            'known_repeat_donors': sum(e['known_prior_exposure'] and e['episode_id'] in donors for e in episodes)}


def revision_changes(previous, episodes):
    old = {r['episode_id']: r for r in previous.fillna('').to_dict('records')}
    new = {r['episode_id']: r for r in episodes}
    rows = []
    for eid in sorted(set(old) | set(new)):
        before, after = old.get(eid, {}), new.get(eid, {})
        flags = []
        if not before or not after:
            flags.append('new_episode' if after else 'removed_episode')
        if before.get('item_key', '') != after.get('item_key', ''):
            flags.append('item_identity_changed')
        if before.get('split', '') != after.get('split', ''):
            flags.append('split_changed')
        if after.get('known_prior_exposure'):
            flags.append('known_prior_exposure_excluded_from_reference')
        if str(before.get('reference_participants', '')) != json.dumps(after.get('reference_participants', []),
                                                                       ensure_ascii=False, separators=(',', ':')):
            flags.append('reference_population_changed')
        rows.append({'episode_id': eid, 'participant_id': after.get('participant_id', before.get('participant_id', '')),
            'old_label': before.get('cognitive_load_label', ''), 'new_label': after.get('cognitive_load_label', ''),
            'old_reason': before.get('label_reason', ''), 'new_reason': after.get('label_reason', ''),
            'old_split': before.get('split', ''), 'new_split': after.get('split', ''),
            'old_item_key': before.get('item_key', ''), 'new_item_key': after.get('item_key', ''),
            'old_reference_count': before.get('reference_count', ''), 'new_reference_count': after.get('reference_count', ''),
            'label_changed': before.get('cognitive_load_label', '') != after.get('cognitive_load_label', ''),
            'change_flags': json.dumps(flags, separators=(',', ':'))})
    return pd.DataFrame(rows)


def write_history(previous, destination, episodes):
    if not previous.exists():
        return None
    digest = shared.digest(previous/'manifest.json')
    if (previous/'history').exists():
        shutil.copytree(previous/'history', destination/'history')
    folder = destination/'history'/digest[:16]
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'episode_labels.csv', 'participant_splits.csv', 'label_reference.json',
                 'item_aliases.json', 'partition_search.json', 'reference_sensitivity.csv'):
        if (previous/name).exists():
            shutil.copyfile(previous/name, folder/name)
    revision_changes(read_frame(folder/'episode_labels.csv'), episodes).to_csv(destination/'revision_changes.csv', index=False)
    return {'previous_manifest_sha256': digest, 'previous_bundle_record': folder.relative_to(destination).as_posix()}


def verify_history(directory, manifest, episodes):
    record = manifest.get('revision_history')
    if not record:
        return
    relative = Path(record['previous_bundle_record'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'history':
        raise ValueError('Invalid revision history path.')
    previous = directory/relative
    if shared.digest(previous/'manifest.json') != record['previous_manifest_sha256']:
        raise ValueError('Previous manifest checksum mismatch.')
    for folder in (directory/'history').iterdir():
        archived = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for artifact in folder.iterdir():
            if artifact.name != 'manifest.json' and shared.digest(artifact) != archived['output_sha256'].get(artifact.name):
                raise ValueError('Revision history evidence checksum mismatch.')
    _same_csv(directory/'revision_changes.csv', revision_changes(read_frame(previous/'episode_labels.csv'), episodes))


def _guard(source, output, *, replace=False):
    source = integrity._no_links(source).resolve()
    output = integrity._no_links(output).resolve()
    if (output.is_relative_to(source) or source.is_relative_to(output) or ROOT.is_relative_to(output)
            or Path(__file__).resolve().is_relative_to(output) or output == Path(__file__).with_name('output')
            or any(p in ('v1_active_task','v2_session_phase') for p in output.parts)):
        raise ValueError(f'Protected v3 output path: {output}')
    if output.exists():
        if not replace:
            raise FileExistsError(f'Output exists: {output}; use --replace only for a verified v3 bundle.')
        verify_bundle(output,check_sources=False)
    return source, output


def publish(staged, output, *, replace=False):
    staged, output = integrity._no_links(staged).resolve(), integrity._no_links(output).resolve()
    if (ROOT.is_relative_to(output) or staged.is_relative_to(output) or output.is_relative_to(staged)
            or any(p in ('v1_active_task','v2_session_phase') for p in output.parts)):
        raise ValueError('Protected publication target.')
    verify_bundle(staged)
    if output.exists():
        if not replace:
            raise FileExistsError(output)
        verify_bundle(output,check_sources=False)
    backup = output.with_name(f'.{output.name}.rollback-{uuid4().hex}') if output.exists() else None
    if backup:
        os.rename(output,backup)
    try:
        os.rename(staged,output)
        if os.name == 'nt':
            subprocess.run(['icacls',str(output),'/reset','/T','/q'],check=True,capture_output=True)
        verify_bundle(output)
    except Exception:
        try:
            if output.exists():
                os.rename(output,staged)
            if backup:
                os.rename(backup,output)
        except OSError as error:
            raise RuntimeError(f'V3 publication recovery failed; previous bundle preserved at {backup}') from error
        raise
    if backup:
        if (backup.resolve().parent != output.parent or backup.is_symlink() or backup.is_junction()
                or not backup.name.startswith(f'.{output.name}.rollback-')):
            raise ValueError('Unsafe v3 rollback cleanup path.')
        shutil.rmtree(backup)


def build_dataset(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, *, replace=False):
    from .v3_documentation import write_reports

    source, output = _guard(source,output,replace=replace)
    source_manifest = integrity.verify_bundle(source)
    if source_manifest['admission_policy'] != features.POLICY:
        raise ValueError('V3 requires a verified feature-aware v2 source.')
    raw_root = Path(source_manifest['input_directory']).resolve()
    if output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError('V3 output overlaps raw sources.')
    source_hashes = shared.snapshot(source)
    if shared.snapshot(raw_root) != source_manifest['source_sha256']:
        raise ValueError('Raw recordings changed; rebuild v2 first.')
    episodes, splits, reference, audits, extras = derive(source,source_manifest)
    index = behavior.WindowIndex(episodes)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-v3-',dir=output.parent) as temp:
        destination = Path(temp)/'bundle'
        destination.mkdir()
        episode_frame(episodes).to_csv(destination/'episode_labels.csv',index=False)
        pd.DataFrame(sorted(splits.items()),columns=['participant_id','split']).to_csv(destination/'participant_splits.csv',index=False)
        for name, value in [('label_reference',reference),('event_audit',audits)]:
            (destination/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for name in ('item_aliases', 'partition_search'):
            (destination/f'{name}.json').write_text(json.dumps(extras[name], ensure_ascii=False, indent=2,
                                                            allow_nan=False)+'\n', encoding='utf-8')
        extras['reference_sensitivity'].to_csv(destination/'reference_sensitivity.csv', index=False)
        history = write_history(output, destination, episodes)
        counts, readiness, exclusions, per_window = {}, {}, {}, {}
        for seconds in WINDOWS:
            key = f'window_{seconds}s_step_5s'
            folder = destination/key
            folder.mkdir()
            views,mappings,rejected = project(read_frame(source/key/'all_data.csv'),index,
                source_manifest['quality_configuration'],source_manifest['spatial_validation'])
            counts[key],classes,split_counts,readiness[key] = distributions(views,mappings, revised=True)
            exclusions[key] = _check_source_exclusions(source,seconds,index)
            weights, balanced = curation.training_views(views['good_acceptable'], mappings)
            per_window[seconds] = {'mapping': mappings,'excluded': rejected,'classes': classes,'split_counts': split_counts,
                                  'weights': weights, 'balanced': balanced}
            for name,frame in [*views.items(),('window_assignments',mappings),('excluded_windows',rejected),
                               ('class_distribution',classes),('split_class_distribution',split_counts),
                               ('train_sample_weights',weights),('train_balanced',balanced)]:
                frame.to_csv(folder/f'{name}.csv',index=False)
            (folder/'feature_columns.json').write_text(json.dumps(feature_contract(source_manifest),indent=2)+'\n',encoding='utf-8')
            dictionary = pd.read_csv(source/key/'data_dictionary.csv',keep_default_na=False)
            dictionary.loc[dictionary.column.eq('cognitive_load_label'),'description'] = 'Exploratory behavioral proxy from concordant active-time and manual-error evidence; not measured cognitive load.'
            dictionary.loc[dictionary.column.eq('window_label_mix'),'description'] = 'Duration map for the v3 behavioral-proxy target; not the source v2 protocol target.'
            dictionary.loc[dictionary.column.eq('curation_selected'),'description'] = 'Core sensor-quality selection only; v2 class quotas are not inherited.'
            dictionary.to_csv(folder/'data_dictionary.csv',index=False)
            print(f'{seconds}s: '+json.dumps(counts[key]['good_acceptable']),flush=True)
        summary = summarize_episodes(episodes)
        write_reports(destination,counts,readiness,summary,per_window,splits,extras=extras,
                      reference_audit=reference_audit(episodes, reference), episodes=episodes)
        if shared.snapshot(source) != source_hashes or shared.snapshot(raw_root) != source_manifest['source_sha256']:
            raise RuntimeError('Inputs changed; no v3 bundle published.')
        code_paths = [Path(__file__),Path(behavior.__file__),Path(curation.__file__),Path(features.__file__),
                      Path(__file__).with_name('v3_documentation.py'), Path(__file__).with_name('v3_item_aliases.json')]
        manifest = {'dataset_version': 3,'generated_at_utc': datetime.now(UTC).isoformat(),
            'source_v2_directory': str(source),'source_v2_sha256': source_hashes,
            'input_directory': str(raw_root),'source_sha256': source_manifest['source_sha256'],'source_integrity_verified': True,
            'label_policy': policy_contract(),'participant_splits': splits,'episode_summary': summary,
            'partition_search': extras['partition_search'], 'reference_audit': reference_audit(episodes, reference),
            'revision_history': history,
            'window_seconds': list(WINDOWS),'step_seconds': 5,'target': 'cognitive_load_label','feature_columns': features.CORE,
            'quality_configuration': source_manifest['quality_configuration'],'spatial_validation': source_manifest['spatial_validation'],
            'timestamp_policies': source_manifest.get('timestamp_policies',{}),
            'sensor_admission_policy': source_manifest['admission_policy'],'datasets': counts,'readiness': readiness,
            'source_pool_exclusions': exclusions,
            'code_sha256': {str(p.resolve().relative_to(ROOT)).replace('\\','/'):shared.digest(p) for p in code_paths},
            'versions': {'python': platform.python_version(),'numpy': np.__version__,'pandas': pd.__version__},
            'output_sha256': shared.snapshot(destination)}
        (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        publish(destination,output,replace=replace)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--replace',action='store_true')
    parser.add_argument('--verify',type=Path,help='Verify an existing v3 bundle without modifying it.')
    args = parser.parse_args()
    result = verify_bundle(args.verify) if args.verify else build_dataset(args.source,args.output,replace=args.replace)
    print(json.dumps({'dataset_version': result['dataset_version'],'datasets': result['datasets'],'readiness': result['readiness']},indent=2))


if __name__ == '__main__':
    main()
