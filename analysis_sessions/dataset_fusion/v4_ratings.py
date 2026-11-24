"""Observed question ratings, linked by recorded submission identity."""

import copy
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd

from . import build_dataset as shared
from . import v2_features as features
from . import v3_behavior as traces

POLICY = 'observed_participant_difficulty_rating_v1'
TARGET = 'difficulty_rating'
RATINGS = tuple(range(1, 11))
RATING_FILES = ('easy-1.csv', 'easy-2.csv', 'medium.csv', 'hard.csv')
LOG_FIELDS = ('timestamp', 'attempt_id', 'session_id', 'puzzle_set', 'puzzle_id',
              'puzzle_title', 'submission_type', 'correct', 'rating')
AUDIT_COLUMNS = ['source_file', 'source_row', 'attempt_id', 'session_id', 'rating_raw',
                 TARGET, 'episode_id', 'participant_id', 'status']
MAP_COLUMNS = ['window_id', 'participant_id', 'session_id', 'episode_id', 'window_start', 'window_end',
               TARGET, 'quality_tier', 'reason', 'source_v2_label']


def load_ratings(directory):
    rows = []
    for name in RATING_FILES:
        with (directory/name).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not set(LOG_FIELDS).issubset(reader.fieldnames or ()):
                raise ValueError(f'Incomplete rating log schema: {name}')
            for number, row in enumerate(reader, 2):
                if None in row or any(row[k] is None for k in LOG_FIELDS):
                    raise ValueError(f'Malformed rating log row: {name}:{number}')
                rows.append(dict(row, source_file=name, source_row=number))
    return rows


def _rating(value):
    text = str(value).strip() if value is not None else ''
    return int(text) if re.fullmatch(r'(?:[1-9]|10)', text) else None


def _timestamp(value):
    try:
        date = datetime.fromisoformat(str(value))
        return date.timestamp()*1000 if date.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _identity(row):
    return (_timestamp(row.get('timestamp')), row.get('session_id', ''), row.get('puzzle_set', ''),
            row.get('puzzle_id', ''), traces.normalize_title(row.get('puzzle_title', '')),
            row.get('submission_type', ''), str(row.get('correct', '')).lower())


def match_ratings(log_rows, attempts, episodes):
    rows = copy.deepcopy(episodes)
    owners = {}
    for ep in rows:
        ep.update(difficulty_rating=None, rating_status='missing_rating', rated_attempt_id='', rating_source_rows=[])
        for aid in ep['attempt_ids']:
            if aid in owners:
                raise ValueError('Attempt assigned to multiple episodes.')
            owners[aid] = ep
    groups, audits, matched = defaultdict(list), [], defaultdict(list)
    for row in log_rows:
        value = _rating(row.get('rating'))
        raw = str(row.get('rating') or '').strip()
        audit = {'source_file': row.get('source_file', ''), 'source_row': row.get('source_row', 0),
                 'attempt_id': row.get('attempt_id', ''), 'session_id': row.get('session_id', ''),
                 'rating_raw': raw, TARGET: value, 'episode_id': '', 'participant_id': '',
                 'status': 'missing_rating' if raw in ('', '-') else 'invalid_rating' if value is None else ''}
        audits.append(audit)
        groups[row.get('attempt_id', '')].append((row, audit))
    for aid, copies in groups.items():
        values = {a[TARGET] for _, a in copies if a[TARGET] is not None}
        conflict = len(values) > 1 or len({_identity(r) for r, _ in copies}) > 1
        ep = owners.get(aid)
        if conflict:
            for _, audit in copies:
                audit['status'] = 'conflicting_attempt_rows'
            if ep:
                ep['rating_status'] = 'conflicting_attempt_rows'
            continue
        accepted = False
        for row, audit in copies:
            if ep:
                audit.update(episode_id=ep['episode_id'], participant_id=ep['participant_id'])
            if audit['status']:
                continue
            event = attempts.get(aid)
            status = ('missing_attempt_id' if not aid else 'attempt_not_in_source_cohort' if event is None else
                      'attempt_not_assigned_to_episode' if ep is None else '')
            if not status:
                if any(not row.get(k) or row[k] != event.get(k) for k in ('session_id', 'puzzle_set', 'puzzle_id', 'submission_type')):
                    status = 'submission_identity_mismatch'
                elif traces.normalize_title(row.get('puzzle_title', '')) != traces.normalize_title(event.get('puzzle_title', '')):
                    status = 'puzzle_title_mismatch'
                elif str(row.get('correct', '')).lower() not in ('true', 'false') or \
                        (str(row['correct']).lower() == 'true') != (event.get('outcome') == 'correct'):
                    status = 'submission_outcome_mismatch'
                else:
                    time = _timestamp(row.get('timestamp'))
                    if time is None or not math.isclose(time, float(event['event_time_ms']), rel_tol=0, abs_tol=.001):
                        status = 'submission_timestamp_mismatch'
            if not status and not ep['valid']:
                status = 'invalid_event_trace'
            if not status and (aid != ep['attempt_ids'][-1] or ep['closed_by'] not in ('correct', 'timeout')):
                status = 'nonterminal_rated_attempt'
            audit['status'] = status or ('duplicate_rating' if accepted else 'matched')
            if not status:
                matched[ep['episode_id']].append((row, audit))
                accepted = True
            elif ep and ep['rating_status'] == 'missing_rating':
                ep['rating_status'] = status
    for ep in rows:
        candidates = matched.get(ep['episode_id'], [])
        values = {a[TARGET] for _, a in candidates}
        if ep['rating_status'] == 'conflicting_attempt_rows' or len(values) > 1:
            for _, audit in candidates:
                audit['status'] = 'conflicting_episode_ratings'
            continue
        if candidates:
            ep.update(difficulty_rating=candidates[0][1][TARGET], rating_status='matched',
                      rated_attempt_id=candidates[0][0]['attempt_id'],
                      rating_source_rows=sorted({f'{a["source_file"]}:{a["source_row"]}' for _, a in candidates}))
    audit = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    audit[TARGET] = pd.array(audit[TARGET], dtype='Int64')
    return rows, audit


def episode_frame(episodes):
    rows = []
    for ep in episodes:
        row = ep.copy()
        for name in ('intervals', 'attempt_ids', 'invalid_reasons', 'rating_source_rows'):
            row[name] = json.dumps(row[name], ensure_ascii=False, separators=(',', ':'))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(['participant_id', 'start_ms', 'episode_id']).reset_index(drop=True)
    frame[TARGET] = pd.array(frame[TARGET], dtype='Int64')
    return frame


def project(frame, episodes, config, validation):
    index = traces.WindowIndex(episodes)
    accepted, mappings, excluded = [], [], []
    for row in frame.to_dict('records'):
        ep, reason = index.match(row['session_id'], shared.epoch_ms(row['window_start']), shared.epoch_ms(row['window_end']))
        if ep:
            if ep['participant_id'] != row['participant_id']:
                raise ValueError('Window/episode participant mismatch.')
            if not ep['valid']:
                reason = 'invalid_event_trace'
            elif ep[TARGET] is None:
                reason = ep['rating_status']
            else:
                segments = [s for s in json.loads(row['window_phase_mix']) if s['duration_seconds'] > 0]
                if not segments or any(s['study_phase'] != 'active_task' for s in segments):
                    reason = 'source_phase_mismatch'
                elif any(s.get('puzzle_set') != ep['puzzle_set'] or s.get('puzzle_id') != ep['puzzle_id'] for s in segments):
                    reason = 'source_puzzle_mismatch'
        mapping = {'window_id': row['window_id'], 'participant_id': row['participant_id'],
                   'session_id': row['session_id'], 'episode_id': ep['episode_id'] if ep else '',
                   'window_start': row['window_start'], 'window_end': row['window_end'],
                   TARGET: ep[TARGET] if ep and not reason else None, 'quality_tier': row['quality_tier'],
                   'reason': reason or 'matched_participant_rating', 'source_v2_label': row.get('cognitive_load_label', '')}
        if reason:
            excluded.append(mapping)
        else:
            accepted.append(dict(row, difficulty_rating=int(ep[TARGET])))
            mappings.append(mapping)
    columns = [*frame.columns, TARGET]
    labeled = pd.DataFrame(accepted, columns=columns).drop(columns=['cognitive_load_label', 'window_label_mix'], errors='ignore')
    original_tiers = labeled.quality_tier.tolist()
    labeled = features.apply_policy(labeled, config)
    if labeled.quality_tier.tolist() != original_tiers:
        raise ValueError('V4 must preserve source sensor quality.')
    labeled[TARGET] = labeled[TARGET].astype('int64')
    views = {name: labeled.copy() if name == 'all_data' else features.curated_projection(
        labeled.loc[labeled.quality_tier.isin(tiers)].copy(), config, validation) for name, tiers in shared.DATASETS.items()}
    mapping = pd.DataFrame(mappings, columns=MAP_COLUMNS)
    rejected = pd.DataFrame(excluded, columns=MAP_COLUMNS)
    for table in (mapping, rejected):
        table[TARGET] = pd.array(table[TARGET], dtype='Int64')
    return views, mapping, rejected
