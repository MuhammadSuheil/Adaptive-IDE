"""Exploratory behavioral proxies, independent of physiological measurements."""

import hashlib
import itertools
import json
import math
import re
import unicodedata
from bisect import bisect_right
from collections import defaultdict

import numpy as np

LEGACY_POLICY = 'behavioral_concordance_v1'
POLICY = 'behavioral_concordance_v2'
LABELS = ('Low', 'Medium', 'High')
MIN_REFERENCE = 5


def normalize_title(value):
    text = unicodedata.normalize('NFKC', str(value or '')).strip()
    return ' '.join(re.sub(r'^\d+\s*[.)-]\s*', '', text).casefold().split())


def partition(people):
    people = sorted(set(people), key=lambda p: hashlib.sha256(f'v3|42|{p}'.encode()).hexdigest())
    if len(people) < 3:
        raise ValueError('At least three people are required for disjoint partitions.')
    count = max(1, round(.15*len(people)))
    training = len(people)-2*count
    return {person: 'Train' if i < training else 'validation' if i < training+count else 'Test'
            for i, person in enumerate(people)}


def _clean_events(events, session_id):
    audit = {'duplicate_events': 0, 'duplicate_attempts': 0, 'conflicting_attempt_ids': [], 'issues': []}
    kept, event_ids, attempts = [], {}, {}
    fields = ('session_id', 'event_time_ms', 'event_type', 'puzzle_set', 'puzzle_id',
              'puzzle_title', 'attempt_id', 'submission_type', 'outcome')
    for original in sorted(events, key=lambda e: float(e['event_time_ms'])):
        e = dict(original, event_time_ms=float(original['event_time_ms']))
        if e.get('session_id') != session_id:
            raise ValueError('Event session identity mismatch.')
        if not math.isfinite(e['event_time_ms']) or e['event_time_ms'] < 0:
            raise ValueError('Invalid event time.')
        signature = tuple(e.get(key, '') for key in fields)
        event_id = e.get('event_id')
        if event_id and event_id in event_ids:
            previous, old_signature = event_ids[event_id]
            if signature != old_signature:
                previous['_trace_conflict'] = 'conflicting_event_id'
                audit['issues'].append({'event_id': event_id, 'reason': 'conflicting_event_id'})
            else:
                audit['duplicate_events'] += 1
            continue
        attempt = e.get('attempt_id') if e['event_type'] == 'attempt_submitted' else None
        if attempt and attempt in attempts:
            previous, old_signature = attempts[attempt]
            if signature != old_signature:
                previous['_trace_conflict'] = 'conflicting_attempt_id'
                audit['conflicting_attempt_ids'].append(attempt)
            else:
                audit['duplicate_attempts'] += 1
            continue
        kept.append(e)
        if event_id:
            event_ids[event_id] = e, signature
        if attempt:
            attempts[attempt] = e, signature
    audit['conflicting_attempt_ids'] = sorted(set(audit['conflicting_attempt_ids']))
    return kept, audit


def reconstruct(events, participant_id, session_id, session_folder):
    events, audit = _clean_events(events, session_id)
    rows, latest = [], {}
    current_set, set_key, active, set_number = None, '', None, 0

    def invalid(row, reason):
        if reason not in row['invalid_reasons']:
            row['invalid_reasons'].append(reason)
        row['valid'] = False

    def stop(time):
        nonlocal active
        if active is not None:
            row, start = active
            if time > start:
                row['intervals'].append([start, time])
                row['active_seconds'] += (time-start)/1000
            row['end_ms'] = time
            active = None

    def close_set(time, reason):
        stop(time)
        for row in latest.values():
            if not row['completed'] and not row['closed_by']:
                row.update(end_ms=time, closed_by=reason, censored=True)

    for e in events:
        kind, time = e['event_type'], e['event_time_ms']
        if kind == 'set_started':
            close_set(time, 'set_changed')
            set_number += 1
            current_set = f'{session_id}|{time}|{set_number}'
            set_key = e.get('puzzle_set', '')
            latest = {}
        elif kind == 'puzzle_started':
            stop(time)
            if current_set is None:
                audit['issues'].append({'event_id': e.get('event_id', ''), 'reason': 'puzzle_without_active_set'})
                continue
            puzzle_id = e.get('puzzle_id', '')
            previous = latest.get(puzzle_id)
            title = normalize_title(e.get('puzzle_title', ''))
            if previous is None or previous['completed'] or previous['closed_by']:
                cycle = sum(r['set_instance_id'] == current_set and r['puzzle_id'] == puzzle_id for r in rows)+1
                episode_id = hashlib.sha256(f'{current_set}|{puzzle_id}|{cycle}'.encode()).hexdigest()[:24]
                previous = {'episode_id': episode_id, 'participant_id': participant_id, 'session_id': session_id,
                    'session_folder': session_folder, 'set_instance_id': current_set, 'puzzle_set': set_key,
                    'puzzle_id': puzzle_id, 'puzzle_title': e.get('puzzle_title', ''), 'title_key': title,
                    'item_key': json.dumps([set_key, puzzle_id, title], ensure_ascii=False, separators=(',', ':')),
                    'start_ms': time, 'end_ms': None, 'active_seconds': 0., 'intervals': [], 'attempt_ids': [],
                    'manual_errors': 0, 'manual_submissions': 0, 'timeout_submissions': 0, 'completed': False,
                    'success_submission_type': '', 'closed_by': '', 'censored': False, 'valid': True, 'invalid_reasons': []}
                rows.append(previous)
                latest[puzzle_id] = previous
            elif previous['title_key'] != title:
                invalid(previous, 'conflicting_puzzle_title')
            if not puzzle_id or not title or not set_key or e.get('puzzle_set') != set_key:
                invalid(previous, 'missing_or_conflicting_puzzle_identity')
            if e.get('_trace_conflict'):
                invalid(previous, e['_trace_conflict'])
            active = previous, time
        elif kind == 'attempt_submitted':
            if active is None:
                candidate = latest.get(e.get('puzzle_id', ''))
                if candidate is not None:
                    invalid(candidate, 'submission_without_active_start')
                audit['issues'].append({'event_id': e.get('event_id', ''), 'attempt_id': e.get('attempt_id', ''),
                                            'reason': 'submission_without_active_start'})
                continue
            row = active[0]
            if e.get('puzzle_id') != row['puzzle_id'] or e.get('puzzle_set') != row['puzzle_set']:
                invalid(row, 'submission_identity_mismatch')
                stop(time)
                continue
            stop(time)
            if e.get('_trace_conflict'):
                invalid(row, e['_trace_conflict'])
            if not e.get('attempt_id'):
                invalid(row, 'missing_attempt_id')
            else:
                row['attempt_ids'].append(e['attempt_id'])
            mode, outcome = e.get('submission_type'), e.get('outcome')
            if mode not in ('manual', 'timeout') or outcome not in ('correct', 'incorrect'):
                invalid(row, 'unknown_submission_outcome')
            if mode == 'manual':
                row['manual_submissions'] += 1
                row['manual_errors'] += int(outcome == 'incorrect')
            elif mode == 'timeout':
                row['timeout_submissions'] += 1
            if outcome == 'correct':
                row.update(completed=True, success_submission_type=mode, closed_by='correct')
            elif mode == 'timeout':
                row.update(closed_by='timeout', censored=True)
        elif kind in ('rating_started', 'rating_ended'):
            if active is not None:
                invalid(active[0], 'rating_without_submission')
                stop(time)
        elif kind in ('set_ended', 'study_ended'):
            close_set(time, kind)
            current_set = None
    for row in rows:
        if not row['completed'] and not row['closed_by']:
            invalid(row, 'missing_episode_end')
        if row['active_seconds'] <= 0:
            invalid(row, 'no_observed_active_time')
    return rows, audit


def make_reference(episodes, splits, *, policy=POLICY):
    """One first-exposure observation per person/item; only successful Train cases."""
    seen, items = set(), defaultdict(list)
    for row in sorted(episodes, key=lambda r: (r['start_ms'], r['episode_id'])):
        identity = row['participant_id'], row['item_key']
        if identity in seen:
            continue
        seen.add(identity)
        if (splits[row['participant_id']] == 'Train' and row['valid'] and row['completed']
                and (policy == LEGACY_POLICY or row.get('reference_eligible', True))
                and row['success_submission_type'] == 'manual' and row['active_seconds'] > 0):
            items[row['item_key']].append({'participant_id': row['participant_id'],
                'episode_id': row['episode_id'], 'active_seconds': row['active_seconds']})
    return {'policy': policy, 'minimum_other_people': MIN_REFERENCE, 'quantiles': [1/3,2/3],
            'quantile_method': 'linear', 'source_split': 'Train', 'self_reference': False,
            'items': {key: sorted(values, key=lambda v: v['participant_id']) for key, values in sorted(items.items())}}


def reference_for(reference, episode):
    values = [v for v in reference['items'].get(episode['item_key'], [])
              if v['participant_id'] != episode['participant_id']]
    result = {'participants': [v['participant_id'] for v in values], 'count': len(values),
                  'lower_seconds': None, 'upper_seconds': None, 'reason': ''}
    if len(values) < MIN_REFERENCE:
        result['reason'] = 'insufficient_reference'
    else:
        lower, upper = np.quantile([v['active_seconds'] for v in values], [1/3,2/3], method='linear')
        result.update(lower_seconds=float(lower), upper_seconds=float(upper))
        if not 0 < lower < upper:
            result['reason'] = 'degenerate_reference'
    return result


def label_episode(episode, reference):
    limits = reference_for(reference, episode)
    result = {'cognitive_load_label': '', 'label_reason': '', 'time_band': '', 'performance_band': '',
                  'reference_count': limits['count'], 'reference_participants': limits['participants'],
                  'lower_seconds': limits['lower_seconds'], 'upper_seconds': limits['upper_seconds']}
    if not episode['valid']:
        result['label_reason'] = 'invalid_event_trace'
    elif limits['reason']:
        result['label_reason'] = limits['reason']
    else:
        seconds, errors = episode['active_seconds'], episode['manual_errors']
        time_band = ('Low' if seconds <= limits['lower_seconds'] else
                     'Medium' if seconds <= limits['upper_seconds'] else 'High')
        performance = ('Low' if episode['completed'] and errors == 0 and episode['manual_submissions'] == 1
                       and episode['success_submission_type'] == 'manual' else
                       'Medium' if episode['completed'] and 1 <= errors <= 2 else 'High' if errors >= 3 else '')
        result.update(time_band=time_band, performance_band=performance)
        # An unfinished duration is only a lower bound: never use it for Low/Medium.
        if performance == time_band and (episode['completed'] or time_band == 'High'):
            result.update(cognitive_load_label=time_band, label_reason='concordant_behavioral_evidence')
        else:
            result['label_reason'] = 'conflicting_or_insufficient_behavior'
    return result


class WindowIndex:
    """Match only continuous active time from one episode, without rounding gaps."""

    def __init__(self, episodes):
        self.sessions = defaultdict(list)
        for episode in episodes:
            merged = []
            for start, end in sorted(episode['intervals']):
                if end <= start:
                    raise ValueError('Invalid active interval.')
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start,end])
            self.sessions[episode['session_id']].extend((start,end,episode) for start,end in merged)
        self.starts = {}
        for session, intervals in self.sessions.items():
            intervals.sort(key=lambda r: r[0])
            if any(b[0] < a[1] for a,b in itertools.pairwise(intervals)):
                raise ValueError('Overlapping active episodes.')
            self.starts[session] = [r[0] for r in intervals]

    def match(self, session_id, start, end):
        if end <= start:
            raise ValueError('Invalid window interval.')
        i = bisect_right(self.starts.get(session_id, []), start)-1
        if i >= 0:
            _, stop, episode = self.sessions[session_id][i]
            if end <= stop:
                return episode, ''
        return None, 'not_single_active_episode'
