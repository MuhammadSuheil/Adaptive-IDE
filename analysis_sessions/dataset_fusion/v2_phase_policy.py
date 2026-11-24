"""Dataset-only task-context labels and deterministic balanced selection."""

import json
from bisect import bisect_right
from itertools import pairwise

import numpy as np

from fuse_and_analyze import context_at

LABEL_POLICY = 'session_phase_context_v3'
CURATION_POLICY = 'balanced_primary_v1'
GRACE_MS = 45000
LABELS = ('Low', 'Medium', 'High')
ACTIVE_LABELS = {'easy': 'Low', 'practice': 'Low', 'demo': 'Low', 'medium': 'Medium', 'hard': 'High'}
OPTIONAL = ('feedback', 'rating', 'rest', 'post_task')
BASE = ('active_task', 'pre_task')


def label_contract():
    return {'version': LABEL_POLICY, 'active_task': dict(ACTIVE_LABELS), 'pre_task': 'Low',
            'optional_phases': list(OPTIONAL), 'optional_target': 'preceding_actual_task_difficulty',
            'rest_post_task_limit_seconds': GRACE_MS/1000,
            'limit_origin': 'last_active_task_end_without_reset_on_phase_change',
            'excluded_phases': ['interrupted', 'baseline', 'unlabeled'], 'window_rule': 'unanimous_target'}


class PhaseTimeline:
    """Keep source phases; recover the task context stripped from rest/post-task."""

    def __init__(self, events, ready_ms):
        self.events = sorted(events, key=lambda e: e['event_time_ms'])
        self.ready_ms = ready_ms
        self.times, self.states = [], []
        state = dict(set_started_ms=None, puzzle_started_ms=None, active_ended_ms=None,
                     difficulty_level='', puzzle_set='', puzzle_id='', puzzle_title='',
                     active=False, closed=False, consistent=True)
        for event in self.events:
            state = state.copy()
            time, kind = event['event_time_ms'], event['event_type']
            if kind == 'set_started':
                state = dict(set_started_ms=time, puzzle_started_ms=None, active_ended_ms=None,
                             difficulty_level=event.get('difficulty_level', '').lower(),
                             puzzle_set=event.get('puzzle_set', ''), puzzle_id='', puzzle_title='',
                             active=False, closed=False, consistent=True)
            elif kind == 'puzzle_started' and state['set_started_ms'] is not None and not state['closed']:
                stated = event.get('difficulty_level', '').lower()
                state.update(puzzle_started_ms=time, active_ended_ms=None, active=True,
                             puzzle_id=event.get('puzzle_id', ''), puzzle_title=event.get('puzzle_title', ''))
                state['consistent'] &= not stated or stated == state['difficulty_level']
            elif kind in ('attempt_submitted', 'rating_started', 'rating_ended', 'set_ended', 'study_ended'):
                if state['active']:
                    state.update(active_ended_ms=time, active=False)
                if kind in ('set_ended', 'study_ended'):
                    state['closed'] = True
            self.times.append(time)
            self.states.append(state)
        boundaries = set(self.times)
        if ready_ms is not None:
            boundaries.add(ready_ms)
        boundaries.update(s['active_ended_ms']+GRACE_MS for s in self.states if s['active_ended_ms'] is not None)
        self.boundaries = sorted(boundaries)
        self.cache = {}

    def _context(self, time):
        context = context_at(time, self.events, self.ready_ms)
        phase = context['study_phase']
        index = bisect_right(self.times, time)-1
        state = self.states[index] if index >= 0 else {}
        source = {key: state.get(key) for key in ('set_started_ms', 'puzzle_started_ms', 'active_ended_ms')}
        reason = ''
        if phase in ('active_task', *OPTIONAL):
            valid = (state.get('puzzle_started_ms') is not None and state.get('consistent')
                     and state.get('difficulty_level') in ACTIVE_LABELS)
            if not valid:
                reason = 'missing_task_context'
            else:
                for key in ('difficulty_level', 'puzzle_set', 'puzzle_id', 'puzzle_title'):
                    context[key] = state[key]
                if phase in ('rest', 'post_task'):
                    stop = state.get('active_ended_ms')
                    source['grace_end_ms'] = stop+GRACE_MS if stop is not None else None
                    if stop is None or time > stop+GRACE_MS:
                        reason = 'outside_optional_grace'
        elif phase != 'pre_task':
            reason = 'interrupted' if phase == 'interrupted' else 'unresolved_or_calibration'
        source.update(eligible=not bool(reason), exclusion_reason=reason)
        context['label_context'] = source
        return context

    def segments(self, start, end):
        if end <= start or not self.events:
            return []
        lo, hi = bisect_right(self.boundaries, start), bisect_right(self.boundaries, end)
        bounds = [start, *(t for t in self.boundaries[lo:hi] if t < end), end]
        parts = []
        for left, right in pairwise(bounds):
            mid = (left+right)/2
            index = bisect_right(self.boundaries, mid)
            if index not in self.cache:
                self.cache[index] = self._context(mid)
            parts.append({**self.cache[index], 'duration_seconds': (right-left)/1000,
                          'proportion': (right-left)/(end-start)})
        return parts


def segment_label(segment):
    if not segment.get('label_context', {}).get('eligible', False):
        return None
    phase = segment.get('study_phase')
    if phase == 'pre_task':
        return 'Low'
    if phase in ('active_task', *OPTIONAL):
        return ACTIVE_LABELS.get(segment.get('difficulty_level', '').lower())
    return None


def label_window(segments):
    totals, failures = {}, []
    for segment in segments:
        if segment['duration_seconds'] <= 0:
            continue
        label = segment_label(segment) or 'unresolved'
        totals[label] = totals.get(label, 0.)+segment['duration_seconds']
        if label == 'unresolved':
            failures.append(segment.get('label_context', {}).get('exclusion_reason') or 'unresolved_or_calibration')
    if not totals or failures:
        label, reason = '', failures[0] if failures else 'unresolved_or_calibration'
    elif len(totals) != 1:
        label, reason = '', 'mixed_cognitive_labels'
    else:
        label, reason = next(iter(totals)), ''
    return {'cognitive_load_label': label, 'window_label_mix': json.dumps(
        {k: round(v, 6) for k, v in totals.items()}, separators=(',', ':')), 'exclusion_reason': reason}


def _allocate(capacities, count):
    """Capacity-limited equal quotas, with lexical round-robin remainder."""
    quota = {key: 0 for key in sorted(capacities)}
    while count:
        for key in quota:
            if quota[key] < capacities[key]:
                quota[key] += 1
                count -= 1
                if not count:
                    break
    return quota


def _take(pool, count):
    chosen = []
    for tier in ('good', 'acceptable'):
        remaining = count-len(chosen)
        if not remaining:
            break
        tier_rows = pool[pool.quality_tier.eq(tier)]
        n = min(remaining, len(tier_rows))
        people = dict(tuple(tier_rows.groupby('participant_id', sort=True)))
        quotas = _allocate({key: len(rows) for key, rows in people.items()}, n)
        for person, quota in quotas.items():
            sessions = dict(tuple(people[person].groupby('session_id', sort=True)))
            per_session = _allocate({key: len(rows) for key, rows in sessions.items()}, quota)
            for session, k in per_session.items():
                rows = sessions[session].sort_values(['window_start', 'window_id'])
                indices = np.floor((np.arange(k)+.5)*len(rows)/k).astype(int) if k else []
                chosen.extend(rows.iloc[indices].window_id.tolist())
    return chosen


def curate(frame):
    """Keep all labeled rows in the pool; select equal primary class counts."""
    if frame.window_id.duplicated().any():
        raise ValueError('Duplicate window IDs during phase curation.')
    result = frame.copy()
    quality_ok = frame.quality_tier.isin(('good', 'acceptable'))
    base = frame.window_phase_mix.map(lambda value: all(
        part['study_phase'] in BASE for part in json.loads(value) if part['duration_seconds'] > 0))
    basis = {label: int((quality_ok & base & frame.cognitive_load_label.eq(label)).sum()) for label in LABELS}
    total = {label: int((quality_ok & frame.cognitive_load_label.eq(label)).sum()) for label in LABELS}
    target = min(max(basis.values()), min(total.values()))
    if target <= 0:
        raise ValueError('Each class needs eligible windows and the dataset needs active/pre-task basis windows.')
    selected = set()
    summary = {'target_per_class': target, 'classes': {}}
    for label in LABELS:
        mask = quality_ok & frame.cognitive_load_label.eq(label)
        take_base = _take(frame[mask & base], min(basis[label], target))
        take_optional = _take(frame[mask & ~base], target-len(take_base))
        selected.update([*take_base, *take_optional])
        summary['classes'][label] = {'base_available': basis[label], 'optional_available': total[label]-basis[label],
            'base_selected': len(take_base), 'optional_selected': len(take_optional),
            'quality_eligible_not_selected': total[label]-target}
    result['curation_selected'] = result.window_id.isin(selected)
    result['curation_reason'] = np.where(~quality_ok, result.acceptable_failed_criteria,
        np.where(~result.curation_selected, 'balance_not_selected', np.where(base, 'basis_selected', 'optional_selected')))
    return result, summary
