"""Evidence-pinned item identities, reference eligibility, and Train sampling."""

import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import v3_behavior as behavior

WINDOWS = (15, 25, 45)
SPLITS = ('Train', 'validation', 'Test')
SEED = 20261124
MAX_CANDIDATES = 1000
WEIGHT_COLUMNS = ['window_id', 'participant_id', 'episode_id', 'cognitive_load_label', 'sample_weight']


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_aliases(registry):
    identities = set()
    for item in registry['items']:
        digest = content_hash(item['content'])
        if item['canonical_key'] != 'sha256:' + digest:
            raise ValueError('Alias content hash mismatch.')
        if item['title_key'] != behavior.normalize_title(item['content']['title']):
            raise ValueError('Alias title mismatch.')
        if item['puzzle_id'] != item['content']['id'] or len(item['evidence']) != 2:
            raise ValueError('Incomplete alias proof.')
        for proof in item['evidence']:
            if proof['content_sha256'] != digest or not re.fullmatch('[0-9a-f]{40}', proof['commit']):
                raise ValueError('Invalid alias proof hash or commit.')
            path = Path(proof['path'])
            if path.parts[:2] != ('codearrange-web', 'puzzles') or len(path.parts) != 4:
                raise ValueError('Invalid alias proof path.')
            key = path.parts[2], item['puzzle_id'], item['title_key']
            if key in identities:
                raise ValueError('Duplicate alias identity.')
            identities.add(key)
    return registry


def load_aliases(path=None):
    path = Path(path) if path else Path(__file__).with_name('v3_item_aliases.json')
    return validate_aliases(json.loads(path.read_text(encoding='utf-8')))


def prepare_episodes(episodes, registry, repeated):
    """Preserve sensor/trace eligibility while excluding known exposure from norms."""
    aliases = {}
    for item in registry['items']:
        for proof in item['evidence']:
            aliases[(Path(proof['path']).parts[2], item['puzzle_id'], item['title_key'])] = item['canonical_key']
    prepared, seen = [], set()
    for original in sorted(episodes, key=lambda e: (e['start_ms'], e['episode_id'])):
        row = copy.deepcopy(original)
        row['source_item_key'] = row['item_key']
        row['item_key'] = aliases.get((row['puzzle_set'], row['puzzle_id'], row['title_key']), row['item_key'])
        identity = row['participant_id'], row['item_key']
        row['observed_first_exposure'] = identity not in seen
        seen.add(identity)
        difficulty = row['puzzle_set'].split('-')[0]
        row['known_prior_exposure'] = difficulty in repeated.get(row['session_folder'], [])
        row['first_exposure'] = row['observed_first_exposure'] and not row['known_prior_exposure']
        reason = ('known_prior_exposure' if row['known_prior_exposure'] else
                  'later_observed_exposure' if not row['observed_first_exposure'] else
                  'invalid_event_trace' if not row['valid'] else
                  'not_completed_manual' if not row['completed'] or row['success_submission_type'] != 'manual' else
                  'no_observed_active_time' if row['active_seconds'] <= 0 else '')
        row['reference_eligible'] = not reason
        row['reference_exclusion_reason'] = reason
        prepared.append(row)
    return prepared


def coverage(episodes, labels, splits, support):
    result = {}
    for seconds, count in support.items():
        people, rows, eps = defaultdict(set), Counter(), defaultdict(set)
        for e in episodes:
            label = labels[e['episode_id']]
            n = count.get(e['episode_id'], 0)
            if label and n:
                key = splits[e['participant_id']], label
                people[key].add(e['participant_id'])
                eps[key].add(e['episode_id'])
                rows[key] += n
        values = {s: {label: {'participants': len(people[(s, label)]),
                              'episodes': len(eps[(s, label)]), 'windows': rows[(s, label)]}
                      for label in behavior.LABELS} for s in SPLITS}
        result[str(seconds)] = {'counts': values,
            'all_classes_present': all(people[(s, label)] for s in SPLITS for label in behavior.LABELS),
            'minimum_people_met': all(len(people[(s, label)]) >= (3 if s == 'Train' else 2)
                                      for s in SPLITS for label in behavior.LABELS)}
        result[str(seconds)]['all_classes_present'] = bool(result[str(seconds)]['all_classes_present'])
    return result


def coverage_tier(stats):
    if not stats['15']['minimum_people_met']:
        return 'unavailable'
    if all(stats[str(w)]['minimum_people_met'] for w in WINDOWS):
        return 'A'
    if all(stats[str(w)]['all_classes_present'] for w in WINDOWS):
        return 'B'
    return 'C' if stats['25']['all_classes_present'] else 'D'


def select_partition(episodes, support, *, people=None, max_candidates=MAX_CANDIDATES):
    people = sorted(set(people if people is not None else (e['participant_id'] for e in episodes)))
    original = behavior.partition(people)
    held = max(1, round(.15 * len(people)))
    rng, first, trials, totals = random.Random(SEED), {}, [], Counter()
    if max_candidates < 1:
        raise ValueError('At least one partition candidate is required.')
    for i in range(max_candidates):
        if i == 0:
            splits = original
        else:
            shuffled = people.copy()
            rng.shuffle(shuffled)
            splits = {p: 'Train' if j < len(people) - 2 * held else
                         'validation' if j < len(people) - held else 'Test'
                      for j, p in enumerate(shuffled)}
        reference = behavior.make_reference(episodes, splits)
        labels = {e['episode_id']: behavior.label_episode(e, reference)['cognitive_load_label'] for e in episodes}
        stats = coverage(episodes, labels, splits, support)
        tier = coverage_tier(stats)
        totals[tier] += 1
        trials.append({'candidate_index': i, 'partition_sha256': content_hash(splits), 'tier': tier})
        if tier not in first:
            first[tier] = (i, splits, stats)
    chosen = next((tier for tier in ('A', 'B', 'C', 'D') if tier in first), 'unavailable')
    i, splits, stats = first[chosen]
    audit = {'version': 'bounded_coverage_v1', 'seed': SEED, 'maximum_candidates': max_candidates,
             'candidate_zero': 'legacy_hash_partition', 'selected_candidate_index': i,
             'selected_tier': chosen, 'tier_counts': dict(totals), 'coverage': stats,
             'minimum_people': {'Train': 3, 'validation': 2, 'Test': 2},
             'primary_window_seconds': 15, 'model_scores_used': False, 'candidates': trials}
    return splits, audit


def reference_sensitivity(episodes, reference):
    rows = []
    for e in episodes:
        label = e['cognitive_load_label']
        if not label:
            continue
        donors = [d for d in reference['items'].get(e['item_key'], []) if d['participant_id'] != e['participant_id']]
        changed, abstained = 0, 0
        for donor in donors:
            reduced = {**reference, 'items': dict(reference['items'])}
            reduced['items'][e['item_key']] = [d for d in reference['items'][e['item_key']]
                                              if d['participant_id'] != donor['participant_id']]
            other = behavior.label_episode(e, reduced)['cognitive_load_label']
            changed += bool(other and other != label)
            abstained += not other
        rows.append({'episode_id': e['episode_id'], 'participant_id': e['participant_id'],
                     'cognitive_load_label': label, 'reference_count': len(donors),
                     'changed_label_removals': changed, 'abstained_removals': abstained,
                     'unchanged_removals': len(donors) - changed - abstained})
    return pd.DataFrame(rows, columns=['episode_id', 'participant_id', 'cognitive_load_label',
                        'reference_count', 'changed_label_removals', 'abstained_removals', 'unchanged_removals'])


def _fair_quotas(capacities, target):
    quotas = dict.fromkeys(sorted(capacities), 0)
    if target > sum(capacities.values()) or target < 0:
        raise ValueError('Impossible sampling quota.')
    while target:
        for key in quotas:
            if target and quotas[key] < capacities[key]:
                quotas[key] += 1
                target -= 1
    return quotas


def _spread(rows, n):
    selected = []
    for tier in ('good', 'acceptable'):
        part = rows.loc[rows.quality_tier.eq(tier)].sort_values(['window_start', 'window_id'])
        remaining = n - len(selected)
        if remaining <= 0:
            break
        k = min(remaining, len(part))
        positions = np.floor((np.arange(k) + .5) * len(part) / k).astype(int) if k else []
        selected.extend(part.iloc[positions].window_id.tolist())
    if len(selected) != n:
        raise ValueError('Training sample contains non-curated quality tiers.')
    return selected


def training_views(data, mapping):
    assigned = mapping[['window_id', 'episode_id', 'split']]
    joined = data.merge(assigned, on='window_id', how='left', validate='one_to_one')
    if joined[['episode_id', 'split']].isna().any().any():
        raise ValueError('Missing training assignment.')
    train = joined.loc[joined.split.eq('Train')].copy()
    counts = train.cognitive_load_label.value_counts()
    if any(counts.get(label, 0) == 0 for label in behavior.LABELS):
        return pd.DataFrame(columns=WEIGHT_COLUMNS), data.iloc[:0].copy()
    weights, selected = [], []
    target = min(int(counts[label]) for label in behavior.LABELS)
    for label in behavior.LABELS:
        by_class = train.loc[train.cognitive_load_label.eq(label)]
        person_cap = by_class.groupby('participant_id').size().to_dict()
        person_quotas = _fair_quotas(person_cap, target)
        for person, person_data in by_class.groupby('participant_id', sort=True):
            ep_cap = person_data.groupby('episode_id').size().to_dict()
            ep_quotas = _fair_quotas(ep_cap, person_quotas[person])
            for eid, ep in person_data.groupby('episode_id', sort=True):
                weight = len(train) / (3 * len(person_cap) * len(ep_cap) * len(ep))
                for window_id in ep.window_id:
                    weights.append({'window_id': window_id, 'participant_id': person, 'episode_id': eid,
                                    'cognitive_load_label': label, 'sample_weight': weight})
                selected.extend(_spread(ep, ep_quotas[eid]))
    balanced = data.loc[data.window_id.isin(selected)].sort_values('window_id').reset_index(drop=True)
    if len(balanced) != 3 * target or not balanced.window_id.is_unique:
        raise ValueError('Invalid balanced Train selection.')
    return pd.DataFrame(weights, columns=WEIGHT_COLUMNS).sort_values('window_id').reset_index(drop=True), balanced
