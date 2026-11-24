"""Reference repairs, coverage selection, and Train-only sampling contracts."""

import copy
import importlib
import unittest
from collections import Counter, defaultdict

import pandas as pd

from . import v3_behavior as behavior
from .test_dataset_v3 import episode


class RevisionTests(unittest.TestCase):
    def curation(self):
        try:
            return importlib.import_module('analysis_sessions.dataset_fusion.v3_curation')
        except ModuleNotFoundError:
            self.fail('V3 reference repairs and Train sampling are not implemented.')

    def test_verified_alias_merges_only_listed_item_identities(self):
        c = self.curation()
        registry = c.load_aliases()
        old = episode('a', 10, puzzle_set='easy-2', puzzle_id='array_sum',
                      title_key='jumlah elemen array')
        new = dict(old, participant_id='b', puzzle_set='easy-1')
        # Use the independently inspected title from the actual proof registry.
        title = next(x['title_key'] for x in registry['items'] if x['puzzle_id'] == 'array_sum')
        old['title_key'] = new['title_key'] = title
        other = dict(new, participant_id='c', title_key='different question')
        prepared = c.prepare_episodes([old, new, other], registry, {})
        self.assertEqual(prepared[0]['item_key'], prepared[1]['item_key'])
        self.assertNotEqual(prepared[1]['item_key'], prepared[2]['item_key'])
        self.assertEqual(old['item_key'], '["set","q","variabel"]')

    def test_alias_proof_rejects_modified_content(self):
        c = self.curation()
        registry = copy.deepcopy(c.load_aliases())
        registry['items'][0]['content']['title'] = 'Altered'
        with self.assertRaisesRegex(ValueError, 'hash'):
            c.validate_aliases(registry)

    def test_known_prior_exposure_never_contributes_reference_but_remains_valid(self):
        c = self.curation()
        rows = [episode('p', 20, session_folder='repeated', puzzle_set='easy-1'),
                episode('q', 40)]
        prepared = c.prepare_episodes(rows, {'items': []}, {'repeated': ['easy']})
        self.assertTrue(prepared[0]['valid'])
        self.assertFalse(prepared[0]['reference_eligible'])
        ref = behavior.make_reference(prepared, {'p': 'Train', 'q': 'Train'})
        self.assertEqual([r['participant_id'] for r in ref['items'][rows[1]['item_key']]], ['q'])

    def test_train_weights_equalize_classes_people_and_episodes(self):
        c = self.curation()
        data = pd.DataFrame([
            {'window_id': f'w{i}', 'participant_id': p, 'cognitive_load_label': label,
                 'quality_tier': 'good', 'window_start': f'2026-01-01T00:00:{i:02}+00:00'}
            for i, (p, label) in enumerate([
                ('a', 'Low'), ('a', 'Low'), ('a', 'Low'), ('b', 'Low'),
                ('a', 'Medium'), ('c', 'Medium'), ('d', 'High'), ('d', 'High'),
                ('held', 'High')])])
        mapping = pd.DataFrame([
            {'window_id': f'w{i}', 'episode_id': e, 'split': 'Test' if i == 8 else 'Train'}
            for i, e in enumerate(['a1', 'a1', 'a2', 'b1', 'a3', 'c1', 'd1', 'd1', 'held1'])])
        weights, balanced = c.training_views(data, mapping)
        self.assertEqual(len(weights), 8)
        self.assertNotIn('w8', set(balanced.window_id))
        self.assertEqual(Counter(balanced.cognitive_load_label), {'Low': 2, 'Medium': 2, 'High': 2})
        self.assertEqual(len(set(balanced.window_id)), 6)
        totals = defaultdict(float)
        for r in weights.to_dict('records'):
            totals[r['cognitive_load_label']] += r['sample_weight']
        for total in totals.values():
            self.assertAlmostEqual(total, 8 / 3)
        low = weights.loc[weights.cognitive_load_label.eq('Low')]
        self.assertAlmostEqual(low.loc[low.participant_id.eq('a'), 'sample_weight'].sum(), 4 / 3)
        self.assertAlmostEqual(low.loc[low.episode_id.eq('a1'), 'sample_weight'].sum(), 2 / 3)
        self.assertAlmostEqual(low.loc[low.episode_id.eq('a2'), 'sample_weight'].sum(), 2 / 3)
        self.assertEqual(c.training_views(data.iloc[::-1], mapping.iloc[::-1])[1].window_id.tolist(),
                         balanced.window_id.tolist())

    def test_missing_train_class_produces_no_fake_balanced_training_data(self):
        c = self.curation()
        data = pd.DataFrame([{'window_id': 'w', 'participant_id': 'a', 'cognitive_load_label': 'High',
                                  'quality_tier': 'good', 'window_start': '2026-01-01T00:00:00Z'}])
        mapping = pd.DataFrame([{'window_id': 'w', 'episode_id': 'a', 'split': 'Train'}])
        weights, balanced = c.training_views(data, mapping)
        self.assertTrue(weights.empty)
        self.assertTrue(balanced.empty)

    def test_coverage_search_is_grouped_deterministic_and_reports_failure(self):
        c = self.curation()
        rows = [episode(f'p{i}', 10 + i) for i in range(8)]
        support = {w: Counter({r['episode_id']: 1 for r in rows}) for w in (15, 25, 45)}
        a, audit = c.select_partition(rows, support, max_candidates=8)
        other, same = c.select_partition(list(reversed(rows)), support, max_candidates=8)
        self.assertEqual(a, other)
        self.assertEqual(audit, same)
        self.assertEqual(audit['selected_tier'], 'unavailable')
        self.assertEqual(Counter(a.values()), {'Train': 6, 'validation': 1, 'Test': 1})

    def test_single_donor_sensitivity_does_not_relabel_or_impute(self):
        c = self.curation()
        donors = [episode(f'd{i}', t) for i, t in enumerate((10, 20, 30, 40, 50))]
        query = episode('query', 10)
        ref = behavior.make_reference(donors, {r['participant_id']: 'Train' for r in donors})
        query.update(behavior.label_episode(query, ref))
        result = c.reference_sensitivity([query], ref)
        self.assertEqual(result.iloc[0]['abstained_removals'], 5)
        self.assertEqual(result.iloc[0]['changed_label_removals'], 0)
        self.assertEqual(query['cognitive_load_label'], 'Low')

    def test_coverage_tiers_require_people_not_just_many_windows(self):
        c = self.curation()
        stats = {str(w): {'minimum_people_met': True, 'all_classes_present': True}
                 for w in (15, 25, 45)}
        self.assertEqual(c.coverage_tier(stats), 'A')
        stats['45']['minimum_people_met'] = False
        self.assertEqual(c.coverage_tier(stats), 'B')
        stats['45']['all_classes_present'] = False
        self.assertEqual(c.coverage_tier(stats), 'C')
        stats['25']['all_classes_present'] = False
        self.assertEqual(c.coverage_tier(stats), 'D')
        stats['15']['minimum_people_met'] = False
        self.assertEqual(c.coverage_tier(stats), 'unavailable')

    def test_current_builder_exports_sampling_and_keeps_legacy_policy_verifiable(self):
        from . import build_dataset_v3 as builder
        self.assertNotEqual(builder.policy_contract()['version'],
                            builder.policy_contract('behavioral_concordance_v1')['version'])
        legacy = builder.policy_contract('behavioral_concordance_v1')
        self.assertEqual(legacy['reference_item'], ['puzzle_set', 'puzzle_id', 'normalized_title'])
        current = builder.policy_contract()
        self.assertEqual(current['sampling_policy'], 'train_class_person_episode_v1')


if __name__ == '__main__':
    unittest.main()
