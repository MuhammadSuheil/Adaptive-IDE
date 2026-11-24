"""Context inheritance, exact phase boundaries, and balanced primary datasets."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def event(time, kind, **fields):
    return dict(event_time_ms=time, event_type=kind, **fields)


class PhasePolicyTests(unittest.TestCase):
    def setUp(self):
        try:
            self.policy = importlib.import_module('analysis_sessions.dataset_fusion.v2_phase_policy')
        except ModuleNotFoundError:
            self.fail('Context-aware phase labeling is not implemented.')

    def timeline(self, difficulty='medium', final='study_ended'):
        events = [event(0, 'set_started', difficulty_level=difficulty, puzzle_set=difficulty),
                  event(1, 'puzzle_started', puzzle_id='a', puzzle_title='A'),
                  event(10000, 'attempt_submitted'), event(11000, 'rating_started'),
                  event(20000, 'rating_ended'), event(25000, 'set_ended')]
        if final:
            events.append(event(100000, final, difficulty_level='hard', puzzle_set='hard'))
        return self.policy.PhaseTimeline(events, -1000)

    def test_feedback_and_rating_inherit_task_difficulty(self):
        for difficulty, expected in [('easy', 'Low'), ('medium', 'Medium'), ('hard', 'High')]:
            with self.subTest(difficulty=difficulty):
                parts = self.timeline(difficulty).segments(10000, 21000)
                self.assertEqual(self.policy.label_window(parts)['cognitive_load_label'], expected)
                self.assertEqual({p['puzzle_id'] for p in parts}, {'a'})

    def test_posttask_follows_last_actual_task_not_forced_high(self):
        parts = self.timeline('easy').segments(25000, 35000)
        self.assertEqual(parts[0]['study_phase'], 'post_task')
        self.assertEqual(parts[0]['difficulty_level'], 'easy')
        self.assertEqual(self.policy.label_window(parts)['cognitive_load_label'], 'Low')

    def test_rest_inherits_prior_set_and_does_not_cross_next_set(self):
        timeline = self.timeline(final='set_started')
        parts = timeline.segments(30000, 40000)
        self.assertEqual(parts[0]['study_phase'], 'rest')
        self.assertEqual(self.policy.label_window(parts)['cognitive_load_label'], 'Medium')
        self.assertEqual(self.policy.label_window(timeline.segments(99999, 100001))['cognitive_load_label'], '')

    def test_rest_posttask_deadline_does_not_reset_at_phase_changes(self):
        timeline = self.timeline()
        self.assertEqual(self.policy.label_window(timeline.segments(45000, 55000))['cognitive_load_label'], 'Medium')
        parts = timeline.segments(50000, 55000.001)
        self.assertEqual(self.policy.label_window(parts)['exclusion_reason'], 'outside_optional_grace')
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]['duration_seconds'], 5.)

    def test_feedback_rating_are_not_capped_by_rest_deadline(self):
        events = [event(0, 'set_started', difficulty_level='hard'), event(1, 'puzzle_started'),
                  event(10000, 'attempt_submitted'), event(20000, 'rating_started')]
        parts = self.policy.PhaseTimeline(events, -1).segments(70000, 90000)
        self.assertEqual(self.policy.label_window(parts)['cognitive_load_label'], 'High')

    def test_manual_end_stops_current_active_interval(self):
        events = [event(0, 'set_started', difficulty_level='hard'), event(1, 'puzzle_started'),
                  event(30000, 'set_ended'), event(100000, 'study_ended')]
        timeline = self.policy.PhaseTimeline(events, -1)
        self.assertEqual(self.policy.label_window(timeline.segments(60000, 75000))['cognitive_load_label'], 'High')
        self.assertEqual(self.policy.label_window(timeline.segments(75000, 75001))['cognitive_load_label'], '')

    def test_retry_resets_task_stop_but_not_same_episode_phase_changes(self):
        events = [event(0, 'set_started', difficulty_level='easy'), event(1, 'puzzle_started'),
                  event(10000, 'attempt_submitted'), event(20000, 'puzzle_started'),
                  event(80000, 'attempt_submitted'), event(81000, 'set_ended'), event(200000, 'study_ended')]
        timeline = self.policy.PhaseTimeline(events, -1)
        self.assertEqual(self.policy.label_window(timeline.segments(110000, 125000))['cognitive_load_label'], 'Low')
        self.assertEqual(self.policy.label_window(timeline.segments(125000, 130000))['cognitive_load_label'], '')

    def test_interrupted_and_task_without_traceable_start_are_excluded(self):
        self.assertEqual(self.policy.label_window(self.timeline(final=None).segments(30000, 40000))['exclusion_reason'], 'interrupted')
        events = [event(0, 'set_started', difficulty_level='medium'), event(50000, 'set_ended'), event(60000, 'study_ended')]
        self.assertEqual(self.policy.label_window(self.policy.PhaseTimeline(events, -1).segments(10, 20))['cognitive_load_label'], '')

    def test_readiness_and_event_endpoint_are_exact(self):
        events = [event(0, 'set_started', difficulty_level='medium'), event(1, 'puzzle_started'),
                  event(25000, 'set_started', difficulty_level='hard'), event(25000, 'puzzle_started')]
        timeline = self.policy.PhaseTimeline(events, -1000)
        self.assertEqual(self.policy.label_window(timeline.segments(-1000, -1))['cognitive_load_label'], 'Low')
        self.assertEqual(self.policy.label_window(timeline.segments(-1001, -1))['cognitive_load_label'], '')
        self.assertEqual(self.policy.label_window(timeline.segments(1, 25000))['cognitive_load_label'], 'Medium')
        self.assertEqual(self.policy.label_window(timeline.segments(1, 25000.001))['exclusion_reason'], 'mixed_cognitive_labels')

    def frame(self, specs):
        rows = []
        for label, base, optional in specs:
            for phase, count in [('active_task', base), ('rating', optional)]:
                for i in range(count):
                    rows.append(dict(window_id=f'{label}-{phase}-{i}', cognitive_load_label=label,
                        quality_tier='good', acceptable_failed_criteria='',
                        participant_id=f'p{i % 2}', session_id=f's{i % 2}', window_start=f'{i:06}',
                        window_phase_mix=json.dumps([dict(study_phase=phase, duration_seconds=15)])))
        return pd.DataFrame(rows)

    def test_balance_uses_base_first_and_only_needed_optional_rows(self):
        frame = self.frame([('Low', 2, 8), ('Medium', 5, 8), ('High', 6, 8)])
        result, audit = self.policy.curate(frame)
        selected = result[result.curation_selected]
        self.assertEqual(selected.cognitive_load_label.value_counts().to_dict(), {'Low': 6, 'Medium': 6, 'High': 6})
        self.assertEqual(audit['target_per_class'], 6)
        self.assertEqual(sum('active_task' in v for v in selected.window_phase_mix), 13)
        self.assertEqual(len(result), len(frame))

    def test_balance_can_downsample_majority_without_changing_measurements(self):
        frame = self.frame([('Low', 2, 1), ('Medium', 5, 1), ('High', 6, 1)])
        result, audit = self.policy.curate(frame)
        self.assertEqual(audit['target_per_class'], 3)
        self.assertEqual(result[result.curation_selected].cognitive_load_label.value_counts().to_dict(), {'Low': 3, 'Medium': 3, 'High': 3})
        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertIn('balance_not_selected', set(result.curation_reason))

    def test_selection_is_stable_under_input_shuffle_and_prioritizes_quality(self):
        frame = self.frame([('Low', 2, 0), ('Medium', 8, 0), ('High', 8, 0)])
        frame.loc[frame.window_id.eq('Medium-active_task-0'), 'quality_tier'] = 'acceptable'
        frame.loc[frame.window_id.eq('High-active_task-0'), ['quality_tier', 'acceptable_failed_criteria']] = ['poor', 'hrv_unusable']
        a, _ = self.policy.curate(frame)
        b, _ = self.policy.curate(frame.sample(frac=1, random_state=91))
        self.assertEqual(set(a.loc[a.curation_selected, 'window_id']), set(b.loc[b.curation_selected, 'window_id']))
        self.assertFalse(a.loc[a.window_id.eq('Medium-active_task-0'), 'curation_selected'].iloc[0])
        self.assertFalse(a.loc[a.window_id.eq('High-active_task-0'), 'curation_selected'].iloc[0])
        for label in ('Medium', 'High'):
            self.assertEqual(a[a.curation_selected & a.cognitive_load_label.eq(label)].participant_id.nunique(), 2)

    def test_absent_class_aborts_instead_of_publishing_empty_balance(self):
        with self.assertRaisesRegex(ValueError, 'class'):
            self.policy.curate(self.frame([('Low', 3, 0), ('Medium', 3, 0)]))

    def test_complete_migration_keeps_sensor_values_and_inherits_policy(self):
        import numpy as np
        from datetime import timedelta
        from analysis_sessions.dataset_fusion import build_dataset as shared
        from analysis_sessions.dataset_fusion import build_dataset_v2 as builder
        from analysis_sessions.dataset_fusion.test_dataset_v2 import ExportTests, raw_rows
        from analysis_sessions.dataset_fusion.test_dataset_fusion import BASE, eye_source

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = ExportTests().source(root/'sources')
            origin = pd.Timestamp(BASE, unit='ms', tz='UTC').to_pydatetime()
            eye = eye_source(np.arange(0, 360001, 50))
            eye['session_id'] = 'source-Andrean'
            eye.to_csv(directory/'eye_tracking.csv', index=False)
            rows = raw_rows(360)
            for i, row in enumerate(rows, 1):
                row.update(received_at=(origin+timedelta(seconds=i)).isoformat(), session_id='source-Andrean')
            pd.DataFrame(rows).to_csv(directory/'rr_raw.csv', index=False)
            events = []
            for start, level in [(0, 'easy'), (120000, 'medium'), (240000, 'hard')]:
                events.extend([event(BASE+start, 'set_started', difficulty_level=level, puzzle_set=level),
                               event(BASE+start, 'puzzle_started', puzzle_id=level),
                               event(BASE+start+80000, 'attempt_submitted'),
                               event(BASE+start+81000, 'rating_started'),
                               event(BASE+start+90000, 'rating_ended'),
                               event(BASE+start+95000, 'set_ended')])
            events.append(event(BASE+350000, 'study_ended'))
            pd.DataFrame(events).assign(session_id='source-Andrean').to_csv(directory/'task_events.csv', index=False)
            before_sources = shared.snapshot(root/'sources')
            old = builder.build_dataset(root/'sources', root/'bundle', admission_policy='feature_aware_quality_v1')
            old_frame = pd.read_csv(root/'bundle/window_25s_step_5s/all_data.csv').set_index('window_id')
            try:
                migrated = builder.build_dataset(root/'sources', root/'bundle', replace=True,
                                                label_policy=self.policy.LABEL_POLICY)
            except TypeError as error:
                self.fail(f'Builder does not support the approved label migration: {error}')
            self.assertEqual(migrated['label_policy'], self.policy.label_contract())
            self.assertEqual(migrated['curation_policy'], 'balanced_primary_v1')
            for seconds in (15, 25, 45):
                path = root/f'bundle/window_{seconds}s_step_5s'
                selected = pd.read_csv(path/'good_acceptable.csv')
                counts = selected.cognitive_load_label.value_counts()
                self.assertEqual(len(counts), 3)
                self.assertEqual(counts.nunique(), 1)
                self.assertTrue(set(pd.read_csv(path/'good_only.csv').window_id).issubset(selected.window_id))
            new_frame = pd.read_csv(root/'bundle/window_25s_step_5s/all_data.csv').set_index('window_id')
            common = old_frame.index.intersection(new_frame.index)
            pd.testing.assert_frame_equal(old_frame.loc[common, shared.FEATURE_COLUMNS],
                                          new_frame.loc[common, shared.FEATURE_COLUMNS])
            self.assertTrue((old_frame.loc[common, 'cognitive_load_label'] != new_frame.loc[common, 'cognitive_load_label']).any())
            self.assertEqual(shared.snapshot(root/'sources'), before_sources)
            self.assertTrue((root/'bundle/history/pre_session_phase_context_v3_manifest.json').exists())
            self.assertEqual(builder.verify_bundle(root/'bundle'), migrated)
            rebuilt = builder.build_dataset(root/'sources', root/'bundle', replace=True)
            self.assertEqual(rebuilt['label_policy'], migrated['label_policy'])
            self.assertEqual(rebuilt['datasets'], migrated['datasets'])
            from analysis_sessions.dataset_fusion.v2_rr_quality import refresh_bundle
            refreshed = refresh_bundle(root/'bundle')
            self.assertEqual(refreshed['label_policy'], migrated['label_policy'])
            self.assertEqual(refreshed['datasets'], migrated['datasets'])
            self.assertEqual(old['label_policy']['Low_phases'], ['pre_task','feedback','rating','rest','post_task','interrupted'])


if __name__ == '__main__':
    unittest.main()
