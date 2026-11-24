"""Participant ratings must be observed, matched, and confined to active episodes."""

import copy
import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from .test_dataset_v3 import episode, event


def rating_row(attempt='a', rating='4', **changes):
    row = {'timestamp': '1970-01-01T00:00:11.000Z', 'attempt_id': attempt,
           'name': 'P', 'puzzle_set': 'set', 'puzzle_id': 'q', 'puzzle_title': '1. Variabel',
           'submission_type': 'manual', 'correct': 'true', 'rating': rating, 'session_id': 's',
           'source_file': 'easy-1.csv', 'source_row': 2}
    row.update(changes)
    return row


class RatingTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('analysis_sessions.dataset_fusion.v4_ratings')
        except ModuleNotFoundError:
            self.fail('Observed-rating matching is not implemented.')

    def fixture(self):
        ep = episode('p', 10, session_id='s', attempt_ids=['a'])
        attempt = event(11000, 'attempt_submitted', attempt_id='a', submission_type='manual', outcome='correct')
        return ep, {'a': attempt}

    def test_manual_rating_is_not_replaced_by_difficulty_or_error_count(self):
        ep, attempts = self.fixture()
        ep.update(manual_errors=3, difficulty_level='hard')
        rows, audit = self.module().match_ratings([rating_row(rating='2')], attempts, [ep])
        self.assertEqual(rows[0]['difficulty_rating'], 2)
        self.assertEqual(rows[0]['rated_attempt_id'], 'a')
        self.assertEqual(audit.iloc[0]['status'], 'matched')
        self.assertNotIn('difficulty_rating', ep)

    def test_missing_rating_and_unrated_timeout_are_not_imputed(self):
        ep, attempts = self.fixture()
        ep.update(completed=False, closed_by='timeout', censored=True)
        attempts['a'].update(submission_type='timeout', outcome='incorrect')
        for value in ('', '-', None):
            with self.subTest(value=value):
                rows, _ = self.module().match_ratings([rating_row(rating=value)], attempts, [ep])
                self.assertIsNone(rows[0]['difficulty_rating'])
                self.assertEqual(rows[0]['rating_status'], 'missing_rating')

    def test_timeout_uses_actual_rating_instead_of_ten(self):
        ep, attempts = self.fixture()
        ep.update(completed=False, closed_by='timeout', censored=True)
        attempts['a'].update(submission_type='timeout', outcome='incorrect')
        rows, _ = self.module().match_ratings([
            rating_row(rating='6', submission_type='timeout', correct='false')], attempts, [ep])
        self.assertEqual(rows[0]['difficulty_rating'], 6)

    def test_invalid_ratings_do_not_become_training_targets(self):
        ep, attempts = self.fixture()
        for value in ('0', '11', '1.5', 'NaN', 'True', '1.0'):
            with self.subTest(value=value):
                rows, audit = self.module().match_ratings([rating_row(rating=value)], attempts, [ep])
                self.assertIsNone(rows[0]['difficulty_rating'])
                self.assertEqual(audit.iloc[0]['status'], 'invalid_rating')

    def test_identity_outcome_title_and_timestamp_mismatches_are_rejected(self):
        ep, attempts = self.fixture()
        for change in ({'session_id': 'other'}, {'puzzle_id': 'other'}, {'puzzle_set': 'other'},
                       {'puzzle_title': 'Different'}, {'correct': 'false'},
                       {'submission_type': 'timeout'}, {'timestamp': '1970-01-01T00:00:11.002Z'}):
            with self.subTest(change=change):
                rows, audit = self.module().match_ratings([rating_row(**change)], attempts, [ep])
                self.assertIsNone(rows[0]['difficulty_rating'])
                self.assertIn('mismatch', audit.iloc[0]['status'])

    def test_unmatched_attempt_is_not_attached_using_participant_name(self):
        ep, attempts = self.fixture()
        rows, audit = self.module().match_ratings([rating_row(attempt='unknown')], attempts, [ep])
        self.assertIsNone(rows[0]['difficulty_rating'])
        self.assertEqual(audit.iloc[0]['status'], 'attempt_not_in_source_cohort')

    def test_duplicate_and_enriched_copies_contribute_one_rating(self):
        ep, attempts = self.fixture()
        same = rating_row(source_file='medium.csv', source_row=9)
        rows, audit = self.module().match_ratings([
            rating_row(rating='-'), rating_row(), same], attempts, [ep])
        self.assertEqual(rows[0]['difficulty_rating'], 4)
        self.assertEqual(len(rows[0]['rating_source_rows']), 2)
        self.assertEqual(audit.status.eq('matched').sum(), 1)
        self.assertEqual(audit.status.eq('duplicate_rating').sum(), 1)

    def test_conflicting_rating_copies_invalidate_the_episode_target(self):
        ep, attempts = self.fixture()
        rows, audit = self.module().match_ratings([rating_row(), rating_row(rating='8')], attempts, [ep])
        self.assertIsNone(rows[0]['difficulty_rating'])
        self.assertTrue(audit.status.eq('conflicting_attempt_rows').all())

    def test_invalid_trace_cannot_receive_a_target(self):
        ep, attempts = self.fixture()
        ep.update(valid=False, invalid_reasons=['missing_episode_end'])
        rows, audit = self.module().match_ratings([rating_row()], attempts, [ep])
        self.assertIsNone(rows[0]['difficulty_rating'])
        self.assertEqual(audit.iloc[0]['status'], 'invalid_event_trace')

    def test_nonterminal_rating_does_not_label_future_work(self):
        ep, attempts = self.fixture()
        ep['attempt_ids'] = ['a', 'b']
        attempts['b'] = dict(attempts['a'], attempt_id='b', event_time_ms=21000)
        rows, audit = self.module().match_ratings([rating_row()], attempts, [ep])
        self.assertIsNone(rows[0]['difficulty_rating'])
        self.assertEqual(audit.iloc[0]['status'], 'nonterminal_rated_attempt')

    def test_project_excludes_gaps_and_unrated_episodes_and_removes_cl_targets(self):
        from . import build_dataset as shared
        from .test_dataset_fusion import BASE
        module = self.module()
        with tempfile.TemporaryDirectory():
            config = shared.load_config()
            # A complete real v2-shaped frame, already verified by the sensor policy.
            from .test_feature_policy import FeaturePolicyTests
            FeaturePolicyTests.setUpClass()
            self.addCleanup(FeaturePolicyTests.tearDownClass)
            frame = FeaturePolicyTests().frame()
            from . import v2_features
            frame = v2_features.apply_policy(frame, config)
            frame['window_id'] = ['s|a|b']
            frame['session_id'] = 's'
            frame['participant_id'] = 'p'
            frame['window_start'] = pd.Timestamp(BASE, unit='ms', tz='UTC').isoformat()
            frame['window_end'] = pd.Timestamp(BASE+15000, unit='ms', tz='UTC').isoformat()
            frame['window_seconds'] = 15
            frame['study_phase'] = 'active_task'
            frame['puzzle_set'] = 'set'
            frame['puzzle_id'] = 'q'
            frame['window_phase_mix'] = json.dumps([{'study_phase': 'active_task', 'duration_seconds': 15,
                                                     'puzzle_set': 'set', 'puzzle_id': 'q'}])
            frame['cognitive_load_label'] = 'Low'
            frame['window_label_mix'] = '{"Low":15}'
            ep = episode('p', 15, session_id='s', intervals=[[BASE, BASE+15000]], difficulty_rating=7)
            views, mapping, excluded = module.project(frame, [ep], config, {})
            self.assertEqual(views['good_acceptable'].difficulty_rating.tolist(), [7])
            self.assertNotIn('cognitive_load_label', views['all_data'])
            self.assertNotIn('window_label_mix', views['all_data'])
            self.assertTrue(excluded.empty)
            self.assertEqual(mapping.episode_id.tolist(), ['p'])
            gap = copy.deepcopy(ep)
            gap['intervals'] = [[BASE, BASE+5000], [BASE+10000, BASE+15000]]
            self.assertTrue(module.project(frame, [gap], config, {})[0]['all_data'].empty)
            ep['difficulty_rating'] = None
            ep['rating_status'] = 'missing_rating'
            self.assertTrue(module.project(frame, [ep], config, {})[0]['all_data'].empty)


class BundleTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('analysis_sessions.dataset_fusion.build_dataset_v4')
        except ModuleNotFoundError:
            self.fail('Verified v4 export and Batch 06 packaging are not implemented.')

    def test_build_verify_package_and_rebuild_preserve_all_inputs(self):
        builder = self.module()
        from . import build_dataset as shared
        from . import build_dataset_v2 as v2
        from .test_dataset_fusion import BASE
        from .test_dataset_v2 import ExportTests
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = ExportTests().source(root/'raw')
            events, logs = [], []
            for i, (key, start, end, mode, outcome, rating) in enumerate([
                    ('easy-1', 0, 30000, 'manual', 'correct', '2'),
                    ('medium', 31000, 60000, 'timeout', 'incorrect', '6'),
                    ('hard', 61000, 119000, 'manual', 'correct', '9')]):
                aid = f'a{i}'
                for kind, when in [('set_started', start), ('puzzle_started', start),
                                   ('attempt_submitted', end), ('set_ended', end+1)]:
                    events.append(event(BASE+when, kind, session_id='source-Andrean', puzzle_set=key,
                        difficulty_level=key.split('-')[0], attempt_id=aid if kind=='attempt_submitted' else '',
                        submission_type=mode if kind=='attempt_submitted' else '',
                        outcome=outcome if kind=='attempt_submitted' else ''))
                logs.append(rating_row(attempt=aid, rating=rating, session_id='source-Andrean', puzzle_set=key,
                    submission_type=mode, correct=str(outcome=='correct').lower(),
                    timestamp=pd.Timestamp(BASE+end, unit='ms', tz='UTC').isoformat()))
            pd.DataFrame(events).to_csv(raw/'task_events.csv', index=False)
            source = root/'v2'
            v2.build_dataset(root/'raw', source, admission_policy='feature_aware_quality_v1')
            log_dir = root/'ratings'
            log_dir.mkdir()
            from . import v4_ratings
            for name in v4_ratings.RATING_FILES:
                with (log_dir/name).open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=v4_ratings.LOG_FIELDS, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(r for r in logs if r['puzzle_set']+'.csv'==name)
            protected = {p.name: shared.snapshot(p) for p in [source, root/'raw', log_dir]}
            output = root/'v4'
            manifest = builder.build_dataset(source, log_dir, output)
            self.assertEqual(builder.verify_bundle(output), manifest)
            self.assertEqual(manifest['summary']['rated_episodes'], 3)
            data = pd.read_csv(output/'window_15s_step_5s/good_acceptable.csv')
            self.assertEqual(set(data.difficulty_rating), {2, 6, 9})
            self.assertNotIn('cognitive_load_label', data)
            self.assertNotIn('window_label_mix', data)
            origin = pd.read_csv(source/'window_15s_step_5s/all_data.csv').set_index('window_id')
            archive = pd.read_csv(output/'window_15s_step_5s/all_data.csv').set_index('window_id')
            pd.testing.assert_frame_equal(archive[shared.FEATURE_COLUMNS], origin.loc[archive.index, shared.FEATURE_COLUMNS])
            package = root/'batch-06/dataset/v4_self_reported_rating'
            builder.export_package(output, package)
            builder.verify_package(package, output)
            self.assertEqual(len(list(package.rglob('*.csv'))), 3)
            with self.assertRaises(FileExistsError):
                builder.build_dataset(source, log_dir, output)
            again = builder.build_dataset(source, log_dir, output, replace=True)
            self.assertEqual(again['datasets'], manifest['datasets'])
            builder.export_package(output, package, replace=True)
            self.assertEqual(protected, {p.name: shared.snapshot(p) for p in [source, root/'raw', log_dir]})
            # Resigning an altered rating must not bypass semantic reconstruction.
            path = output/'window_15s_step_5s/good_acceptable.csv'
            forged = pd.read_csv(path)
            forged.loc[0, 'difficulty_rating'] = 10
            forged.to_csv(path, index=False)
            signed = json.loads((output/'manifest.json').read_text())
            signed['output_sha256'][path.relative_to(output).as_posix()] = shared.digest(path)
            (output/'manifest.json').write_text(json.dumps(signed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'derived artifact'):
                builder.verify_bundle(output)
            protected_package = shared.snapshot(package)
            with self.assertRaises(ValueError):
                builder.export_package(output, package, replace=True)
            self.assertEqual(shared.snapshot(package), protected_package)

    def test_failed_publication_restores_previous_directory(self):
        builder = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, staged = root/'v4', root/'staged'
            old.mkdir()
            staged.mkdir()
            (old/'sentinel').write_text('old')
            (staged/'sentinel').write_text('new')
            def verify(path):
                if Path(path)==old and (old/'sentinel').read_text()=='new':
                    raise ValueError('Publication check failed')
            with self.assertRaisesRegex(ValueError, 'Publication check failed'):
                builder.publish(staged, old, verify, replace=True)
            self.assertEqual((old/'sentinel').read_text(), 'old')
            self.assertEqual((staged/'sentinel').read_text(), 'new')

    def test_publisher_rejects_protected_dataset_directories(self):
        builder = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root/'stage'
            staged.mkdir()
            for name in ('v2_session_phase', 'v3_behavioral_proxy'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    builder.publish(staged, root/name, lambda _p: None)


if __name__ == '__main__':
    unittest.main()
