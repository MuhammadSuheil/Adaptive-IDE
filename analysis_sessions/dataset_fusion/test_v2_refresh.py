"""Regression tests for quality-only admission and transactional replacement."""

import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset_v2 as v2
from analysis_sessions.dataset_fusion import build_dataset as v1
from analysis_sessions.dataset_fusion import test_dataset_v2 as fixtures


class RefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = tempfile.TemporaryDirectory()
        root = Path(cls.template.name)
        fixtures.ExportTests().source(root / 'sources')
        v2.build_dataset(root / 'sources', root / 'bundle')

    @classmethod
    def tearDownClass(cls):
        cls.template.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(Path(self.template.name) / 'sources', self.root / 'sources')
        shutil.copytree(Path(self.template.name) / 'bundle', self.root / 'active')

    def function(self, name):
        function = getattr(v2, name, None)
        self.assertTrue(callable(function), f'Missing v2 behavior: {name}')
        return function

    def test_unreviewed_and_questionable_names_do_not_exclude_good_windows(self):
        old = pd.read_csv(self.root / 'active/window_25s_step_5s/all_data.csv')
        # Identical sensor values and quality, different legacy review labels.
        copies = []
        for i, review in enumerate(('proposed_good', 'questionable', 'unreviewed')):
            f = old.copy()
            f['session_id'] = f.session_id + f'_copy{i}'
            f['window_id'] = f.session_id + '|' + f.window_start + '|' + f.window_end
            f['team_review_group'] = review
            f['participant_id'] = f'person{i}'
            copies.append(f)
        records = pd.concat(copies, ignore_index=True)
        path = self.root / 'views'
        v2.export_window(path, records.to_dict('records'), [], [], [], 25)
        good = pd.read_csv(path / 'good_only.csv')
        acceptable = pd.read_csv(path / 'good_acceptable.csv')
        self.assertEqual(set(good.participant_id), {'person0', 'person1', 'person2'})
        self.assertEqual(set(acceptable.participant_id), {'person0', 'person1', 'person2'})

    def test_explicit_replace_rebuilds_and_removes_temporary_rollback(self):
        self.assertIn('replace', inspect.signature(v2.build_dataset).parameters)
        before = v1.snapshot(self.root / 'sources')
        first = v1.snapshot(self.root / 'active')
        manifest = v2.build_dataset(self.root / 'sources', self.root / 'active', replace=True)
        self.assertEqual(manifest['admission_policy'], 'window_quality_only')
        self.assertEqual(before, v1.snapshot(self.root / 'sources'))
        for name, digest in first.items():
            if name.endswith('.csv'):
                self.assertEqual(digest, v1.digest(self.root / 'active' / name), name)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['active', 'sources'])

    def test_unrecognized_directory_is_not_replaced(self):
        guard = self.function('guard_output')
        unknown = self.root / 'unknown'
        unknown.mkdir()
        (unknown / 'sentinel').write_text('keep')
        with self.assertRaises(ValueError):
            guard(self.root / 'sources', unknown, replace=True)
        self.assertEqual((unknown / 'sentinel').read_text(), 'keep')

    def test_source_root_output_root_and_v1_are_protected(self):
        guard = self.function('guard_output')
        for target in (self.root, self.root / 'sources', self.root / 'sources/sub',
                       v2.ROOT, Path(v2.__file__).parent / 'output',
                       Path(v2.__file__).parent / 'output/v1_active_task'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                guard(self.root / 'sources', target, replace=True)

    def test_symlink_output_is_rejected_before_resolution(self):
        guard = self.function('guard_output')
        link = self.root / 'link'
        try:
            link.symlink_to(self.root / 'active', target_is_directory=True)
        except OSError:
            self.skipTest('Symlink permission unavailable')
        with self.assertRaises(ValueError):
            guard(self.root / 'sources', link, replace=True)

    def test_corrupt_existing_bundle_is_not_destroyed(self):
        guard = self.function('guard_output')
        source = self.root / 'active/window_25s_step_5s/good_only.csv'
        source.write_text('unexpected change')
        with self.assertRaises(ValueError):
            guard(self.root / 'sources', self.root / 'active', replace=True)
        self.assertEqual(source.read_text(), 'unexpected change')

    def test_publish_rename_failure_restores_previous_dataset(self):
        publish = self.function('publish_bundle')
        target = self.root / 'active'
        staged = self.root / 'staged'
        shutil.copytree(target, staged)
        before = v1.snapshot(target)
        rename = os.rename

        def fail_install(source, destination):
            if Path(source) == staged:
                raise OSError('injected install failure')
            return rename(source, destination)

        with patch('os.rename', side_effect=fail_install), self.assertRaises(OSError):
            publish(staged, target, replace=True)
        self.assertEqual(before, v1.snapshot(target))
        self.assertFalse(list(self.root.glob('.*rollback*')))

    def test_post_install_verification_failure_restores_previous_dataset(self):
        publish = self.function('publish_bundle')
        target, staged = self.root / 'active', self.root / 'staged'
        shutil.copytree(target, staged)
        before = v1.snapshot(target)
        rename = os.rename

        def corrupt_install(source, destination):
            result = rename(source, destination)
            if Path(source) == staged:
                (target / 'window_25s_step_5s/good_only.csv').write_text('corrupt')
            return result

        with patch('os.rename', side_effect=corrupt_install), self.assertRaises(ValueError):
            publish(staged, target, replace=True)
        self.assertEqual(before, v1.snapshot(target))

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL failure')
    def test_permission_failure_restores_previous_dataset(self):
        publish = self.function('publish_bundle')
        target, staged = self.root / 'active', self.root / 'staged'
        shutil.copytree(target, staged)
        before = v1.snapshot(target)
        with patch('subprocess.run', side_effect=OSError('ACL failure')), self.assertRaises(OSError):
            publish(staged, target, replace=True)
        self.assertEqual(before, v1.snapshot(target))

    def test_unrecoverable_install_keeps_old_dataset_in_rollback_directory(self):
        publish = self.function('publish_bundle')
        target, staged = self.root / 'active', self.root / 'staged'
        shutil.copytree(target, staged)
        before = v1.snapshot(target)
        rename = os.rename

        def fail_install_and_restore(source, destination):
            if Path(source) == staged or 'rollback' in Path(source).name:
                raise OSError('injected rename failure')
            return rename(source, destination)

        with patch('os.rename', side_effect=fail_install_and_restore), self.assertRaises(RuntimeError):
            publish(staged, target, replace=True)
        backups = list(self.root.glob('.*rollback*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(v1.snapshot(backups[0]), before)

    def test_validation_rejects_wrong_blink_ratio_label_and_window_duration(self):
        validate = self.function('validate_frame')
        original = pd.read_csv(self.root / 'active/window_25s_step_5s/all_data.csv')
        config = v1.load_config()
        for column, value in [('blink_rate_per_minute', 999), ('rmssd_baseline_ratio', 999),
                              ('cognitive_load_label', 'Unknown'), ('window_seconds', 15),
                              ('eye_state_fixation_ratio', 2), ('ear_left_std', -1),
                              ('session_id', 'another-recording')]:
            broken = original.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate(broken, config, 25)

    def test_poor_windows_are_not_admitted_even_for_former_good_names(self):
        views = getattr(v2, 'quality_views', None)
        self.assertTrue(callable(views))
        f = pd.read_csv(self.root / 'active/window_25s_step_5s/all_data.csv').iloc[:3].copy()
        f['quality_tier'] = ['good', 'acceptable', 'poor']
        f['team_review_group'] = 'proposed_good'
        result = views(f)
        self.assertEqual(len(result['good_only']), 1)
        self.assertEqual(len(result['good_acceptable']), 2)
        self.assertEqual(len(result['all_data']), 3)

    def test_validation_allows_explained_missing_values_but_not_invented_zero(self):
        validate = self.function('validate_frame')
        f = pd.read_csv(self.root / 'active/window_25s_step_5s/all_data.csv')
        f.loc[0, ['baseline_usable', 'baseline_recorder_ready']] = [False, True]
        f.loc[0, 'baseline_coverage'] = .6
        f.loc[0, ['baseline_rmssd_ms', 'rmssd_baseline_ratio', 'blink_count', 'blink_rate_per_minute']] = np.nan
        f['baseline_rejection_reason'] = f.baseline_rejection_reason.astype(object)
        f.loc[0, 'baseline_rejection_reason'] = 'insufficient_baseline_coverage'
        f.loc[0, 'blink_counter_discontinuities'] = 1
        validate(f, v1.load_config(), 25)
        f.loc[0, 'rmssd_baseline_ratio'] = 0.
        with self.assertRaises(ValueError):
            validate(f, v1.load_config(), 25)

    def test_manifest_rejects_a_stale_count_even_with_valid_checksums(self):
        verify = self.function('verify_bundle')
        path = self.root / 'active/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['datasets']['window_25s_step_5s']['good_only']['windows'] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            verify(self.root / 'active')

    def test_independent_sensor_check_detects_changed_replay_and_eye_aggregate(self):
        audit = self.function('audit_sensor_samples')
        directory = self.root / 'sources/Andrean'
        eye, replay, _, _, _, _ = v2.load_session(directory, v1.load_config())
        import csv
        from datetime import datetime
        with (directory / 'rr_raw.csv').open(newline='') as stream:
            raw = list(csv.DictReader(stream))
        origin = datetime.fromisoformat(replay[25][0]['window_start'])
        replay[15][0]['rmssd_ms'] = 999.
        with self.assertRaises(ValueError):
            audit(raw, eye, replay, origin, v1.load_config())
        replay[15][0]['rmssd_ms'] = 0.
        aggregate = v1.aggregate_eye

        def wrong_blink(*args, **kwargs):
            result = aggregate(*args, **kwargs)
            result['blink_rate_per_minute'] = 999.
            return result

        with patch.object(v1, 'aggregate_eye', side_effect=wrong_blink), self.assertRaises(ValueError):
            audit(raw, eye, replay, origin, v1.load_config())

    def test_build_updates_root_index_from_v1_and_actual_v2_counts(self):
        archive = self.root / 'output/v1_active_task'
        archive.mkdir(parents=True)
        pd.DataFrame({'participant_id': ['a', 'a', 'b', 'b'],
                      'difficulty_level': ['easy', 'easy', 'medium', 'hard']}).to_csv(
                          archive / 'good_acceptable.csv', index=False)
        before = v1.snapshot(archive)
        v2.build_dataset(self.root / 'sources', self.root / 'output/v2_session_phase')
        self.assertTrue((self.root / 'output/README.md').exists())
        text = (self.root / 'output/README.md').read_text(encoding='utf-8')
        self.assertIn('| 2 | 1 | 1 | 4 | 2 |', text)
        self.assertIn('| 2 | 2 | 8 | 12 | 1 |', text)
        self.assertEqual(v1.snapshot(archive), before)

    @unittest.skipUnless(os.name == 'nt', 'Windows permission inheritance')
    def test_root_index_inherits_output_permissions_after_private_staging(self):
        archive = self.root / 'public/v1_active_task'
        archive.mkdir(parents=True)
        pd.DataFrame({'participant_id': ['a'], 'difficulty_level': ['easy']}).to_csv(
            archive / 'good_acceptable.csv', index=False)
        subprocess.run(['icacls', str(archive.parent), '/grant', '*S-1-1-0:(OI)(CI)RX', '/q'],
                       check=True, capture_output=True)
        v2.build_dataset(self.root / 'sources', archive.parent / 'v2_session_phase')
        readme = archive.parent / 'README.md'
        command = f"$acl = Get-Acl -LiteralPath '{readme}'; $acl.AreAccessRulesProtected; $acl.Access | ForEach-Object {{ $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value }}"
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.splitlines()[0].strip(), 'False')
        self.assertIn('S-1-1-0', result.stdout)


if __name__ == '__main__':
    unittest.main()
