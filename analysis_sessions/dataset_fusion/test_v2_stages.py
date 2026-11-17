"""Staged curation locks previous replacements and excludes ambiguous timing."""
import copy
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset as shared
from analysis_sessions.dataset_fusion import build_dataset_v2 as builder
from analysis_sessions.dataset_fusion import test_dataset_v2 as fixtures
from analysis_sessions.dataset_fusion.test_dataset_fusion import BASE, eye_source
from analysis_sessions.dataset_fusion.test_v2_replacement import row


def feature(module, name, test):
    try:
        return getattr(importlib.import_module('analysis_sessions.dataset_fusion.'+module), name)
    except (ImportError, AttributeError):
        test.fail(f'Missing audited behavior: {module}.{name}')


class TimestampTests(unittest.TestCase):
    def source(self):
        frame = eye_source([0, 100, 100, 200, 300])
        frame['frame_index'] = [1, 2, 3, 4, 5]
        frame['capture_frame_id'] = [10, 11, 12, 13, 14]
        frame['total_blinks'] = [0, 0, 1, 1, 2]
        return frame

    def test_default_remains_strict_but_explicit_ties_preserve_rows_and_blinks(self):
        source = self.source()
        original = source.copy(deep=True)
        with self.assertRaises(ValueError):
            shared.prepare_eye(source, {'width': 1000, 'height': 800}, shared.load_config())
        try:
            prepared = shared.prepare_eye(source, {'width': 1000, 'height': 800}, shared.load_config(),
                                          allow_equal_timestamps=True)
        except TypeError as error:
            self.fail(f'Explicit audited timestamp handling is missing: {error}')
        self.assertEqual(prepared.timestamp_ms.tolist(), original.timestamp_ms.tolist())
        self.assertEqual(prepared.blink_increment.iloc[1:].tolist(), [0., 1., 0., 1.])
        pd.testing.assert_frame_equal(source, original)

    def test_nonfinite_backward_or_repeated_frame_id_still_fails(self):
        for field, value in [('timestamp_ms', float('inf')), ('timestamp_ms', BASE-1),
                             ('frame_index', 2), ('capture_frame_id', 11)]:
            f = self.source()
            f.loc[2, field] = value
            with self.subTest(field=field, value=value):
                try:
                    with self.assertRaises(ValueError):
                        shared.prepare_eye(f, {'width': 1000, 'height': 800}, shared.load_config(),
                                           allow_equal_timestamps=True)
                except TypeError as error:
                    self.fail(f'Audited timestamp validation is missing: {error}')

    def test_right_closed_boundary_and_policy_hash_are_checked(self):
        make = feature('v2_timestamps', 'make_policy', self)
        check = feature('v2_timestamps', 'check_policy', self)
        touched = feature('v2_timestamps', 'touches', self)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            f = self.source()
            f.to_csv(directory/'eye_tracking.csv', index=False)
            policy = make(directory)
            self.assertEqual(len(policy['groups']), 1)
            self.assertEqual(policy['groups'][0]['csv_lines'], [3, 4])
            check(directory, f, policy)
            self.assertTrue(touched(BASE, BASE+100, policy))
            self.assertFalse(touched(BASE+100, BASE+200, policy))
            f.loc[0, 'EAR_Left'] = .5
            f.to_csv(directory/'eye_tracking.csv', index=False)
            with self.assertRaises(ValueError):
                check(directory, f, policy)


class StageTests(unittest.TestCase):
    def context(self):
        stage = dict(policy='conservative_quality_replacement_v1', tolerance=1e-9,  # noqa: C408
                     algorithm='maximum_cardinality_flow_sorted_window_ids_with_participant_class_cap',
                     reference_manifest_sha256='a'*64, reference_session_ids=['old'],
                     reference_window_ids={str(s): ['keep', 'old-a', 'old-b'] for s in (15, 25, 45)},
                     candidate_session_ids=['mon'])
        second = copy.deepcopy(stage)
        second['reference_session_ids'] = ['mon', 'old']
        second['reference_window_ids'] = {str(s): ['keep', 'new-mon', 'old-b'] for s in (15, 25, 45)}
        second['candidate_session_ids'] = ['alif']
        return {**copy.deepcopy(stage), 'policy': 'conservative_quality_replacement_v2',
                'candidate_session_ids': ['mon', 'alif'], 'stages': [stage, second], 'timestamp_policies': {}}

    def test_second_stage_preserves_first_pair_and_original_good(self):
        curate = feature('v2_history', 'curate_history', self)
        # Mon dominates old-a but not old-b; Alif can replace the remaining old-b.
        f = pd.DataFrame([row('old-a'), row('old-b', hrv_coverage=.96),
                          row('keep', quality_tier='good'), row('new-mon', new=True),
                          row('new-alif', new=True, session_id='alif', participant_id='Alif', hrv_coverage=.99)])
        selected, pairs = curate(f, self.context(), 25)
        self.assertEqual(set(selected.loc[selected.curation_selected, 'window_id']), {'keep', 'new-mon', 'new-alif'})
        self.assertEqual([(p['selection_stage'], p['old_window_id'], p['new_window_id']) for p in pairs],
                         [(1, 'old-a', 'new-mon'), (2, 'old-b', 'new-alif')])
        self.assertEqual(selected.set_index('window_id').loc['new-mon', 'curation_reason'], 'replacement_selected')
        _, again = curate(f.iloc[::-1], self.context(), 25)
        self.assertEqual(pairs, again)

    def test_stage_cannot_remove_reference_ids_or_reuse_candidate_cohort(self):
        curate = feature('v2_history', 'curate_history', self)
        f = pd.DataFrame([row('old-a'), row('old-b', hrv_coverage=.96),
                          row('keep', quality_tier='good'), row('new-mon', new=True),
                          row('new-alif', new=True, session_id='alif', participant_id='Alif', hrv_coverage=.99)])
        for change in ('reference', 'cohort'):
            context = self.context()
            if change == 'reference':
                context['stages'][1]['reference_window_ids']['25'] = ['keep', 'new-mon']
            else:
                context['stages'][1]['candidate_session_ids'] = ['mon']
            with self.subTest(change=change), self.assertRaises(ValueError):
                curate(f, context, 25)

    def test_expected_counts_are_a_publication_guard_not_a_selection_quota(self):
        curate = feature('v2_history', 'curate_history', self)
        f = pd.DataFrame([row('old-a'), row('old-b', hrv_coverage=.96),
                          row('keep', quality_tier='good'), row('new-mon', new=True),
                          row('new-alif', new=True, session_id='alif', participant_id='Alif', hrv_coverage=.99)])
        context = self.context()
        context['stages'][1]['expected_counts'] = {'25': {'Low': 99, 'Medium': 0, 'High': 0}}
        with self.assertRaisesRegex(ValueError, 'expected|Expected'):
            curate(f, context, 25)


class StageBuildTests(unittest.TestCase):
    def copy_session(self, source, target, session_id, *, clip=False, tie=False):
        shutil.copytree(source, target)
        for name in ('eye_tracking.csv', 'hrv.csv', 'rr_raw.csv', 'task_events.csv'):
            f = pd.read_csv(target/name, keep_default_na=False)
            f['session_id'] = session_id
            if name == 'eye_tracking.csv':
                f['gaze_x_raw'] = 500.
                if clip:
                    f.loc[f.index % 10 == 0, 'gaze_x_raw'] = -100.
                f['frame_index'] = range(1, len(f)+1)
                f['capture_frame_id'] = range(1, len(f)+1)
                if tie:
                    f.loc[10, 'timestamp_ms'] = f.loc[9, 'timestamp_ms']
            f.to_csv(target/name, index=False)
        calibration = json.loads((target/'eye_summary_calibration.json').read_text())
        calibration['attempts'][-1]['passed_thresholds'] = True
        (target/'eye_summary_calibration.json').write_text(json.dumps(calibration))

    def test_migrate_mon_then_alif_exclude_ties_and_rebuild_without_new_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixtures.ExportTests().source(root/'sources')
            eye = pd.read_csv(old/'eye_tracking.csv')
            eye.loc[eye.index % 10 == 0, 'gaze_x_raw'] = -100.
            eye.to_csv(old/'eye_tracking.csv', index=False)
            builder.build_dataset(root/'sources', root/'active')
            self.copy_session(old, root/'sources/Mon', 'mon')
            builder.build_dataset(root/'sources', root/'active', replace=True, replacement_session='mon')
            old_pairs = pd.read_csv(root/'active/replacement_pairs.csv')
            self.copy_session(old, root/'sources/Alif', 'alif', tie=True)
            source_hashes = shared.snapshot(root/'sources')
            try:
                manifest = builder.build_dataset(root/'sources', root/'active', replace=True,
                    replacement_session='alif', exclude_tied_eye_windows=True)
            except TypeError as error:
                self.fail(f'Staged audited build is missing: {error}')
            self.assertEqual(manifest['admission_policy'], 'conservative_quality_replacement_v2')
            pairs = pd.read_csv(root/'active/replacement_pairs.csv')
            pd.testing.assert_frame_equal(pairs.loc[pairs.selection_stage.eq(1), old_pairs.columns].reset_index(drop=True), old_pairs)
            excluded = pd.read_csv(root/'active/window_15s_step_5s/excluded_windows.csv')
            self.assertTrue(excluded.exclusion_reason.eq('ambiguous_eye_timestamp').any())
            first = shared.snapshot(root/'active')
            builder.build_dataset(root/'sources', root/'active', replace=True, replacement_session='alif')
            for name, digest in first.items():
                if name.endswith('.csv') or name == 'replacement_selection.json':
                    self.assertEqual(shared.digest(root/'active'/name), digest, name)
            self.assertEqual(source_hashes, shared.snapshot(root/'sources'))
            context = json.loads((root/'active/replacement_selection.json').read_text())
            self.assertEqual(len(context['stages']), 2)


if __name__ == '__main__':
    unittest.main()
