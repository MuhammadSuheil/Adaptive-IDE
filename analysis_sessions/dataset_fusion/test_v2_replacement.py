"""Quality replacements preserve class counts and existing source support."""
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


def row(identity, *, new=False, **changes):
    result = dict(window_id=identity, session_id='mon' if new else 'old',  # noqa: C408
                  participant_id='Mon' if new else 'Person', window_seconds=25,
                  cognitive_load_label='Low', quality_tier='acceptable',
                  gaze_validation_passed=new, baseline_usable=True,
                  eye_temporal_coverage=.95 if new else .9, face_ear_ratio=1.,
                  valid_gaze_ratio=.9 if new else .8,
                  iris_available_ratio=.9 if new else .8,
                  clipping_ratio_valid=.05 if new else .15,
                  hrv_coverage=.95 if new else .82,
                  hrv_data_age_seconds=.5 if new else 2.5)
    return {**result, **changes}


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('analysis_sessions.dataset_fusion.v2_replacement')
        except ModuleNotFoundError:
            self.fail('Conservative quality replacement is not implemented yet.')

    def select(self, rows, reference=('old-a', 'keep')):
        return self.module.curate_window(pd.DataFrame(rows), reference, ['mon'])

    def source(self, candidate=None):
        return [row('old-a'), row('keep', quality_tier='good'),
                candidate or row('new-a', new=True)]

    def test_replaces_only_weaker_acceptable_and_keeps_counts(self):
        frame, pairs = self.select(self.source())
        selected = frame[frame.curation_selected]
        self.assertEqual(set(selected.window_id), {'keep', 'new-a'})
        self.assertEqual(selected.cognitive_load_label.value_counts().to_dict(), {'Low': 2})
        self.assertEqual([(p['old_window_id'], p['new_window_id']) for p in pairs], [('old-a', 'new-a')])
        self.assertEqual(frame.set_index('window_id').loc['old-a', 'curation_reason'], 'replaced_reference')

    def test_each_quality_metric_baseline_and_validation_is_required(self):
        for name, value in [('eye_temporal_coverage', .899), ('face_ear_ratio', .99),
                            ('valid_gaze_ratio', .799), ('iris_available_ratio', .799),
                            ('clipping_ratio_valid', .151), ('hrv_coverage', .819),
                            ('hrv_data_age_seconds', 2.501), ('baseline_usable', False),
                            ('gaze_validation_passed', False), ('quality_tier', 'poor'),
                            ('valid_gaze_ratio', float('nan'))]:
            with self.subTest(metric=name):
                frame, pairs = self.select(self.source(row('new-a', new=True, **{name: value})))
                self.assertEqual(pairs, [])
                self.assertEqual(set(frame.loc[frame.curation_selected, 'window_id']), {'old-a', 'keep'})

    def test_old_good_passed_validation_and_last_person_class_are_protected(self):
        for changes in ({'quality_tier': 'good'}, {'gaze_validation_passed': True},
                        {'participant_id': 'OnlyPerson'}):
            with self.subTest(changes=changes):
                records = self.source()
                records[0].update(changes)
                _, pairs = self.select(records)
                self.assertEqual(pairs, [])

    def test_other_class_and_window_length_cannot_replace_reference(self):
        _, pairs = self.select(self.source(row('new-a', new=True, cognitive_load_label='High')))
        self.assertEqual(pairs, [])
        with self.assertRaises(ValueError):
            self.select(self.source(row('new-a', new=True, window_seconds=15)))

    def test_matching_finds_two_pairs_when_greedy_would_find_one(self):
        records = [row('old-a'), row('old-b', hrv_coverage=.91),
                   row('keep', quality_tier='good'),
                   row('new-a', new=True), row('new-b', new=True, hrv_coverage=.85)]
        frame, pairs = self.select(records, ('old-a', 'old-b', 'keep'))
        self.assertEqual({(p['old_window_id'], p['new_window_id']) for p in pairs},
                         {('old-b', 'new-a'), ('old-a', 'new-b')})
        reversed_frame, reversed_pairs = self.select(list(reversed(records)), ('keep', 'old-b', 'old-a'))
        self.assertEqual(pairs, reversed_pairs)
        self.assertEqual(set(frame.loc[frame.curation_selected, 'window_id']),
                         set(reversed_frame.loc[reversed_frame.curation_selected, 'window_id']))

    def test_tolerance_and_unknown_flags_do_not_create_false_improvements(self):
        _, pairs = self.select(self.source(row('new-a', new=True, hrv_coverage=.82-5e-10)))
        self.assertEqual(len(pairs), 1)
        for value in ('unknown', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.select(self.source(row('new-a', new=True, baseline_usable=value)))

    def test_reference_and_candidate_identity_errors_fail_closed(self):
        for records, reference in [(self.source(), ('missing',)),
                                   (self.source()+[row('new-a', new=True)], ('old-a', 'keep')),
                                   (self.source(), ('old-a', 'keep', 'new-a'))]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.select(records, reference)


class ReplacementBuildTests(unittest.TestCase):
    def test_full_rebuild_and_second_rebuild_preserve_selected_csv_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixtures.ExportTests().source(root/'sources')
            eye = pd.read_csv(old/'eye_tracking.csv')
            eye.loc[eye.index % 10 == 0, 'gaze_x_raw'] = -100.
            eye.to_csv(old/'eye_tracking.csv', index=False)
            builder.build_dataset(root/'sources', root/'active')
            fresh = root/'sources/Mon'
            shutil.copytree(old, fresh)
            for filename in ('eye_tracking.csv', 'hrv.csv', 'rr_raw.csv', 'task_events.csv'):
                frame = pd.read_csv(fresh/filename, keep_default_na=False)
                frame['session_id'] = 'mon'
                if filename == 'eye_tracking.csv':
                    frame['gaze_x_raw'] = 500.
                frame.to_csv(fresh/filename, index=False)
            calibration = json.loads((fresh/'eye_summary_calibration.json').read_text())
            calibration['attempts'][-1]['passed_thresholds'] = True
            (fresh/'eye_summary_calibration.json').write_text(json.dumps(calibration))
            before = shared.snapshot(root/'sources')
            try:
                manifest = builder.build_dataset(root/'sources', root/'active', replace=True,
                                                 replacement_session='mon')
            except TypeError as error:
                self.fail(f'Builder does not support conservative replacement: {error}')
            self.assertEqual(before, shared.snapshot(root/'sources'))
            self.assertEqual(manifest['admission_policy'], 'conservative_quality_replacement_v1')
            self.assertTrue((root/'active/replacement_selection.json').exists())
            pairs = pd.read_csv(root/'active/replacement_pairs.csv')
            self.assertEqual(pairs.groupby('window_seconds').size().to_dict(), {15: 15, 25: 9, 45: 3})
            first = shared.snapshot(root/'active')
            builder.build_dataset(root/'sources', root/'active', replace=True)
            for filename, digest in first.items():
                if filename.endswith('.csv') or filename == 'replacement_selection.json':
                    self.assertEqual(shared.digest(root/'active'/filename), digest, filename)
            manifest_path = root/'active/manifest.json'
            broken = json.loads(manifest_path.read_text())
            ledger = root/'active/replacement_pairs.csv'
            tampered = pd.read_csv(ledger)
            tampered.loc[0, 'new_hrv_coverage'] = 999.
            tampered.to_csv(ledger, index=False)
            broken['output_sha256']['replacement_pairs.csv'] = shared.digest(ledger)
            manifest_path.write_text(json.dumps(broken))
            with self.assertRaises((ValueError, AssertionError)):
                builder.verify_bundle(root/'active')


if __name__ == '__main__':
    unittest.main()
