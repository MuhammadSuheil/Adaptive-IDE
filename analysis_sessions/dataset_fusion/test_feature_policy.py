"""Feature-aware admission, masking, migration and reproducible publication."""

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset as shared
from analysis_sessions.dataset_fusion import build_dataset_v2 as builder
from analysis_sessions.dataset_fusion import test_dataset_v2 as fixtures
from analysis_sessions.dataset_fusion import test_v2_stages as stage_fixtures

POLICY = 'feature_aware_quality_v1'
CORE = ['rmssd_ms', 'sdnn_ms', 'blink_rate_per_minute', 'ear_left_mean',
        'ear_left_std', 'ear_right_mean', 'ear_right_std', 'iris_size_estimate_mean',
        'iris_size_estimate_std', 'iris_size_delta_mean']
SPATIAL = ['gaze_x_norm_mean', 'gaze_x_norm_std', 'gaze_y_norm_mean', 'gaze_y_norm_std',
           'eye_state_fixation_ratio', 'eye_state_saccade_ratio', 'eye_state_transition_ratio',
           'eye_state_unclassified_ratio', 'eye_state_other_ratio']
FORBIDDEN = {'core_quality_tier', 'core_usable', 'head_pose_usable', 'spatial_usable'}


class FeaturePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        fixtures.ExportTests().source(cls.root/'sources')
        cls.config = shared.load_config()
        cls.session, _ = builder.build_session(cls.root/'sources/Andrean', cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def frame(self, **changes):
        row = copy.deepcopy(self.session[25][0][0])
        row.update(changes)
        row.update(shared.quality_tier(row, self.config))
        return pd.DataFrame([row])

    def module(self):
        return importlib.import_module('analysis_sessions.dataset_fusion.v2_features')

    def test_gaze_failure_does_not_remove_complete_core(self):
        frame = self.frame(valid_gaze_ratio=.1, clipping_ratio_valid=.9)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'views'
            counts = builder.export_window(out, frame.to_dict('records'), [], [], [], 25,
                                           config=self.config, policy=POLICY)
            self.assertEqual(counts['good_acceptable']['windows'], 1)
            curated = pd.read_csv(out/'good_acceptable.csv')
            self.assertEqual(curated.quality_tier.tolist(), ['good'])
            self.assertTrue(curated[SPATIAL].isna().all().all())
            self.assertTrue(np.isfinite(curated[CORE].to_numpy()).all())

    def test_failed_manual_validation_masks_curated_not_measurement_archive(self):
        frame = self.frame(gaze_validation_passed=False)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'views'
            builder.export_window(out, frame.to_dict('records'), [], [], [], 25,
                                  config=self.config, policy=POLICY)
            raw = pd.read_csv(out/'all_data.csv')
            curated = pd.read_csv(out/'good_acceptable.csv')
            self.assertTrue(raw[SPATIAL].notna().all().all())
            self.assertTrue(curated[SPATIAL].isna().all().all())
            pd.testing.assert_frame_equal(raw[CORE], curated[CORE])
            self.assertFalse(FORBIDDEN.intersection(curated.columns))

    def test_default_contract_excludes_redundancy_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'views'
            builder.export_window(out, self.frame().to_dict('records'), [], [], [], 25,
                                  config=self.config, policy=POLICY)
            contract = json.loads((out/'feature_columns.json').read_text())
            self.assertEqual(contract['features'], CORE)
            self.assertEqual({k: len(v['features']) for k, v in contract['profiles'].items()},
                             {'core': 10, 'core_head': 14, 'validated_fusion': 22})
            self.assertNotIn('iris_size_delta_std', contract['profiles']['validated_fusion']['features'])
            self.assertEqual(contract['target'], 'cognitive_load_label')

    def test_missing_core_is_rejected_not_imputed(self):
        mod = self.module()
        frame = self.frame(blink_rate_per_minute=np.nan)
        selected = mod.apply_policy(frame, self.config)
        self.assertEqual(selected.quality_tier.tolist(), ['poor'])
        self.assertFalse(selected.curation_selected.any())
        self.assertTrue(selected.blink_rate_per_minute.isna().all())
        self.assertIn('blink_rate_per_minute', selected.iloc[0].acceptable_failed_criteria)

    def test_core_rejects_insufficient_hrv_and_invalid_values(self):
        mod = self.module()
        for change in ({'hrv_coverage': .79}, {'hrv_rr_pairs': 9}, {'hrv_data_age_seconds': 3.01},
                       {'hrv_status': 'warming_up'}, {'hrv_usable': False}, {'sdnn_ms': -1},
                       {'ear_left_mean': 1.1}, {'iris_size_estimate_mean': 0}, {'ear_left_std': np.inf}):
            with self.subTest(change=change):
                result = mod.apply_policy(self.frame(**change), self.config)
                self.assertEqual(result.quality_tier.tolist(), ['poor'])

    def test_baseline_hrv_ratio_not_required_and_negative_iris_delta_valid(self):
        mod = self.module()
        frame = self.frame(baseline_usable=False, rmssd_baseline_ratio=np.nan, iris_size_delta_mean=-.03)
        result = mod.apply_policy(frame, self.config)
        self.assertEqual(result.quality_tier.tolist(), ['good'])
        self.assertEqual(result.iris_size_delta_mean.tolist(), [-.03])

    def test_optional_profiles_filter_rows_without_new_flags(self):
        mod = self.module()
        frame = mod.apply_policy(self.frame(gaze_validation_passed=True), self.config)
        verified = {frame.iloc[0].session_id: {'verified': True}}
        self.assertEqual(len(mod.profile_frame(frame, 'core', self.config, verified)), 1)
        self.assertEqual(len(mod.profile_frame(frame, 'core_head', self.config, verified)), 1)
        self.assertEqual(len(mod.profile_frame(frame, 'validated_fusion', self.config, verified)), 1)
        for change in ({'gaze_validation_passed': False}, {'clipping_ratio_valid': .21},
                       {'valid_gaze_ratio': .69}, {'gaze_x_norm_std': np.nan}, {'head_pitch_std': np.nan}):
            with self.subTest(change=change):
                changed = frame.assign(**change)
                self.assertEqual(len(mod.profile_frame(changed, 'validated_fusion', self.config, verified)), 0)
        self.assertEqual(len(mod.profile_frame(frame, 'validated_fusion', self.config, {})), 0)

    def test_validation_decision_is_verified_against_source_points_and_thresholds(self):
        mod = self.module()
        source = {'config': {'validation': {'enabled': True, 'max_median_error_px': 180,
                                          'max_p95_error_px': 350, 'max_corner_error_px': 300}},
                  'attempts': [{'phase': 'held_out_validation', 'passed_thresholds': True,
                                'mean_target_error_median_px': 100., 'mean_target_error_p95_px': 100.,
                                'max_corner_error_px': 100.,
                                'points': [{'target': [0., 0.], 'samples': [{'prediction': [100., 0.]}],
                                            'mean_prediction_error_px': 100.} for _ in range(5)]}]}
        self.assertTrue(mod.audit_validation(source)['verified'])
        failed = copy.deepcopy(source)
        failed['attempts'][-1].update(passed_thresholds=False, decision='accepted_manual')
        self.assertFalse(mod.audit_validation(failed)['verified'])
        forged = copy.deepcopy(source)
        forged['attempts'][-1]['mean_target_error_median_px'] = 20.
        self.assertFalse(mod.audit_validation(forged)['verified'])
        self.assertFalse(mod.audit_validation({})['verified'])

    def test_explicit_migration_rebuild_inherits_policy_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root/'sources')
            old = builder.build_dataset(root/'sources', root/'bundle')
            before = shared.snapshot(root/'sources')
            new = builder.build_dataset(root/'sources', root/'bundle', replace=True, admission_policy=POLICY)
            self.assertEqual(new['admission_policy'], POLICY)
            self.assertEqual(new['previous_admission_policy'], old['admission_policy'])
            self.assertEqual(before, shared.snapshot(root/'sources'))
            first = shared.snapshot(root/'bundle')
            again = builder.build_dataset(root/'sources', root/'bundle', replace=True)
            self.assertEqual(again['admission_policy'], POLICY)
            for key, digest in first.items():
                if key.endswith(('.csv', 'feature_columns.json')):
                    self.assertEqual(digest, shared.digest(root/'bundle'/key), key)
            builder.verify_bundle(root/'bundle')

    def test_rehashed_unmasked_gaze_is_rejected_by_bundle_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root/'sources')
            builder.build_dataset(root/'sources', root/'bundle', admission_policy=POLICY)
            path = root/'bundle/window_25s_step_5s/good_acceptable.csv'
            frame = pd.read_csv(path)
            frame.loc[0, 'gaze_x_norm_mean'] = .5
            frame.to_csv(path, index=False)
            manifest_path = root/'bundle/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            hashes = shared.snapshot(root/'bundle')
            hashes.pop('manifest.json')
            manifest['output_sha256'] = hashes
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                builder.verify_bundle(root/'bundle')

    def test_migration_preserves_staged_history_and_pinned_timestamp_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = fixtures.ExportTests().source(root/'sources')
            eye = pd.read_csv(old/'eye_tracking.csv')
            eye.loc[eye.index % 10 == 0, 'gaze_x_raw'] = -100.
            eye.to_csv(old/'eye_tracking.csv', index=False)
            builder.build_dataset(root/'sources', root/'bundle')
            copier = stage_fixtures.StageBuildTests()
            copier.copy_session(old, root/'sources/Mon', 'mon')
            builder.build_dataset(root/'sources', root/'bundle', replace=True, replacement_session='mon')
            copier.copy_session(old, root/'sources/Alif', 'alif', tie=True)
            builder.build_dataset(root/'sources', root/'bundle', replace=True,
                                  replacement_session='alif', exclude_tied_eye_windows=True)
            context = json.loads((root/'bundle/replacement_selection.json').read_text())
            pair_hash = shared.digest(root/'bundle/replacement_pairs.csv')
            sources = shared.snapshot(root/'sources')
            new = builder.build_dataset(root/'sources', root/'bundle', replace=True, admission_policy=POLICY)
            self.assertEqual(new['timestamp_policies'], context['timestamp_policies'])
            self.assertEqual(shared.digest(root/'bundle/history/replacement_pairs.csv'), pair_hash)
            self.assertEqual(shared.snapshot(root/'sources'), sources)
            for w in (15, 25, 45):
                data = pd.read_csv(root/f'bundle/window_{w}s_step_5s/good_acceptable.csv')
                excluded = pd.read_csv(root/f'bundle/window_{w}s_step_5s/excluded_windows.csv')
                self.assertTrue(set(data.session_id).issuperset({'mon', 'alif'}))
                self.assertTrue(set(data.window_id).isdisjoint(excluded.window_id))
            builder.verify_bundle(root/'bundle')


if __name__ == '__main__':
    unittest.main()
