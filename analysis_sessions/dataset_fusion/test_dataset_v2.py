"""Contracts for phase labels, causal RR replay and versioned exports."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import subprocess
import unittest

import numpy as np
import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset_v2 as v2
from analysis_sessions.dataset_fusion import build_dataset as v1
from analysis_sessions.dataset_fusion.test_dataset_fusion import fixture, BASE


class LabelTests(unittest.TestCase):
    def test_submillisecond_label_transition_is_not_rounded_away(self):
        events = [dict(event_time_ms=0, event_type='set_started', difficulty_level='medium'),
                  dict(event_time_ms=0, event_type='puzzle_started'),
                  dict(event_time_ms=24999.999, event_type='rating_started')]
        segments = v2.segments_for_window(0, 25000, events, -1)
        self.assertEqual(v2.label_window(segments)['exclusion_reason'], 'mixed_cognitive_labels')

    def test_event_at_endpoint_has_no_duration_in_previous_window(self):
        events = [dict(event_time_ms=0, event_type='set_started', difficulty_level='medium'),
                  dict(event_time_ms=0, event_type='puzzle_started'),
                  dict(event_time_ms=25000, event_type='rating_started')]
        segments = v2.segments_for_window(0, 25000, events, -1)
        self.assertEqual(v2.label_window(segments)['cognitive_load_label'], 'Medium')

    def test_phase_policy_does_not_relabel_active_medium_or_high(self):
        for phase, difficulty, expected in [
            ('active_task', 'easy', 'Low'), ('active_task', 'practice', 'Low'),
            ('active_task', 'demo', 'Low'), ('active_task', 'medium', 'Medium'),
            ('active_task', 'hard', 'High'), ('rating', 'hard', 'Low'),
            ('feedback', 'medium', 'Low'), ('pre_task', '', 'Low'),
            ('rest', '', 'Low'), ('post_task', '', 'Low'), ('interrupted', '', 'Low'),
            ('baseline', '', None), ('unlabeled', '', None), ('active_task', '', None),
        ]:
            with self.subTest(phase=phase, difficulty=difficulty):
                self.assertEqual(v2.segment_label({'study_phase': phase, 'difficulty_level': difficulty}), expected)

    def test_same_label_mixed_phases_are_retained(self):
        segments = [dict(study_phase='active_task', difficulty_level='easy', duration_seconds=10),
                    dict(study_phase='rating', difficulty_level='hard', duration_seconds=15)]
        result = v2.label_window(segments)
        self.assertEqual(result['cognitive_load_label'], 'Low')
        self.assertEqual(result['exclusion_reason'], '')
        self.assertEqual(json.loads(result['window_label_mix']), {'Low': 25.0})

    def test_even_a_short_different_label_is_not_a_majority_vote(self):
        segments = [dict(study_phase='active_task', difficulty_level='medium', duration_seconds=24.999),
                    dict(study_phase='rating', difficulty_level='medium', duration_seconds=.001)]
        result = v2.label_window(segments)
        self.assertEqual(result['cognitive_load_label'], '')
        self.assertEqual(result['exclusion_reason'], 'mixed_cognitive_labels')

    def test_unknown_and_empty_intervals_are_excluded(self):
        for segments in ([], [dict(study_phase='unlabeled', duration_seconds=25)],
                         [dict(study_phase='baseline', duration_seconds=1),
                          dict(study_phase='pre_task', duration_seconds=24)]):
            self.assertEqual(v2.label_window(segments)['exclusion_reason'], 'unresolved_or_calibration')


def raw_rows(count=60):
    origin = datetime(2026, 10, 30, tzinfo=timezone.utc)
    return [dict(raw_id=i, packet_id=i, received_at=(origin+timedelta(seconds=i)).isoformat(),
                 elapsed_seconds=i, timestamp_relative=i, rr_original_ms=1000,
                 contact_detected='true', phase='task') for i in range(1, count+1)]


class ReplayTests(unittest.TestCase):
    def test_live_endpoint_ordering_is_audited_not_hidden(self):
        raw = raw_rows(30)
        origin = datetime.fromisoformat(raw[0]['received_at'])-timedelta(seconds=1)
        source = dict(window_end=(origin+timedelta(seconds=25)).isoformat(),
                      rr_count=24, rr_pairs=23, rmssd_ms=0, sdnn_ms=0,
                      coverage=.96, accepted_rr_seconds=24, data_age_seconds=1,
                      status='ready', reason='')
        replay = v2.replay_rr(raw, origin, 30, (25,))[25]
        audit = v2.audit_replay(raw, [source], origin, replay)
        self.assertEqual(audit['exact_replay_windows'], 0)
        self.assertEqual(audit['endpoint_publication_differences'], 1)
        self.assertEqual(audit['unexplained_windows'], 0)
        self.assertEqual(replay[0]['rr_count'], 25)

    def test_task_offset_uses_datetime_subtraction_without_epoch_cancellation(self):
        rows = raw_rows()
        origin = datetime.fromisoformat(rows[0]['received_at'])-timedelta(seconds=1)
        result = v2.replay_rr(rows, origin, 60, (15,25,45))
        self.assertEqual([r['timestamp_relative'] for r in result[25]], list(range(25,61,5)))
        row = result[25][0]
        self.assertEqual(row['rr_count'], 25)
        self.assertEqual(row['rr_pairs'], 24)
        self.assertEqual(row['rmssd_ms'], 0)
        self.assertEqual(row['sdnn_ms'], 0)
        self.assertEqual(row['status'], 'ready')
        self.assertEqual(row['coverage'], 1)

    def test_future_packets_cannot_repair_an_already_closed_window(self):
        rows = raw_rows(30)
        rows[25]['rr_original_ms'] = 1600
        origin = datetime.fromisoformat(rows[0]['received_at'])-timedelta(seconds=1)
        result = v2.replay_rr(rows, origin, 30, (25,))
        self.assertEqual(result[25][0]['rmssd_ms'], 0)
        self.assertGreater(result[25][1]['rmssd_ms'], 0)

    def test_baseline_rows_do_not_enter_task_replay(self):
        rows = raw_rows(45)
        for row in rows[:10]:
            row['phase'] = 'baseline'
        origin = datetime.fromisoformat(rows[0]['received_at'])+timedelta(seconds=9)
        result = v2.replay_rr(rows, origin, 35, (25,))
        self.assertEqual(result[25][0]['rr_count'], 25)
        self.assertEqual(result[25][0]['coverage'], 1)


class ExportTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows permission inheritance')
    def test_bundle_inherits_public_output_parent_not_private_staging_acl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root/'sources')
            public = root/'public'
            public.mkdir()
            subprocess.run(['icacls', str(public), '/grant', '*S-1-1-0:(OI)(CI)RX', '/q'], check=True, capture_output=True)
            v2.build_dataset(root/'sources', public/'bundle')
            for path in (public/'bundle', public/'bundle'/'window_25s_step_5s'/'good_acceptable.csv'):
                command = f"(Get-Acl -LiteralPath '{path}').Access | ForEach-Object {{ $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value }}"
                acl = subprocess.run(['powershell', '-NoProfile', '-Command', command], check=True, capture_output=True, text=True)
                self.assertIn('S-1-1-0', acl.stdout)

    def test_v1_cannot_overwrite_its_archived_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root/'sources'/'Andrean')
            archive = root/'v1_active_task'
            archive.mkdir()
            (archive/'sentinel').write_bytes(b'old bytes')
            with self.assertRaises(FileExistsError):
                v1.build_dataset(root/'sources', archive)
            self.assertEqual(list(archive.iterdir()), [archive/'sentinel'])

    def source(self, root):
        directory = fixture(root/'Andrean')
        origin = pd.Timestamp(BASE, unit='ms', tz='UTC').to_pydatetime()
        eye = pd.read_csv(directory/'eye_tracking.csv')
        extra = eye.iloc[-600:].copy()
        extra['timestamp_ms'] += 30000
        pd.concat([eye, extra]).to_csv(directory/'eye_tracking.csv', index=False)
        cached = pd.read_csv(directory/'hrv.csv', keep_default_na=False)
        cached.loc[cached.phase.eq('task'), ['rr_count','rr_pairs','rmssd_ms','sdnn_ms','coverage','reason']] = [25,24,0.,0.,1.,'']
        cached['accepted_rr_seconds'] = np.nan
        cached.loc[cached.phase.eq('task'), ['accepted_rr_seconds','data_age_seconds']] = [25.,0.]
        cached.to_csv(directory/'hrv.csv', index=False)
        rows = raw_rows(120)
        for i, row in enumerate(rows, 1):
            row['received_at'] = (origin+timedelta(seconds=i)).isoformat()
            row['session_id'] = 'source-Andrean'
        pd.DataFrame(rows).to_csv(directory/'rr_raw.csv', index=False)
        return directory

    def test_build_preserves_source_and_exports_all_windows_with_one_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root/'sources')
            before = v1.snapshot(root/'sources')
            manifest = v2.build_dataset(root/'sources', root/'output')
            self.assertTrue(manifest['source_integrity_verified'])
            self.assertEqual(v1.snapshot(root/'sources'), before)
            for seconds in (15,25,45):
                path = root/'output'/f'window_{seconds}s_step_5s'
                data = pd.read_csv(path/'good_acceptable.csv')
                self.assertFalse(data.empty)
                self.assertEqual(set(data.cognitive_load_label), {'Low','Medium','High'} if seconds==15 else {'Low','Medium','High'} if seconds==25 else {'High'})
                self.assertTrue(data.window_id.is_unique)
                self.assertTrue(np.isfinite(data[v1.FEATURE_COLUMNS].select_dtypes('number')).all().all())
                allowlist = json.loads((path/'feature_columns.json').read_text())
                self.assertEqual(allowlist['target'], 'cognitive_load_label')
                self.assertNotIn('difficulty_level', allowlist['features'])
                dictionary = pd.read_csv(path/'data_dictionary.csv')
                self.assertEqual(dictionary.loc[dictionary.role.eq('target'), 'column'].tolist(), ['cognitive_load_label'])
            self.assertTrue((root/'output'/'manifest.json').exists())

    def test_existing_bundle_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root/'sources')
            (root/'output').mkdir()
            (root/'output'/'sentinel').write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                v2.build_dataset(root/'sources', root/'output')
            self.assertEqual((root/'output'/'sentinel').read_bytes(), b'existing')

    def test_missing_readiness_is_not_converted_to_pretask_low(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = self.source(Path(temp)/'sources')
            (directory/'hrv_calib_status.json').unlink()
            with self.assertRaisesRegex(ValueError, 'calibration'):
                v2.build_dataset(directory.parent, Path(temp)/'output')

    def test_ready_but_low_coverage_baseline_does_not_make_a_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = self.source(Path(temp)/'sources')
            hrv = pd.read_csv(directory/'hrv.csv')
            hrv.loc[hrv.phase.eq('baseline'), 'coverage'] = .7
            hrv.to_csv(directory/'hrv.csv', index=False)
            results, _ = v2.build_session(directory, v1.load_config())
            row = results[25][0][0]
            self.assertTrue(row['baseline_recorder_ready'])
            self.assertFalse(row['baseline_usable'])
            self.assertEqual(row['baseline_rejection_reason'], 'insufficient_baseline_coverage')
            self.assertTrue(np.isnan(row['rmssd_baseline_ratio']))
            self.assertEqual(row['rmssd_ms'], 0.)

    def test_unexplained_cached_hrv_difference_stops_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.source(root/'sources')
            hrv = pd.read_csv(directory/'hrv.csv')
            hrv.loc[hrv.phase.eq('task'), 'rmssd_ms'] = 999
            hrv.to_csv(directory/'hrv.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Unexplained'):
                v2.build_dataset(root/'sources', root/'output')
            self.assertFalse((root/'output').exists())

    def test_reversed_cached_window_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.source(root/'sources')
            hrv = pd.read_csv(directory/'hrv.csv')
            pd.concat([hrv.iloc[:1], hrv.iloc[1:].iloc[::-1]]).to_csv(directory/'hrv.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'grid'):
                v2.build_dataset(root/'sources', root/'output')
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
