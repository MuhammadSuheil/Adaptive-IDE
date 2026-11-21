"""Conservative, label-blind RR window exclusion and unchanged-source refresh."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset as shared
from analysis_sessions.dataset_fusion import build_dataset_v2 as builder
from analysis_sessions.dataset_fusion import test_dataset_v2 as fixtures
from analysis_sessions.dataset_fusion import v2_features as features


class RRQualityTests(unittest.TestCase):
    def module(self):
        from analysis_sessions.dataset_fusion import v2_rr_quality
        return v2_rr_quality

    def source(self, root, jump=1600):
        rows = fixtures.raw_rows(40)
        for row in rows:
            row['session_id'] = 'test'
        rows[7]['rr_original_ms'] = jump
        root.mkdir(parents=True, exist_ok=True)
        with (root/'rr_raw.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_isolated_jump_is_audited_without_changing_source_or_live_filter(self):
        from hrv_monitor_prototype import config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root)
            before = shared.snapshot(root)
            live = config.FILTER_ENABLED
            audit = self.module().audit_session(root)
            self.assertEqual([r['raw_id'] for r in audit['candidates']], [8])
            self.assertEqual(audit['candidates'][0]['reference_median_ms'], 1000)
            self.assertEqual(audit['candidates'][0]['deviation_ms'], 600)
            self.assertEqual(before, shared.snapshot(root))
            self.assertEqual(config.FILTER_ENABLED, live)

    def test_exact_threshold_and_insufficient_reference_are_not_called_jumps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root, jump=1250)
            self.assertEqual(self.module().audit_session(root)['candidates'], [])

    def test_no_median_is_invented_before_five_reference_rr(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.source(root, jump=1000)
            rows[2]['rr_original_ms'] = 1600
            pd.DataFrame(rows).to_csv(root/'rr_raw.csv', index=False)
            self.assertEqual(self.module().audit_session(root)['candidates'], [])

    def test_rejected_out_of_range_rr_is_not_counted_as_an_accepted_jump(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root, jump=2500)
            self.assertEqual(self.module().audit_session(root)['candidates'], [])

    def test_membership_uses_open_start_closed_end_and_packet_availability(self):
        mod = self.module()
        origin = '2026-10-30T00:00:'
        frame = pd.DataFrame([
            {'window_id': 'at_end', 'session_id': 'test', 'window_start': origin+'00+00:00', 'window_end': origin+'10+00:00'},
            {'window_id': 'at_start', 'session_id': 'test', 'window_start': origin+'10+00:00', 'window_end': origin+'25+00:00'},
        ])
        audit = {'session_id': 'test', 'candidates': [
            {'beat_time': origin+'10+00:00', 'available_at': origin+'10+00:00'}]}
        self.assertEqual(mod.affected_windows(frame, audit), {'at_end'})
        audit['candidates'][0]['available_at'] = origin+'10.000001+00:00'
        self.assertEqual(mod.affected_windows(frame, audit), set())
        audit['session_id'] = 'another'
        self.assertEqual(mod.affected_windows(frame, audit), set())

    def test_marker_is_idempotent_preserves_other_reasons_and_is_label_blind(self):
        mod = self.module()
        frame = pd.DataFrame([{'window_id': 'x', 'session_id': 'test',
                               'window_start': '2026-10-30T00:00:00+00:00',
                               'window_end': '2026-10-30T00:00:25+00:00',
                               'hrv_reason': 'existing_reason', 'cognitive_load_label': 'Low', 'rmssd_ms': 100}])
        audit = {'session_id': 'test', 'candidates': [
            {'beat_time': '2026-10-30T00:00:10+00:00', 'available_at': '2026-10-30T00:00:10+00:00'}]}
        selected = mod.apply_window_policy(frame, audit)
        self.assertEqual(selected.hrv_reason.tolist(), ['existing_reason;rr_local_median_jump'])
        pd.testing.assert_frame_equal(selected, mod.apply_window_policy(selected, audit))
        self.assertEqual(selected.rmssd_ms.tolist(), [100])
        for label in ('Medium', 'High'):
            self.assertEqual(mod.apply_window_policy(frame.assign(cognitive_load_label=label), audit).hrv_reason.tolist(),
                             selected.hrv_reason.tolist())

    def test_refresh_matches_full_rebuild_without_touching_raw_or_predictors(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root/'sources')
            first = builder.build_dataset(root/'sources', root/'bundle', admission_policy=features.POLICY)
            self.assertEqual(first['rr_quality_control']['policy'], mod.POLICY)
            before = shared.snapshot(root/'sources')
            hashes = shared.snapshot(root/'bundle')
            refreshed = mod.refresh_bundle(root/'bundle')
            self.assertEqual(refreshed['datasets'], first['datasets'])
            self.assertEqual(before, shared.snapshot(root/'sources'))
            for name, digest in hashes.items():
                if name.endswith(('.csv', 'feature_columns.json')):
                    self.assertEqual(shared.digest(root/'bundle'/name), digest, name)
            builder.verify_bundle(root/'bundle')

    def test_real_build_excludes_jump_windows_and_preserves_original_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = fixtures.ExportTests().source(root/'sources')
            raw = pd.read_csv(directory/'rr_raw.csv', keep_default_na=False)
            raw.loc[15, 'rr_original_ms'] = 1600
            raw.to_csv(directory/'rr_raw.csv', index=False)
            cached = pd.read_csv(directory/'hrv.csv', keep_default_na=False)
            origin = pd.Timestamp(cached.loc[cached.phase.eq('task'), 'window_start'].iloc[0]).to_pydatetime()
            replay = builder.replay_rr(raw.to_dict('records'), origin, 120, (25,))[25]
            replay = {pd.Timestamp(row['window_end']): row for row in replay}
            cached_rows = cached.to_dict('records')
            for row in cached_rows:
                if row['phase'] != 'task':
                    continue
                actual = replay[pd.Timestamp(row['window_end'])]
                for key in ('rmssd_ms', 'sdnn_ms', 'rr_count', 'rr_pairs', 'coverage',
                            'accepted_rr_seconds', 'data_age_seconds', 'status', 'reason'):
                    row[key] = actual[key]
            pd.DataFrame(cached_rows).to_csv(directory/'hrv.csv', index=False)
            sources = shared.snapshot(root/'sources')
            builder.build_dataset(root/'sources', root/'bundle')
            before = pd.read_csv(root/'bundle/window_25s_step_5s/all_data.csv')
            builder.build_dataset(root/'sources', root/'bundle', replace=True, admission_policy=features.POLICY)
            frame = pd.read_csv(root/'bundle/window_25s_step_5s/all_data.csv')
            curated = pd.read_csv(root/'bundle/window_25s_step_5s/good_acceptable.csv')
            marked = frame.hrv_reason.fillna('').str.contains('rr_local_median_jump')
            self.assertGreater(int(marked.sum()), 0)
            self.assertEqual(set(frame.loc[marked, 'quality_tier']), {'poor'})
            self.assertTrue(set(curated.window_id).isdisjoint(frame.loc[marked, 'window_id']))
            pd.testing.assert_frame_equal(frame[shared.FEATURE_COLUMNS], before[shared.FEATURE_COLUMNS])
            self.assertEqual(shared.snapshot(root/'sources'), sources)
            self.module().refresh_bundle(root/'bundle')
            builder.verify_bundle(root/'bundle')

    def test_refresh_rejects_changed_raw_source_before_publishing(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = fixtures.ExportTests().source(root/'sources')
            builder.build_dataset(root/'sources', root/'bundle', admission_policy=features.POLICY)
            before = shared.snapshot(root/'bundle')
            with (folder/'rr_raw.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                mod.refresh_bundle(root/'bundle')
            self.assertEqual(shared.snapshot(root/'bundle'), before)

    def test_rehashed_rr_marker_removal_is_rejected(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root/'sources')
            builder.build_dataset(root/'sources', root/'bundle', admission_policy=features.POLICY)
            manifest_path = root/'bundle/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            registry = manifest['rr_quality_control']
            frame = pd.read_csv(root/'bundle/window_25s_step_5s/all_data.csv')
            first = frame.iloc[0]
            audit = copy.deepcopy(next(iter(registry['sessions'].values())))
            audit['candidates'] = [{'raw_id': 99, 'packet_id': 99, 'beat_time': first.window_end,
                                    'available_at': first.window_end, 'rr_original_ms': 1600.,
                                    'reference_median_ms': 1000., 'deviation_ms': 600.}]
            registry['sessions'][first.session_id] = audit
            with self.assertRaisesRegex(ValueError, 'RR window'):
                mod.verify_frame(frame, registry, manifest['source_sha256'])


if __name__ == '__main__':
    unittest.main()
