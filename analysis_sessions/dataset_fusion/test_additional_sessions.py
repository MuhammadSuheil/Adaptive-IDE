"""Additional acquisitions retain identities and explicit source exclusions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis_sessions import session_inventory as inventory
from analysis_sessions.dataset_fusion import build_dataset as v1
from analysis_sessions.dataset_fusion import build_dataset_v2 as v2
from analysis_sessions.dataset_fusion import test_dataset_v2 as fixtures


class IdentityTests(unittest.TestCase):
    def test_two_bangsur_recordings_are_one_person_but_ihsan_is_not_ikhsan(self):
        self.assertEqual(inventory.participant_group('bangsur_easy_only'), 'bangsur')
        self.assertEqual(inventory.participant_group('bangsur_easy_medium'), 'bangsur')
        self.assertNotEqual(inventory.participant_group('ihsan'), inventory.participant_group('ikhsan'))
        self.assertEqual(inventory.participant_group('Ramadhan-2'), 'Ramadhan')

    def test_exclusion_is_by_session_id_even_after_folder_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder, session in [('renamed_folder', 'unconnected'), ('complete', 'connected')]:
                directory = root / folder
                directory.mkdir()
                pd.DataFrame({'session_id': [session]}).to_csv(directory / 'eye_tracking.csv', index=False)
            before = v1.snapshot(root)
            chosen, entries = inventory.select_session_directories(root, session_exclusions={'unconnected': 'missing_task_events'})
            self.assertEqual([p.name for p in chosen], ['complete'])
            excluded = next(r for r in entries if r['session_id'] == 'unconnected')
            self.assertFalse(excluded['selected'])
            self.assertIn('missing_task_events', excluded['reason'])
            self.assertEqual(before, v1.snapshot(root))
            with self.assertRaisesRegex(ValueError, 'excluded|Excluded'):
                inventory.select_session_directories(root, ['renamed_folder'], session_exclusions={'unconnected': 'missing_task_events'})


class ConfigurationTests(unittest.TestCase):
    def test_old_config_still_loads_and_optional_exclusions_require_reasons(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            config = v1.read_json(v1.DEFAULT_CONFIG)
            config.pop('session_exclusions', None)
            path.write_text(json.dumps(config))
            self.assertNotIn('session_exclusions', v1.load_config(path))
            config['session_exclusions'] = {'session-x': 'operator did not connect CodeArrange'}
            path.write_text(json.dumps(config))
            self.assertEqual(v1.load_config(path)['session_exclusions']['session-x'], 'operator did not connect CodeArrange')
            for invalid in ([], {'': 'reason'}, {'session-x': ''}, {'session-x': 123}):
                config['session_exclusions'] = invalid
                path.write_text(json.dumps(config))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    v1.load_config(path)


class BuildExclusionTests(unittest.TestCase):
    def test_output_index_uses_permission_preserving_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root / 'sources')
            archive = root / 'v1_active_task'
            archive.mkdir()
            pd.DataFrame({'difficulty_level': ['easy'], 'participant_id': ['old']}).to_csv(
                archive / 'good_acceptable.csv', index=False)
            with patch.object(v1, 'publish_artifact', wraps=v1.publish_artifact) as publish:
                v2.build_dataset(root / 'sources', root / 'v2_session_phase')
            publish.assert_called_once()
            self.assertIn('Pilihan Dataset', (root / 'README.md').read_text(encoding='utf-8'))

    def test_update_summary_counts_only_new_recordings_and_preserves_exclusions(self):
        from analysis_sessions.dataset_fusion.v2_documentation import (
            build_update_summary,
        )
        results, counts, previous_counts = {}, {}, {}
        for seconds in (15, 25, 45):
            rows = []
            for session in ('old', 'new'):
                for minute, label in ((1, 'Low'), (2, 'Low'), (3, 'Medium')):
                    end = pd.Timestamp('2026-01-01T00:00:00Z') + pd.Timedelta(minutes=minute)
                    rows.append({'session_id': session, 'session_folder': session, 'participant_id': session,
                                 'window_start': (end-pd.Timedelta(seconds=seconds)).isoformat(),
                                 'window_end': end.isoformat(), 'quality_tier': 'good', 'cognitive_load_label': label})
            results[seconds] = (rows, [], [{'session_id': n, 'participant_id': n, 'session_folder': n} for n in ('old', 'new')])
            name = f'window_{seconds}s_step_5s'
            counts[name] = {'good_acceptable': {'Low': 4, 'Medium': 2, 'High': 0, 'windows': 6}}
            previous_counts[name] = {'good_acceptable': {'Low': 2, 'Medium': 1, 'High': 0, 'windows': 3}}
        previous = {'selected_session_ids': ['old'], 'datasets': previous_counts}
        excluded = {'session_id': 'missing', 'folder': 'missing', 'selected': False, 'reason': 'excluded_session: missing_task_events'}
        summary = build_update_summary(results, counts, [excluded], previous)
        self.assertEqual(summary['additional_session_ids'], ['new'])
        self.assertEqual(summary['windows']['15']['added'], {'Low': 2, 'Medium': 1, 'High': 0, 'windows': 3})
        self.assertEqual(summary['introduced_recordings'][0]['windows']['15']['unique_minutes']['Low'], .5)
        self.assertEqual(summary['excluded_sessions'][0]['session_id'], 'missing')

    def test_explicitly_excluded_recording_stays_in_inventory_not_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures.ExportTests().source(root / 'sources')
            excluded = root / 'sources' / 'bangsur_easy_only'
            excluded.mkdir()
            pd.DataFrame({'session_id': ['unconnected']}).to_csv(excluded / 'eye_tracking.csv', index=False)
            config = v1.read_json(v1.DEFAULT_CONFIG)
            config['session_exclusions'] = {'unconnected': 'missing_task_events'}
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            before = v1.snapshot(root / 'sources')
            manifest = v2.build_dataset(root / 'sources', root / 'output', path)
            self.assertEqual(before, v1.snapshot(root / 'sources'))
            self.assertNotIn('bangsur_easy_only', manifest['selected_folders'])
            self.assertEqual(manifest['excluded_sessions'][0]['session_id'], 'unconnected')
            for seconds in (15, 25, 45):
                output = root / 'output' / f'window_{seconds}s_step_5s'
                data = pd.read_csv(output / 'all_data.csv')
                self.assertNotIn('unconnected', set(data.session_id))
                entries = pd.read_csv(output / 'session_inventory.csv')
                self.assertFalse(entries.loc[entries.session_id.eq('unconnected'), 'selected'].iloc[0])

    def test_missing_unexcluded_ledger_fails_before_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fixtures.ExportTests().source(root / 'sources')
            (source / 'task_events.csv').unlink()
            with self.assertRaisesRegex((FileNotFoundError, ValueError), 'task_events'):
                v2.build_dataset(root / 'sources', root / 'output')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
