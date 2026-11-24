"""Behavioral-proxy labels: observed intervals, abstention, and train-only norms."""

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def event(time, kind, **changes):
    result = {'event_id': f'{kind}-{time}', 'session_id': 's', 'event_time_ms': time,
                  'event_type': kind, 'puzzle_set': 'set', 'puzzle_id': 'q', 'puzzle_title': '1. Variabel',
                  'attempt_id': '', 'submission_type': '', 'outcome': ''}
    result.update(changes)
    return result


def episode(person, seconds, errors=0, completed=True, **changes):
    result = {'episode_id': person, 'participant_id': person, 'session_id': 's-'+person,
                  'session_folder': person, 'set_instance_id': 'set', 'puzzle_set': 'set', 'puzzle_id': 'q',
                  'puzzle_title': 'Variabel', 'title_key': 'variabel', 'item_key': '["set","q","variabel"]',
                  'start_ms': 1000., 'end_ms': seconds*1000+1000., 'active_seconds': float(seconds),
                  'manual_errors': errors, 'manual_submissions': errors+int(completed),
                  'completed': completed, 'success_submission_type': 'manual' if completed else '',
                  'closed_by': 'correct' if completed else 'set_ended', 'censored': not completed,
                  'valid': True, 'invalid_reasons': [], 'intervals': [[1000.,seconds*1000+1000.]],
                  'attempt_ids': [], 'first_exposure': True}
    result.update(changes)
    return result


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        try:
            self.behavior = importlib.import_module('analysis_sessions.dataset_fusion.v3_behavior')
        except ModuleNotFoundError:
            self.fail('V3 behavioral reconstruction and labeling are not implemented.')

    def reconstruct(self, events):
        return self.behavior.reconstruct(events, 'p', 's', 'folder')

    def test_retry_excludes_feedback_and_rating_time(self):
        rows, _ = self.reconstruct([event(0,'set_started'), event(1000,'puzzle_started'),
            event(11000,'attempt_submitted',attempt_id='a',submission_type='manual',outcome='incorrect'),
            event(12000,'rating_started'), event(20000,'rating_ended'), event(21000,'puzzle_started'),
            event(31000,'attempt_submitted',attempt_id='b',submission_type='manual',outcome='correct'),
            event(40000,'set_ended')])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['active_seconds'],20.)
        self.assertEqual(rows[0]['manual_errors'],1)
        self.assertTrue(rows[0]['completed'])
        self.assertTrue(rows[0]['valid'])
        self.assertEqual(rows[0]['intervals'],[[1000.,11000.],[21000.,31000.]])

    def test_navigation_does_not_assign_other_question_time(self):
        rows, _ = self.reconstruct([event(0,'set_started'), event(1000,'puzzle_started'),
            event(10000,'puzzle_started',puzzle_id='b'),
            event(20000,'attempt_submitted',puzzle_id='b',attempt_id='a',submission_type='manual',outcome='correct'),
            event(30000,'puzzle_started'),
            event(40000,'attempt_submitted',attempt_id='b',submission_type='manual',outcome='correct')])
        self.assertEqual({r['puzzle_id']:r['active_seconds'] for r in rows},{'q':19.,'b':10.})

    def test_opening_completed_question_creates_new_episode(self):
        rows, _ = self.reconstruct([event(0,'set_started'),event(1000,'puzzle_started'),
            event(11000,'attempt_submitted',attempt_id='a',submission_type='manual',outcome='correct'),
            event(21000,'puzzle_started'),
            event(31000,'attempt_submitted',attempt_id='b',submission_type='manual',outcome='correct')])
        self.assertEqual(len(rows),2)
        self.assertNotEqual(rows[0]['episode_id'],rows[1]['episode_id'])

    def test_duplicate_attempt_is_not_counted_twice(self):
        attempt=event(11000,'attempt_submitted',attempt_id='a',submission_type='manual',outcome='incorrect')
        rows, audit = self.reconstruct([event(0,'set_started'),event(1000,'puzzle_started'),attempt,
            dict(attempt,event_id='retry-copy'),event(21000,'set_ended')])
        self.assertEqual(rows[0]['manual_errors'],1)
        self.assertEqual(audit['duplicate_attempts'],1)
        self.assertTrue(rows[0]['valid'])

    def test_conflicting_attempt_id_invalidates_episode(self):
        attempt=event(11000,'attempt_submitted',attempt_id='a',submission_type='manual',outcome='incorrect')
        rows, audit = self.reconstruct([event(0,'set_started'),event(1000,'puzzle_started'),attempt,
            dict(attempt,event_id='conflict',outcome='correct'),event(21000,'set_ended')])
        self.assertFalse(rows[0]['valid'])
        self.assertIn('conflicting_attempt_id',rows[0]['invalid_reasons'])
        self.assertEqual(audit['conflicting_attempt_ids'],['a'])

    def test_missing_episode_end_is_not_invented(self):
        rows, _ = self.reconstruct([event(0,'set_started'),event(1000,'puzzle_started')])
        self.assertFalse(rows[0]['valid'])
        self.assertIn('missing_episode_end',rows[0]['invalid_reasons'])
        self.assertEqual(rows[0]['active_seconds'],0.)

    def test_mismatched_question_or_session_is_rejected(self):
        rows, _ = self.reconstruct([event(0,'set_started'),event(1000,'puzzle_started'),
            event(11000,'attempt_submitted',puzzle_id='other',attempt_id='a',submission_type='manual',outcome='correct'),
            event(21000,'set_ended')])
        self.assertFalse(rows[0]['valid'])
        with self.assertRaisesRegex(ValueError,'session'):
            self.reconstruct([event(0,'set_started',session_id='different')])

    def test_title_normalization_preserves_distinct_items(self):
        self.assertEqual(self.behavior.normalize_title('  12.  Variabel  Dasar '),'variabel dasar')
        self.assertEqual(self.behavior.normalize_title('２) Variabel Dasar'),'variabel dasar')
        self.assertNotEqual(self.behavior.normalize_title('Merge Array'),self.behavior.normalize_title('Merge Sort'))

    def test_split_is_deterministic_and_groups_people(self):
        people=[f'p{i:02}' for i in range(22)]
        a=self.behavior.partition(people)
        b=self.behavior.partition(list(reversed(people))+['p00'])
        self.assertEqual(a,b)
        self.assertEqual({s:list(a.values()).count(s) for s in ('Train','validation','Test')},
                         {'Train':16,'validation':3,'Test':3})

    def reference_fixture(self):
        rows=[episode(f'p{i}',seconds) for i,seconds in enumerate((10,20,30,40,50))]
        splits={r['participant_id']:'Train' for r in rows}
        splits['query']='Test'
        return rows,splits

    def test_train_only_reference_excludes_query_person_and_later_exposures(self):
        rows,splits=self.reference_fixture()
        rows += [episode('heldout',999),episode('p0',999,episode_id='later',start_ms=999000.)]
        splits['heldout']='validation'
        reference=self.behavior.make_reference(rows,splits)
        q=self.behavior.reference_for(reference,episode('query',30))
        self.assertEqual(q['participants'],['p0','p1','p2','p3','p4'])
        self.assertAlmostEqual(q['lower_seconds'],70/3)
        self.assertAlmostEqual(q['upper_seconds'],110/3)
        own=self.behavior.reference_for(reference,episode('p0',30))
        self.assertEqual(own['reason'],'insufficient_reference')
        changed=copy.deepcopy(rows)
        changed[-2]['active_seconds']=1
        self.assertEqual(self.behavior.make_reference(changed,splits),reference)

    def test_unsolved_first_exposure_cannot_be_replaced_with_later_success(self):
        rows,splits=self.reference_fixture()
        rows[0]['completed']=False
        rows.append(episode('p0',10,episode_id='later',start_ms=200000.))
        ref=self.behavior.make_reference(rows,splits)
        self.assertEqual(self.behavior.reference_for(ref,episode('query',10))['reason'],'insufficient_reference')

    def test_rubric_requires_both_indicators_and_preserves_censoring(self):
        rows,splits=self.reference_fixture()
        ref=self.behavior.make_reference(rows,splits)
        for seconds,errors,completed,expected in [(10,0,True,'Low'),(30,1,True,'Medium'),
            (30,2,True,'Medium'),(50,3,True,'High'),(50,3,False,'High'),
            (10,0,False,''),(50,0,True,''),(10,3,True,''),(30,0,True,''),(30,1,False,'')]:
            with self.subTest(seconds=seconds,errors=errors,completed=completed):
                result=self.behavior.label_episode(episode('query',seconds,errors,completed),ref)
                self.assertEqual(result['cognitive_load_label'],expected)

    def test_quantile_boundary_and_degenerate_reference(self):
        rows,splits=self.reference_fixture()
        ref=self.behavior.make_reference(rows,splits)
        limits=self.behavior.reference_for(ref,episode('query',10))
        self.assertEqual(self.behavior.label_episode(episode('query',limits['lower_seconds']),ref)['cognitive_load_label'],'Low')
        self.assertEqual(self.behavior.label_episode(episode('query',limits['upper_seconds'],1),ref)['cognitive_load_label'],'Medium')
        for r in rows:r['active_seconds']=10.
        self.assertEqual(self.behavior.label_episode(episode('query',10),self.behavior.make_reference(rows,splits))['label_reason'],'degenerate_reference')

    def test_invalid_timeout_and_sensor_fields_cannot_force_a_label(self):
        rows,splits=self.reference_fixture();ref=self.behavior.make_reference(rows,splits)
        query=episode('query',100,0,False,closed_by='timeout',rating=10,difficulty_level='hard',rmssd_ms=.1)
        self.assertEqual(self.behavior.label_episode(query,ref)['cognitive_load_label'],'')
        good=episode('query',10)
        before=self.behavior.label_episode(good,ref)
        good.update(rmssd_ms=9999,difficulty_level='hard',rating=10)
        self.assertEqual(self.behavior.label_episode(good,ref),before)

    def test_window_must_fit_one_active_episode_and_not_bridge_feedback(self):
        a=episode('p',20,intervals=[[0.,10000.],[20000.,30000.]])
        index=self.behavior.WindowIndex([a])
        self.assertEqual(index.match('s-p',0,10000)[0]['episode_id'],'p')
        self.assertIsNone(index.match('s-p',5000,25000)[0])
        self.assertIsNone(index.match('s-p',10000,20000)[0])
        b=episode('p',20,intervals=[[0.,10000.],[10000.,20000.]])
        self.assertIsNotNone(self.behavior.WindowIndex([b]).match('s-p',0,20000)[0])


class BundleTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('analysis_sessions.dataset_fusion.build_dataset_v3')
        except ModuleNotFoundError:
            self.fail('V3 bundle export and verification are not implemented.')

    def test_complete_bundle_preserves_sources_and_can_be_verified_and_rebuilt(self):
        module=self.module()
        from datetime import timedelta

        import numpy as np
        import pandas as pd

        from analysis_sessions.dataset_fusion import build_dataset as shared
        from analysis_sessions.dataset_fusion import build_dataset_v2 as v2
        from analysis_sessions.dataset_fusion import v3_behavior as behavior
        from analysis_sessions.dataset_fusion.test_dataset_fusion import (
            BASE,
            eye_source,
            fixture,
        )
        from analysis_sessions.dataset_fusion.test_dataset_v2 import raw_rows

        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            people=[f'p{i:02}' for i in range(8)]
            split=behavior.partition(people)
            train=sorted(p for p in people if split[p]=='Train')
            timings={p:(seconds,errors) for p,seconds,errors in zip(train,[30,45,60,90,120,180],[0,0,1,1,3,3])}
            for person in people:
                seconds,errors=timings.get(person,(90,1))
                directory=fixture(root/'raw'/person)
                eye=eye_source(np.arange(0,240001,50)).assign(session_id=f'source-{person}')
                eye.to_csv(directory/'eye_tracking.csv',index=False)
                origin=pd.Timestamp(BASE,unit='ms',tz='UTC').to_pydatetime()
                raw=raw_rows(240)
                for i,row in enumerate(raw,1):
                    row.update(session_id=f'source-{person}',received_at=(origin+timedelta(seconds=i)).isoformat())
                pd.DataFrame(raw).to_csv(directory/'rr_raw.csv',index=False)
                hrv=pd.read_csv(directory/'hrv.csv',keep_default_na=False)
                hrv.loc[hrv.phase.eq('task'),['rr_count','rr_pairs','rmssd_ms','sdnn_ms','coverage','reason']]=[25,24,0.,0.,1.,'']
                hrv['accepted_rr_seconds']=np.nan
                hrv.loc[hrv.phase.eq('task'),['accepted_rr_seconds','data_age_seconds']]=[25.,0.]
                hrv.to_csv(directory/'hrv.csv',index=False)
                ev=[event(BASE,'set_started'),event(BASE,'puzzle_started')]
                used=0.;time=BASE
                for i in range(errors):
                    time+=15000;used+=15
                    ev.append(event(time,'attempt_submitted',attempt_id=f'{person}-{i}',submission_type='manual',outcome='incorrect'))
                    time+=1000;ev.append(event(time,'puzzle_started'))
                time+=(seconds-used)*1000
                ev.extend([event(time,'attempt_submitted',attempt_id=f'{person}-ok',submission_type='manual',outcome='correct'),
                           event(time+1000,'set_ended'),event(time+2000,'study_ended')])
                for e in ev:
                    e.update(session_id=f'source-{person}',puzzle_set='easy',difficulty_level='easy')
                pd.DataFrame(ev).to_csv(directory/'task_events.csv',index=False)
            v2.build_dataset(root/'raw',root/'v2',admission_policy='feature_aware_quality_v1')
            before_raw=shared.snapshot(root/'raw');before_v2=shared.snapshot(root/'v2')
            manifest=module.build_dataset(root/'v2',root/'v3')
            self.assertEqual(manifest['label_policy']['version'],'behavioral_concordance_v2')
            self.assertEqual(module.verify_bundle(root/'v3'),manifest)
            for seconds in (15,25,45):
                folder=root/f'v3/window_{seconds}s_step_5s'
                data=pd.read_csv(folder/'good_acceptable.csv')
                self.assertEqual(set(data.cognitive_load_label),{'Low','Medium','High'})
                self.assertTrue(np.isfinite(data[manifest['feature_columns']].to_numpy(float)).all())
                self.assertTrue(data[shared.FEATURE_COLUMNS[-4:]].isna().any().any())
                original=pd.read_csv(root/f'v2/window_{seconds}s_step_5s/all_data.csv').set_index('window_id')
                archive=pd.read_csv(folder/'all_data.csv').set_index('window_id')
                pd.testing.assert_frame_equal(archive[shared.FEATURE_COLUMNS],original.loc[archive.index,shared.FEATURE_COLUMNS])
                contract=json.loads((folder/'feature_columns.json').read_text())
                self.assertTrue(contract['requires_fixed_participant_splits'])
                self.assertNotIn('manual_errors',contract['features'])
                weights=pd.read_csv(folder/'train_sample_weights.csv')
                balanced=pd.read_csv(folder/'train_balanced.csv')
                mapping=pd.read_csv(folder/'window_assignments.csv').set_index('window_id')
                self.assertTrue(mapping.loc[weights.window_id,'split'].eq('Train').all())
                self.assertTrue(mapping.loc[balanced.window_id,'split'].eq('Train').all())
                self.assertTrue(balanced.window_id.is_unique)
                self.assertEqual(len(set(balanced.cognitive_load_label.value_counts())),1)
                for total in weights.groupby('cognitive_load_label').sample_weight.sum():
                    self.assertAlmostEqual(total,len(weights)/3)
            self.assertEqual(shared.snapshot(root/'raw'),before_raw)
            self.assertEqual(shared.snapshot(root/'v2'),before_v2)
            with self.assertRaises(FileExistsError):module.build_dataset(root/'v2',root/'v3')
            # A valid newer v2 source must allow explicit v3 replacement, not strand the old bundle.
            updated=json.loads((root/'v2/manifest.json').read_text())
            updated['generated_at_utc']='2030-01-01T00:00:00+00:00'
            (root/'v2/manifest.json').write_text(json.dumps(updated),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'source bundle changed'):
                module.verify_bundle(root/'v3')
            again=module.build_dataset(root/'v2',root/'v3',replace=True)
            self.assertEqual(again['datasets'],manifest['datasets'])
            self.assertTrue((root/'v3/revision_changes.csv').exists())
            changes=pd.read_csv(root/'v3/revision_changes.csv')
            self.assertFalse(changes.label_changed.any())
            # A forged output checksum must not legitimize altered sampling weights.
            weight_path=root/'v3/window_15s_step_5s/train_sample_weights.csv'
            weight_bytes=weight_path.read_bytes()
            altered_weights=pd.read_csv(weight_path)
            altered_weights.loc[0,'sample_weight']*=2
            altered_weights.to_csv(weight_path,index=False)
            signed=json.loads((root/'v3/manifest.json').read_text())
            signed['output_sha256'][weight_path.relative_to(root/'v3').as_posix()]=shared.digest(weight_path)
            (root/'v3/manifest.json').write_text(json.dumps(signed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'derived artifact'):
                module.verify_bundle(root/'v3')
            weight_path.write_bytes(weight_bytes)
            (root/'v3/manifest.json').write_text(json.dumps(again),encoding='utf-8')
            # Checksums alone must not allow a relabeled artifact to bypass verification.
            path=root/'v3/window_15s_step_5s/good_acceptable.csv'
            altered=pd.read_csv(path);altered.loc[0,'cognitive_load_label']='High'
            altered.to_csv(path,index=False)
            signed=json.loads((root/'v3/manifest.json').read_text())
            signed['output_sha256'][path.relative_to(root/'v3').as_posix()]=shared.digest(path)
            (root/'v3/manifest.json').write_text(json.dumps(signed),encoding='utf-8')
            with self.assertRaises(ValueError):module.verify_bundle(root/'v3')

    def test_failed_publication_restores_the_previous_v3_bundle(self):
        module=self.module()
        # This test uses the publisher's real filesystem moves, failing only its final check.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'v3';staged=root/'stage'
            old.mkdir();staged.mkdir()
            (old/'sentinel').write_text('old',encoding='utf-8')
            (staged/'sentinel').write_text('new',encoding='utf-8')
            def check(path,**_kwargs):
                if Path(path)==old and (old/'sentinel').read_text()=='new':
                    raise ValueError('Simulated final verification failure')
                return {'dataset_version':3}
            with patch.object(module,'verify_bundle',side_effect=check), self.assertRaisesRegex(ValueError,'Simulated'):
                module.publish(staged,old,replace=True)
            self.assertEqual((old/'sentinel').read_text(),'old')
            self.assertEqual((staged/'sentinel').read_text(),'new')
            self.assertEqual(sorted(p.name for p in root.iterdir()),['stage','v3'])


if __name__ == '__main__':
    unittest.main()
