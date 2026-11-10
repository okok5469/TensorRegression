import unittest
import importlib
import tempfile
from pathlib import Path
import numpy as np


class RunContractTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('empirical_analysis.spectrofood_apple.run')
        except ModuleNotFoundError:
            self.fail('Apple runner is not implemented')

    def test_group_splits_complete_and_disjoint(self):
        api = self.api()
        ids = np.array(['s%d' % i for i in range(30)])
        groups = np.repeat(np.arange(15).astype(str), 2)
        splits = api.make_splits(ids, groups)
        self.assertEqual(len(splits), 15)
        for repeat in range(1, 4):
            test_ids = [i for s in splits if s['repeat'] == repeat for i in s['test']]
            self.assertEqual(sorted(test_ids), list(range(30)))
        for split in splits:
            self.assertFalse(set(groups[split['train']]) & set(groups[split['test']]))
            for inner in split['inner']:
                self.assertFalse(set(groups[np.array(split['train'])[inner['train']]]) &
                                 set(groups[np.array(split['train'])[inner['validation']]]))
        self.assertEqual(splits, api.make_splits(ids, groups))

    def test_complete_oof_metrics_not_average_fold_r2(self):
        api = self.api()
        y = np.array([0., 1., 10., 12.])
        p = np.array([1., 2., 8., 10.])
        m = api.metrics(y, p)
        self.assertAlmostEqual(m['rmse'], np.sqrt(2.5))
        self.assertAlmostEqual(m['mae'], 1.5)
        self.assertAlmostEqual(m['r2'], 1 - 10 / np.sum((y-y.mean())**2))

    def test_fingerprint_changes_for_every_contract_component(self):
        api = self.api()
        base = dict(data='a', representation=[8,8,24], split={'train':['A1']},
                    preprocessing='train_fold_xyz', model={'id':'M1'}, source='b')
        key = api.fingerprint(base)
        for field in base:
            changed = dict(base, **{field: 'changed'})
            self.assertNotEqual(key, api.fingerprint(changed))

    def test_oof_rejects_missing_or_duplicate_sample(self):
        api = self.api()
        rows = [dict(sample_id='A1', entity_id='A1', repeat=1, fold=1, model='M0',
                     y_true_pct=10., y_pred_pct=11.)]
        with self.assertRaises(ValueError):
            api.validate_oof(rows, ['A1','A2'], ['M0'], repeats=1)
        with self.assertRaises(ValueError):
            api.validate_oof(rows*2, ['A1'], ['M0'], repeats=1)

    def test_saved_models_replay_and_corrupt_cache_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            data_dir = directory/'data'
            data_dir.mkdir()
            rng=np.random.default_rng(51)
            X=rng.normal(size=(30,2,2,2))
            y=16+X[:,1,1,1]+rng.normal(size=30)*.1
            ids=np.array(['s%d'%i for i in range(30)])
            np.savez_compressed(data_dir/'data.npz',X=X,y_pct=y,sample_ids=ids,entity_ids=ids)
            config=dict(data_dir=str(data_dir),output_dir=str(directory/'out'),run_id='test',
                release_manifest='doc/releases/v5.2-search-final/RELEASE_MANIFEST.json',
                models=['M1'],model_config={})
            api.freeze(config)
            for task in range(15):
                api.run_one(config,task)
            rows,_,_=api.collect(config)
            self.assertEqual(len(rows),90)
            replay=api.read_json(directory/'out'/'OOF_REPLAY_VALIDATION.json')
            self.assertEqual(replay['status'],'PASS')
            marker=directory/'out'/'tasks'/'r1_f1_M1'/'COMPLETE.json'
            before=marker.read_bytes()
            api.run_one(config,0)
            self.assertEqual(before,marker.read_bytes())
            model=marker.parent/'model.pkl'
            model.write_bytes(model.read_bytes()+b'corruption')
            with self.assertRaisesRegex(ValueError,'Corrupt completed artifact'):
                api.run_one(config,0)


if __name__ == '__main__':
    unittest.main()
