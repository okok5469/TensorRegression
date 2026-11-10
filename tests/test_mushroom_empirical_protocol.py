import importlib
import unittest
import numpy as np


class ProtocolTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('empirical_analysis.spectrofood_mushroom.protocol')
        except ModuleNotFoundError:
            self.fail('Task15 ID-based protocol is missing')

    def test_screen_ids_remap_after_cache_reordering(self):
        api = self.api()
        ids = np.array(['M4','M2','M1','M3'])
        data = dict(ids=ids, groups=ids, blocks=np.arange(1,5))
        old = [dict(repeat=1,fold=1,seed=23001,train=[0,1,2],test=[3],
                    train_ids=['M1','M2','M3'],test_ids=['M4'],
                    inner=[dict(train=[0,1],validation=[2],train_ids=['M1','M2'],validation_ids=['M3'])])]
        result = api.remap_main(data,old)[0]
        self.assertEqual(result['train'],[2,1,3])
        self.assertEqual(result['test'],[0])
        self.assertEqual(result['inner'][0]['train'],[0,1])
        self.assertEqual(result['inner'][0]['validation'],[2])

    def test_lobo_inner_three_folds_never_split_files(self):
        api = self.api()
        ids=np.array([f'M{i}' for i in range(1,21)])
        data=dict(ids=ids,groups=ids,blocks=np.repeat(np.arange(1,6),4))
        folds=api.make_blocks(data)
        self.assertEqual(len(folds),5)
        seen=[]
        for split in folds:
            tr=np.array(split['train']);te=np.array(split['test']);seen.extend(te)
            self.assertEqual(len(split['inner']),3)
            self.assertFalse(set(data['blocks'][tr])&set(data['blocks'][te]))
            validation=[]
            for inside in split['inner']:
                a=np.array(inside['train']);b=np.array(inside['validation']);validation.extend(b)
                self.assertFalse(set(data['blocks'][tr[a]])&set(data['blocks'][tr[b]]))
            self.assertEqual(sorted(validation),list(range(len(tr))))
        self.assertEqual(sorted(seen),list(range(20)))

    def test_crossfile_duplicate_refuses_incompatible_lobo(self):
        api=self.api()
        data=dict(ids=np.array(['M1','M2','M3','M4','M5']),
                  groups=np.array(['same','same','c','d','e']),blocks=np.arange(1,6))
        with self.assertRaisesRegex(ValueError,'group'):
            api.make_blocks(data)


if __name__=='__main__': unittest.main()
