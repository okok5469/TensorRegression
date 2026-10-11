"""Independent source alignment and corrupted-input rejection tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        (self.source / 'qc').mkdir(parents=True)
        (self.source / 'tensors').mkdir()
        self.ids = np.array(['a', 'b', 'c', 'd'])
        self.X = np.arange(16.).reshape(4, 2, 2, 1)
        self.raw_y = np.array([10., 20., 30., 40.])
        self.C = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=float)
        enc = {'id_column':'participant_id','age_column':'age_band','sex_column':'sex',
               'age_encoding':'categorical','age_categories':['20-25','25-30'],
               'age_reference':'20-25','age_mapping':{'20-25':'20-25','25-30':'25-30'},
               'sex_mapping':{'1':0,'2':1}}
        self.meta = {'shape':list(self.X.shape),'channels':['C1','C2'],'bands':['alpha'],
                     'band_edges_hz':{'alpha':[8.,13.]},'time_windows':2,'window_sec':25.,
                     'feature':'log_absolute_band_power','sfreq':250.}
        self.json('tensors/tensor_meta.json',self.meta)
        self.json('config_used.json',{'config':{'response':{'column':'TMT_5','primary_transform':'log'},
                  'confounds':enc,'events':{'n_windows':2,'window_sec':25.},
                  'bands':{'alpha':[8.,13.]},'preprocess':{'target_sfreq':250.}}})
        self.json('qc/covariate_design.json',{'columns':['intercept','age[25-30]','sex'],'encoding':enc})
        np.save(self.source / 'tensors/X_unstandardized.npy',self.X)
        self.write_npz()
        rows=[]
        for i,s in enumerate(self.ids):
            age=enc['age_categories'][int(self.C[i,1])]
            rows.append({'subject_id':s,'tensor_index':i,'tmt5_raw':self.raw_y[i],
                         'tmt5_log':np.log(self.raw_y[i]),'age_source':age,
                         'sex_source':str(int(self.C[i,2])+1),'age_category':age,
                         'sex':self.C[i,2],'include_primary':'True'})
        self.table(self.source/'qc/final_subject_manifest.csv',rows)
        self.tmt=self.root/'TMT.csv'; self.parts=self.root/'participants.tsv'
        self.table(self.tmt,[{'ID':s,'TMT_5':v} for s,v in zip(self.ids,self.raw_y)])
        self.table(self.parts,[{'participant_id':r['subject_id'],'age_band':r['age_source'],
                                'sex':r['sex_source']} for r in rows], '\t')

    def json(self,relative,value):
        (self.source/relative).write_text(json.dumps(value),encoding='utf-8')

    @staticmethod
    def table(path,rows,delimiter=','):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=delimiter)
            w.writeheader(); w.writerows(rows)

    def write_npz(self):
        np.savez(self.source/'qc/analysis_dataset.npz',X=self.X,y=np.log(self.raw_y),C=self.C,subject_ids=self.ids)

    def load(self):
        from .data_loader import load_audited_dataset
        return load_audited_dataset(self.source,self.tmt,self.parts)

    def test_full_provenance(self):
        d=self.load(); self.assertTrue(d.audit['passed']); self.assertEqual(d.audit['n_subjects'],4)

    def test_altered_tensor_blocked(self):
        self.X+=1; self.write_npz()
        with self.assertRaisesRegex(ValueError,'unstandardized'): self.load()

    def test_wrong_log_response_blocked(self):
        self.raw_y+=1; self.write_npz()
        with self.assertRaisesRegex(ValueError,'response'): self.load()

    def test_covariate_mismatch_blocked(self):
        self.C[0,-1]=1; self.write_npz()
        with self.assertRaisesRegex(ValueError,'(?i)covariate'): self.load()

    def test_metadata_corruption_blocked(self):
        self.meta['channels']=['C1','C1']; self.json('tensors/tensor_meta.json',self.meta)
        with self.assertRaisesRegex(ValueError,'metadata'): self.load()


if __name__=='__main__': unittest.main()
