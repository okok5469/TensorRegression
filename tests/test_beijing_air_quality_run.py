import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np


class BeijingRun(unittest.TestCase):
    def test_temporal_worker_resume_and_future_rejection(self):
        api=importlib.import_module('empirical_analysis.beijing_air_quality.run')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'out';out.mkdir();path=root/'data.npz'
            dates=np.arange('2015-01-01','2015-01-13',dtype='datetime64[D]').astype(str)
            np.savez(path,X=np.ones((12,2,2,3)),y=np.arange(12.)+10,persistence=np.arange(12.)+9,dates=dates,stations=['Aotizhongxin','B'])
            config=dict(data_path=str(path),output_dir=str(out),models=['M1'],source_files=[],synthetic_fixture=True)
            split=dict(protocol='development',fold=1,train=list(range(8)),test=list(range(8,12)),inner=[dict(train=[0,1,2],validation=[3,4,5])])
            def freeze():
                body=dict(config=config,source=api.source_identity(config),data_sha256=api.file_hash(path),splits=[split])
                api.write_json(out/'contract.json',dict(**body,fingerprint=api.fingerprint(body)))
            freeze();api.run_one(config,0)
            self.assertEqual(api.verify_task(config,0)['prediction_count'],4)
            api.run_one(config,0)
            split['train']=[10,11];split['test']=[8,9];freeze()
            with self.assertRaisesRegex(ValueError,'chronolog|future|temporal'):
                api.run_one(config,0)
