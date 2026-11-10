import importlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class MushroomRun(unittest.TestCase):
    def test_frozen_worker_resume_replay_and_corruption_rejection(self):
        api=importlib.import_module('empirical_analysis.spectrofood_mushroom.run')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.npz';out=root/'out';out.mkdir()
            rng=np.random.default_rng(2);X=rng.normal(size=(12,2,2,3))
            np.savez(data,ids=np.array(['M'+str(i) for i in range(12)]),X_tensor=X,
                     X_native=X.mean((1,2)),y_pct=np.arange(12.)+10,groups=np.arange(12).astype(str),blocks=np.ones(12))
            config=dict(data_path=str(data),output_dir=str(out),models=['M0'],source_files=[])
            split=dict(protocol='main',repeat=1,fold=1,inner_seed=1,train=list(range(8)),test=list(range(8,12)),
                       inner=[dict(train=list(range(4)),validation=list(range(4,8)))])
            body=dict(config=config,source=api.source_identity(config),data_sha256=api.file_hash(data),splits=[split])
            api.write_json(out/'contract.json',dict(**body,fingerprint=api.fingerprint(body)))
            api.run_one(config,0)
            task=out/'tasks/main_r1_f1_M0'
            marker=(task/'COMPLETE.json').read_bytes()
            api.run_one(config,0)
            self.assertEqual(marker,(task/'COMPLETE.json').read_bytes())
            self.assertEqual(api.verify_task(config,0)['prediction_count'],4)
            (task/'model.pkl').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash|Corrupt'):
                api.run_one(config,0)
