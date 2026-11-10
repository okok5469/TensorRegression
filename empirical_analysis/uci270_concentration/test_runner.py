import tempfile
import unittest
from pathlib import Path
import numpy as np
from empirical_analysis.uci270_concentration import runner


class RunnerContracts(unittest.TestCase):
    def test_seal_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.py'; source.write_text('frozen')
            contract={'source':{str(source):runner.sha(source)}}
            runner.verify_files(contract)
            source.write_text('changed')
            with self.assertRaises(ValueError):runner.verify_files(contract)

    def test_final_requires_matching_independent_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):runner.require_lock(root,'DEVELOPMENT_ACCEPTED.json','abc')
            runner.write_json(root/'DEVELOPMENT_ACCEPTED.json',{'status':'PASS','contract_sha256':'other'})
            with self.assertRaises(ValueError):runner.require_lock(root,'DEVELOPMENT_ACCEPTED.json','abc')
            runner.write_json(root/'DEVELOPMENT_ACCEPTED.json',{'status':'PASS','contract_sha256':'abc'})
            runner.require_lock(root,'DEVELOPMENT_ACCEPTED.json','abc')

    def test_replay_saved_model(self):
        from empirical_analysis.uci270_concentration.models import train_model,predict_model
        import pickle
        rng=np.random.default_rng(17); X=rng.normal(size=(32,16,3,2)); raw=rng.normal(size=(32,128));y=rng.normal(size=32)
        artifact=train_model('M4',X,y,{'alpha':1.})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pkl';path.write_bytes(pickle.dumps(artifact))
            expected=predict_model(artifact,X,raw)
            self.assertLessEqual(runner.replay(path,X,raw,expected),1e-10)
            with self.assertRaises(AssertionError):runner.replay(path,X,raw,expected+1.)

if __name__=='__main__':unittest.main()
