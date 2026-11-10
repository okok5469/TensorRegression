import unittest
import numpy as np
from empirical_analysis.uci270_concentration import protocol


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        batch=np.repeat(np.arange(1,11),15)
        return dict(batch=batch,gas=np.ones(150,int),y=np.tile(np.arange(15.),10),
                    group96=np.arange(150).astype(str),record_id=np.arange(150).astype(str))

    def test_latest_inner_and_final_labels_unused(self):
        d=self.fixture(); a=protocol.make_splits(d)
        self.assertEqual([p['boundary'] for p in a[0]['inner']],[3,4,5])
        d['y'][d['batch']==10]=np.nan
        self.assertEqual(a,protocol.make_splits(d))

    def test_duplicate_purges_training_only(self):
        d=self.fixture(); d['group96'][0]=d['group96'][90]
        s=protocol.make_splits(d)[0]
        self.assertNotIn(0,s['train']);self.assertIn(90,s['test'])
        self.assertEqual(s['purged_train'],[0])

    def test_insufficient_boundaries_retained(self):
        d=self.fixture();d['gas'][d['batch']<6]=2
        s=protocol.make_splits(d)[0]
        self.assertEqual(s['status'],'not_evaluable')

if __name__=='__main__':unittest.main()
