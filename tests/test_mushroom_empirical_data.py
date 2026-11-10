import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat


def data_module():
    name = 'empirical_analysis.spectrofood_mushroom.data'
    try:
        spec=importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec=None
    assert spec is not None, 'Mushroom data module missing'
    return __import__(name, fromlist=['data'])


class MushroomDataTests(unittest.TestCase):
    def fixture(self, root, wavelength_mismatch=False):
        c = np.arange(3*4*4, dtype=float).reshape(3,4,4)
        wave = np.array([400.,403.,406.,409.])
        savemat(root/'Mushroom_1.mat', dict(M002=c, M001=c, Labels=['M002','M001'],
                 Wavelengths=wave + int(wavelength_mismatch)))
        (root/'SpectroFood_dataset.csv').write_text('Mush No,Dry matter,400,403,406,409\nM1,.1,1,2,3,4\nM2,.2,1,2,3,4\n')
        return c

    def test_id_join_keeps_duplicate_cube_and_conflicting_responses(self):
        d = data_module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=self.fixture(root)
            metadata=d.prepare(root,root/'out',expected_md5={},expected_count=2)
            with np.load(root/'out/data.npz') as a:
                self.assertEqual(a['ids'].tolist(),['M1','M2'])
                np.testing.assert_array_equal(a['y_pct'],[10.,20.])
                np.testing.assert_allclose(a['X_tensor'][0],c)
                np.testing.assert_allclose(a['X_mat_mean'][0],c.mean((0,1)))
                np.testing.assert_array_equal(a['X_native'][0],[1,2,3,4])
                self.assertEqual(a['groups'][0],a['groups'][1])
            self.assertEqual(len(metadata['duplicate_cubes']),1)
            self.assertTrue(metadata['duplicate_cubes'][0]['conflicting_response'])

    def test_wavelength_mismatch_stops_before_cache(self):
        d=data_module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,True)
            with self.assertRaisesRegex(ValueError,'wavelength'):
                d.prepare(root,root/'out',expected_md5={},expected_count=2)
            self.assertFalse((root/'out/data.npz').exists())

    def test_canonical_id_mapping_requires_decimal_identity(self):
        d=data_module()
        self.assertEqual(d.canonical_id('M001'),'M1')
        with self.assertRaises(ValueError):
            d.canonical_id('M_001')


if __name__=='__main__': unittest.main()
