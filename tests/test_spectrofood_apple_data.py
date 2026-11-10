import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat


def module():
    name = 'empirical_analysis.spectrofood_apple.data'
    assert importlib.util.find_spec(name) is not None, 'Apple data module is missing'
    return __import__(name, fromlist=['data'])


class DataTests(unittest.TestCase):
    def test_area_average_preserves_mean_and_axis_order(self):
        d = module()
        cube = np.arange(5 * 7 * 3).reshape(5, 7, 3)
        actual = d.area_average(cube, (3, 4))
        np.testing.assert_allclose(actual.mean((0, 1)), cube.mean((0, 1)))
        np.testing.assert_allclose(d.area_average(cube, (5, 7)), cube)
        np.testing.assert_allclose(d.area_average(cube, (1, 1))[0, 0], cube.mean((0, 1)))

    def test_band_groups_sort_merge_duplicates_and_keep_adjacent(self):
        d = module()
        result, mapping = d.bin_wavelengths(np.array([450, 430, 430, 440]), 2)
        self.assertEqual([x['source_indices'] for x in mapping], [[1, 2, 3], [0]])
        np.testing.assert_allclose(result, [435, 450])
        np.testing.assert_allclose(mapping[0]['weights'], [.25, .25, .5])

    def test_segment_parser_handles_repeated_header_and_rejects_duplicate_ids(self):
        d = module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('\ufeffLeek,Dry matter,800,\nL1,.2,1,\n\nApple,Dry matter,430,434,\nA2,.15,2,3,\nApple,Dry matter,430,434,\nA1,.16,4,5,\n', encoding='utf-8')
            ids, y, wave, spectra = d.read_apple_csv(path)
            self.assertEqual(ids, ['A2', 'A1'])
            np.testing.assert_allclose(y, [.15, .16])
            np.testing.assert_array_equal(wave, [430, 434])
            with path.open('a') as stream:
                stream.write('A1,.16,4,5,\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                d.read_apple_csv(path)

    def test_prepare_joins_by_id_and_preserves_rectangular_roi(self):
        d = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube1 = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
            cube2 = cube1 + 50
            savemat(root / 'Apple.mat', {'A2': cube2, 'A1': cube1,
                    'Labels': np.array(['A2 ', 'A1 ']), 'Wavelengths': [430, 434, 438]})
            (root / 'data.csv').write_text('Apple,Dry matter,430,434,438\nA1,.16,1,2,3\nA2,.15,4,5,6\n')
            audit, paths = d.audit_and_prepare(root / 'Apple.mat', root / 'data.csv', root / 'out', expected_md5={})
            a = np.load(paths['data'])
            self.assertEqual(a['sample_ids'].tolist(), ['A1', 'A2'])
            np.testing.assert_allclose(a['y_pct'], [16, 15])
            np.testing.assert_allclose(a['X'][0], cube1)
            np.testing.assert_allclose(a['spectrum_native'][0], cube1.mean((0, 1)))
            self.assertEqual(audit['n_samples'], 2)
            a.close()

    def test_prepare_rejects_unmatched_ids(self):
        d = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            savemat(root / 'Apple.mat', {'A1': np.zeros((2, 2, 2)), 'Labels': ['A1'], 'Wavelengths': [430, 434]})
            (root / 'data.csv').write_text('Apple,Dry matter,430,434\nA2,.15,1,2\n')
            with self.assertRaisesRegex(ValueError, 'ID mismatch'):
                d.audit_and_prepare(root / 'Apple.mat', root / 'data.csv', root / 'out', expected_md5={})

    def test_identical_cubes_with_conflicting_response_block_preparation(self):
        d = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube = np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
            savemat(root / 'Apple.mat', {'A1': cube, 'A2': cube,
                    'Labels': ['A1', 'A2'], 'Wavelengths': [430, 434, 438]})
            (root / 'data.csv').write_text('Apple,Dry matter,430,434,438\nA1,.15,1,2,3\nA2,.16,1,2,3\n')
            with self.assertRaisesRegex(ValueError, 'duplicate cube.*conflicting response'):
                d.audit_and_prepare(root / 'Apple.mat', root / 'data.csv', root / 'out', expected_md5={})
            self.assertFalse((root / 'out' / 'data.npz').exists())


if __name__ == '__main__':
    unittest.main()
