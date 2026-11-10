import unittest
import numpy as np
from empirical_analysis.uci270_concentration import data


class DataTests(unittest.TestCase):
    def test_index_order_and_label(self):
        row = '2;25.5 ' + ' '.join(f'{i}:{-i}' for i in range(128, 0, -1))
        gas, y, label, values = data.parse_line(row)
        self.assertEqual((gas, y, label), (2, 25.5, '2;25.5'))
        np.testing.assert_array_equal(values, -np.arange(1, 129))

    def test_reject_invalid_records(self):
        good = [f'{i}:{i}' for i in range(1, 129)]
        cases = ['1 ' + ' '.join(good), '1;nan ' + ' '.join(good),
                 '1;1 ' + ' '.join(good[:-1]),
                 '1;1 ' + ' '.join(good[:-1] + ['1:2']),
                 '1;1 ' + ' '.join(good[:-1] + ['128:inf'])]
        for row in cases:
            with self.subTest(row=row[:25]), self.assertRaises(ValueError):
                data.parse_line(row)

    def test_mapping_all_coordinates(self):
        raw = np.arange(256.).reshape(2, 128)
        tensor = data.make_tensor(raw)
        for j in range(16):
            for k in range(3):
                np.testing.assert_array_equal(tensor[:, j, k, 0], raw[:, j*8+2+k])
                np.testing.assert_array_equal(tensor[:, j, k, 1], raw[:, j*8+5+k])
        beta = np.arange(96.).reshape(16, 3, 2)
        np.testing.assert_allclose(np.einsum('nijk,ijk->n', tensor, beta),
                                   tensor.reshape(2, 96) @ beta.ravel(order='C'))

    def test_duplicate_groups_use_x_only_and_normalize_signed_zero(self):
        x = np.array([[0., 1.], [-0., 1.], [0., 2.]])
        groups = data.exact_groups(x)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])


if __name__ == '__main__':
    unittest.main()
