"""Returned evidence must remain bound to its original Linux execution."""
import importlib
import importlib.util
import io
import json
import unittest
import warnings
import zipfile


def api():
    name = 'research.taskcard18.return_audit'
    assert importlib.util.find_spec(name) is not None, 'Missing read-only returned-archive acceptance'
    return importlib.import_module(name)


class ReturnAuditTests(unittest.TestCase):
    def test_archive_rejects_traversal_and_duplicate_members(self):
        for names in (['../outside'], ['C:/outside'], ['a/b'], ['a', 'a']):
            with self.subTest(names=names), warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, 'w') as archive:
                    for name in names:
                        archive.writestr(name, b'content')
                raw = stream.getvalue().replace(b'a/b', b'a\\b') if names == ['a/b'] else stream.getvalue()
                with zipfile.ZipFile(io.BytesIO(raw)) as archive, self.assertRaises(ValueError):
                    api().archive_members(archive)

    def test_archive_does_not_extract_files(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('valid/file.json', '{}')
        with zipfile.ZipFile(stream) as archive:
            self.assertEqual(api().archive_members(archive), {'valid/file.json'})

    def test_seal_rejects_altered_record(self):
        value = {'identity': {'platform': 'Linux'}, 'tasks': []}
        value['record_sha256'] = api().digest(value)
        api().verify_seal(value)
        value['identity']['platform'] = 'Windows'
        with self.assertRaises(ValueError):
            api().verify_seal(value)

    def test_evaluation_uses_recorded_linux_identity_without_local_substitution(self):
        method = {'identity': {'platform': 'Linux'}, 'unique_input_id': 'input_1',
                  'method_id': 'lasso_v2', 'npz_sha256': 'a' * 64}
        evaluation = {'identity': method['identity'].copy(), 'unique_input_id': 'input_1',
                      'method_id': 'lasso_v2', 'fit_npz_sha256': 'a' * 64,
                      'metrics': {'pmse': 1., 'signal_mse': 0., 'r2_test': -1.},
                      'test_stream_hashes': [{'X': 'a' * 64, 'y': 'b' * 64, 'signal': 'c' * 64}] * 5}
        api().verify_evaluation(evaluation, method, 500)
        evaluation['identity'] = {'platform': 'Windows'}
        with self.assertRaises(ValueError):
            api().verify_evaluation(evaluation, method, 500)

    def test_evaluation_rejects_nonfinite_scores_and_missing_test_batch(self):
        method = {'identity': {'platform': 'Linux'}, 'unique_input_id': 'input_1',
                  'method_id': 'lasso_v2', 'npz_sha256': 'a' * 64}
        base = {'identity': method['identity'], 'unique_input_id': 'input_1',
                'method_id': 'lasso_v2', 'fit_npz_sha256': 'a' * 64,
                'metrics': {'pmse': 1., 'signal_mse': 0., 'r2_test': 0.},
                'test_stream_hashes': [{'X': 'a' * 64, 'y': 'b' * 64, 'signal': 'c' * 64}] * 5}
        for change in ('nan', 'batch', 'hash'):
            value = json.loads(json.dumps(base))
            if change == 'nan': value['metrics']['pmse'] = float('nan')
            if change == 'batch': value['test_stream_hashes'].pop()
            if change == 'hash': value['test_stream_hashes'][0]['X'] = 'unsealed'
            with self.subTest(change=change), self.assertRaises(ValueError):
                api().verify_evaluation(value, method, 500)

    def test_csv_numeric_rounding_passes_but_missing_or_changed_pair_fails(self):
        expected = [{'uid': 'a', 'value': 1.23456789}, {'uid': 'b', 'value': float('nan')}]
        actual = [{'uid': 'a', 'value': '1.23456789000001'}, {'uid': 'b', 'value': 'nan'}]
        result = api().compare_rows(actual, expected, ('uid',))
        self.assertEqual(result['rows'], 2)
        for corrupt in ([actual[0]], [actual[0], actual[0]],
                        [{'uid': 'a', 'value': '1.9'}, actual[1]]):
            with self.assertRaises(ValueError):
                api().compare_rows(corrupt, expected, ('uid',))

    def test_fold_indices_and_final_response_are_checked_against_recorded_input(self):
        import numpy as np
        from research.taskcard18.baselines import _array_hash
        n = 10
        diagnostics = [{'fit_kind': 'cv', 'fold_index': 0, 'n_fit': 8,
                        'train_indices_hash': _array_hash(np.arange(2, 10, dtype=np.int64))},
                       {'fit_kind': 'final', 'fold_index': None, 'n_fit': 10,
                        'train_indices_hash': _array_hash(np.arange(10, dtype=np.int64)),
                        'train_response_hash': 'response', 'warm_start_from': None}]
        self.assertTrue(hasattr(api(), 'verify_fit_data'), 'Missing fitting-subset identity audit')
        api().verify_fit_data(diagnostics, n, 'response')
        for name, value in (('train_indices_hash', 'wrong-fold'), ('train_response_hash', 'wrong-response'), ('warm_start_from', 1)):
            changed = json.loads(json.dumps(diagnostics))
            changed[1][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                api().verify_fit_data(changed, n, 'response')


if __name__ == '__main__':
    unittest.main(verbosity=2)
