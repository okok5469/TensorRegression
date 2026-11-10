"""A returned-run status must stay bound to its archive and rebuilt evidence."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


class ReturnReportTests(unittest.TestCase):
    def implementation(self):
        name = 'research.taskcard18.return_report'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Missing sealed return-report reader')
        return importlib.import_module(name)

    def test_return_evidence_rejects_archive_and_derived_table_mutation(self):
        implementation = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'returned.zip'
            table = root / 'table.csv'
            archive.write_bytes(b'original archive')
            table.write_bytes(b'original table')
            evidence = {'status': 'PASS', 'archive_repo_relative_path': 'returned.zip',
                        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                        'certified_csv_sha256': {'table.csv': hashlib.sha256(table.read_bytes()).hexdigest()}}
            evidence['record_sha256'] = digest(evidence)
            path = root / 'validation.json'
            path.write_text(json.dumps(evidence), encoding='utf-8')
            self.assertEqual(implementation.verified_return_evidence(root, path), evidence)
            table.write_bytes(b'changed table')
            with self.assertRaisesRegex(ValueError, 'table|Table|hash|Hash'):
                implementation.verified_return_evidence(root, path)
            table.write_bytes(b'original table')
            archive.write_bytes(b'changed archive')
            with self.assertRaisesRegex(ValueError, 'archive|Archive|hash|Hash'):
                implementation.verified_return_evidence(root, path)

    def test_unsealed_or_failed_return_cannot_grant_any_gate(self):
        implementation = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'validation.json'
            for value in ({'status': 'PASS'}, {'status': 'BLOCKED'}):
                value['record_sha256'] = digest(value)
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises((ValueError, KeyError)):
                    implementation.verified_return_evidence(root, path)

    def test_returned_counts_require_every_fit_evaluation_and_statistical_unit(self):
        implementation = self.implementation()
        self.assertTrue(hasattr(implementation, 'require_complete_counts'), 'Missing complete-return count gate')
        counts = {'historical_records': 700, 'historical_npz_verified': 700,
                  'label_method_rows': 2800, 'unique_inputs': 600, 'configurations': 10,
                  'valid_fits': 1200, 'evaluations': 1200, 'failures': 0,
                  'unique_method_rows': 2400, 'config_method_cells': 40,
                  'label_method_cells': 56, 'paired_config_comparisons': 30}
        self.assertTrue(implementation.require_complete_counts(counts))
        for key in counts:
            wrong = dict(counts)
            wrong[key] += 1
            with self.assertRaises(ValueError, msg=key):
                implementation.require_complete_counts(wrong)


if __name__ == '__main__':
    unittest.main(verbosity=2)
