"""Figure-data admission tests: complete four-method comparisons only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import figures


def fixture():
    configs, summary, unique = {}, [], []
    method_ids = {
        'adaptive': 'v52_search_final_original_msic',
        'oracle': 'oracle_ols_frozen',
        'lasso': 'vector_lasso_cv_isolated_v2',
        'glasso': 'sequential_tensor_glasso_certified_v2',
    }
    for index in range(10):
        cid = f'cfg_{index}'
        size = 150 if index == 0 else 50
        configs[cid] = {'n_train': 200, 'dims': [32, 20, 25], 'snr': 2, 'rho': [.3]*3}
        for method, method_id in method_ids.items():
            values = np.arange(size, dtype=float) / size + 1
            successes = size // 2
            lo, hi = figures.wilson(successes, size)
            stage = 'frozen_verified' if method in ('adaptive', 'oracle') else 'certified_new'
            summary.append({'config_id': cid, 'method': method, 'method_id': method_id,
                            'evidence_stage': stage, 'n': str(size),
                            'pmse_mean': str(values.mean()), 'pmse_sd': str(values.std(ddof=1)),
                            'pmse_valid_n': str(size), 'exact_recovery_valid_n': str(size),
                            'exact_successes': str(successes), 'exact_recovery_mean': str(successes/size),
                            'exact_wilson_low': str(lo), 'exact_wilson_high': str(hi),
                            'ree_mean': '0.5', 'element_f1_mean': '0.6', 'q_rect_mean': '60',
                            'element_nonzero_mean': '40', 'element_recall_mean': '0.7',
                            'element_fdr_mean': '0.2', 'mode_f1_macro_mean': '0.8'})
            unique.extend({'config_id': cid, 'method': method, 'method_id': method_id,
                           'evidence_stage': stage, 'unique_input_id': f'{cid}_{rep}',
                           'pmse': str(value), 'exact_recovery': str(int(rep < successes))}
                          for rep, value in enumerate(values))
    return summary, unique, configs


class CertifiedFigureDataTests(unittest.TestCase):
    def test_return_audit_binding_rejects_stale_csv(self):
        self.assertTrue(hasattr(figures, 'accepted_csv_hashes'), 'Accepted return-CSV binding is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root/'doc/taskcard18/cluster_review/RETURN_VALIDATION.json'
            audit_path.parent.mkdir(parents=True)
            name = 'doc/taskcard18/results/certified_config_summary.csv'
            source = root/name
            source.parent.mkdir(parents=True)
            source.write_text('source\n', encoding='utf8')
            audit = {'status': 'PASS', 'blocking_issues': [],
                     'certified_csv_sha256': {name: hashlib.sha256(source.read_bytes()).hexdigest()}}
            audit['record_sha256'] = hashlib.sha256(json.dumps(audit, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()
            audit_path.write_text(json.dumps(audit), encoding='utf8')
            hashes = figures.accepted_csv_hashes(root, [name])
            self.assertEqual(hashes[name], audit['certified_csv_sha256'][name])
            source.write_text('changed\n', encoding='utf8')
            with self.assertRaises(ValueError):
                figures.accepted_csv_hashes(root, [name])

    def test_all_four_methods_all_configurations_and_correct_intervals(self):
        summary, unique, configs = fixture()
        self.assertTrue(hasattr(figures, 'certified_table'), 'Certified four-method admission is missing')
        table, checks = figures.certified_table(summary, unique, configs)
        self.assertEqual(len(table), 40)
        self.assertEqual({r['method'] for r in table}, {'adaptive', 'oracle', 'lasso', 'glasso'})
        self.assertEqual(len(checks), 40)
        for method in ('adaptive', 'oracle', 'lasso', 'glasso'):
            self.assertEqual(sorted(r['n_unique_inputs'] for r in table if r['method'] == method), [50]*9+[150])
        row = next(r for r in table if r['config_id'] == 'cfg_1' and r['method'] == 'lasso')
        self.assertLess(row['pmse_ci95_low'], row['pmse_mean'])
        self.assertGreater(row['pmse_ci95_high'], row['pmse_mean'])

    def test_missing_replicate_and_duplicate_cannot_be_hidden_by_summary(self):
        summary, unique, configs = fixture()
        self.assertTrue(hasattr(figures, 'certified_table'), 'Certified four-method admission is missing')
        with self.assertRaises(ValueError):
            figures.certified_table(summary, unique[:-1], configs)
        with self.assertRaises(ValueError):
            figures.certified_table(summary, unique+[unique[0]], configs)

    def test_old_method_or_changed_denominator_rejected(self):
        summary, unique, configs = fixture()
        self.assertTrue(hasattr(figures, 'certified_table'), 'Certified four-method admission is missing')
        old = copy.deepcopy(summary)
        old[0]['method_id'] = 'legacy_lasso'
        with self.assertRaises(ValueError):
            figures.certified_table(old, unique, configs)
        wrong = copy.deepcopy(summary)
        wrong[0]['n'] = '50'
        with self.assertRaises(ValueError):
            figures.certified_table(wrong, unique, configs)


if __name__ == '__main__':
    unittest.main()
