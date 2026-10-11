"""Acceptance tests for subject isolation, training-only selection and recovery."""
import unittest
import numpy as np


class NestedCvTests(unittest.TestCase):
    def test_repeated_splits_cover_each_subject_and_share_models(self):
        from .nested_cv import make_splits
        C = np.column_stack((np.ones(30), np.repeat([0, 1, 2], 10)))
        splits = make_splits(C, 5, [101, 102, 103])
        for repeat in range(3):
            folds = [s for s in splits if s['repeat'] == repeat]
            self.assertEqual(len(folds), 5)
            tests = np.concatenate([s['test'] for s in folds])
            np.testing.assert_array_equal(np.sort(tests), np.arange(30))
            for s in folds:
                self.assertEqual(len(set(s['train']) & set(s['test'])), 0)

    def test_preprocessing_restores_covariate_only_ols(self):
        from .preprocessing import fit_preprocessor
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 2, 2, 2))
        C = np.column_stack((np.ones(20), rng.normal(size=20)))
        y = C @ np.array([4., 2.]) + rng.normal(size=20)
        p = fit_preprocessor(X[:15], y[:15], C[:15])
        got = p.predict(np.zeros((2, 2, 2)), X[15:], C[15:])
        expected = C[15:] @ np.linalg.lstsq(C[:15], y[:15], rcond=None)[0]
        np.testing.assert_allclose(got, expected, atol=1e-12)
        np.testing.assert_allclose(p.apply(X[:15], C[:15]).mean(axis=0), 0, atol=1e-12)

    def test_select_parameters_fits_each_inner_training_only(self):
        from .nested_cv import select_parameters
        from .preprocessing import fit_preprocessor
        rng = np.random.default_rng(8)
        X = rng.normal(size=(24, 2, 2, 1))
        C = np.column_stack((np.ones(24), rng.normal(size=24)))
        y = C @ np.array([4., 1.]) + X[:, 0, 0, 0] * .7
        seen = []
        class Fit:
            B = np.zeros((2, 2, 1))
        def solver(name, x, response, parameter, options):
            seen.append((len(x), x.mean(axis=0), response.mean()))
            return Fit()
        candidates = [{'lambda': .1}, {'lambda': 1.}]
        parameter, rows, traces = select_parameters('Ridge', X, y, C,
                                                   candidates, 3, 22, {}, solver)
        self.assertEqual(parameter, candidates[-1])  # stronger penalty breaks ties
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(traces), 3)
        for n, means, ymean in seen:
            self.assertEqual(n, 16)
            np.testing.assert_allclose(means, 0, atol=1e-12)
            self.assertLess(abs(ymean), 1e-12)
        for trace in traces:
            self.assertFalse(set(trace['train']) & set(trace['validation']))

    def test_response_and_covariate_prediction_recovery(self):
        from .preprocessing import fit_preprocessor
        rng = np.random.default_rng(21)
        C = np.column_stack((np.ones(40), rng.normal(size=40)))
        X = rng.normal(size=(40, 2, 2, 1)) + C[:, 1, None, None, None]
        y = C @ np.array([3., .2]) + 2 * X[:, 0, 0, 0]
        p = fit_preprocessor(X[:30], y[:30], C[:30])
        coef = np.linalg.lstsq(p.apply(X[:30], C[:30]).reshape(30, -1),
                              p.transform_response(y[:30], C[:30]), rcond=None)[0]
        np.testing.assert_allclose(p.predict(coef.reshape(2, 2, 1), X[30:], C[30:]),
                                   y[30:], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
