"""Small deterministic fixed/adaptive import and fit check, not scientific validation."""
from pathlib import Path
import sys
import json
import numpy as np


def smoke():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from tensor_abess.tensor.abess import tensor_abess
    import importlib
    module = importlib.import_module('tensor_abess.tensor.abess')
    assert Path(module.__file__).resolve() == root / 'src/tensor_abess/tensor/abess.py'
    rng = np.random.default_rng(90210)
    X = rng.normal(size=(48, 6, 5, 4))
    truth = np.zeros((6, 5, 4))
    truth[:2, :2, :2] = 1
    y = np.einsum('nijk,ijk->n', X, truth) + rng.normal(0, .1, 48)
    results = {}
    for mode, sizes in [('fixed', (2, 2, 2)), ('adaptive', None)]:
        result = tensor_abess(X, y, target_sparsity=sizes, k_max=1, c=3, criterion='bic',
                              n_lambdas=6, cv_folds=3, max_iter=80, max_cycles=3)
        supports = [result.S1, result.S2, result.S3]
        q = int(np.prod([len(s) for s in supports]))
        assert result.B.shape == (6, 5, 4) and np.isfinite(result.B).all()
        assert np.isfinite(result.loss) and result.loss >= 0 and np.isfinite(result.rss)
        assert abs(result.rss - 2 * len(y) * result.loss) < 1e-8
        assert q <= int(np.floor(.7 * len(y)))
        assert sizes is None or tuple(len(s) for s in supports) == sizes
        mask = np.zeros_like(truth, dtype=bool)
        mask[np.ix_(*supports)] = True
        assert np.count_nonzero(result.B[~mask]) == 0
        results[mode] = {'B_shape': list(result.B.shape), 'supports': [s.tolist() for s in supports],
                         'q': q, 'loss': float(result.loss), 'rss': float(result.rss)}
    return results


if __name__ == '__main__':
    print(json.dumps({'smoke': 'PASSED', 'results': smoke()}))
