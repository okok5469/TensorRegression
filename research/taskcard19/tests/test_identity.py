"""Release runtime evidence is measured and checked against the application lock."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class IdentityChecks(unittest.TestCase):
    def runtime(self):
        module = importlib.import_module("tensor_abess_ovr.identity")
        self.assertTrue(hasattr(module, "runtime_identity"), "Missing measured dependency/thread/import identity")
        return module.runtime_identity

    def test_measured_dependency_import_and_thread_evidence(self):
        evidence = self.runtime()()
        self.assertEqual(evidence["dependencies"]["numpy"], "1.26.4")
        self.assertEqual(evidence["dependencies"]["scikit-learn"], "1.5.2")
        self.assertTrue(evidence["threadpools"])
        self.assertTrue(all(pool["num_threads"] == 1 for pool in evidence["threadpools"]))
        self.assertTrue(evidence["actual_import_files"]["tensor_abess"].endswith("__init__.py"))
        self.assertTrue(evidence["actual_import_files"]["tensor_abess_ovr"].endswith("__init__.py"))

    def test_wrong_python_dependency_or_active_threads_are_rejected(self):
        runtime = self.runtime()
        with patch("tensor_abess_ovr.identity.platform.python_version_tuple", return_value=("3", "11", "9")):
            with self.assertRaisesRegex(RuntimeError, "Python"):
                runtime()
        with patch("importlib.metadata.version", return_value="999.0"):
            with self.assertRaisesRegex(RuntimeError, "dependency"):
                runtime()
        with patch("threadpoolctl.threadpool_info", return_value=[{"num_threads": 2, "internal_api": "openblas"}]):
            with self.assertRaisesRegex(RuntimeError, "thread"):
                runtime()

    def test_ordinary_api_import_sets_single_threads_before_numerical_imports(self):
        names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
        environment = os.environ.copy()
        environment.update({name: "8" for name in names})
        script = "import os,sys,json; import tensor_abess_ovr; print(json.dumps({'numpy_loaded':'numpy' in sys.modules,'threads':{k:os.environ[k] for k in " + repr(names) + "}}))"
        result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True, check=True)
        measured = json.loads(result.stdout)
        self.assertFalse(measured["numpy_loaded"])
        self.assertEqual(set(measured["threads"].values()), {"1"})

    def test_public_fit_and_load_reject_already_active_multithreading(self):
        import numpy as np
        from threadpoolctl import threadpool_limits
        from tensor_abess_ovr.model import TensorABESSOvR
        X = np.sin(np.arange(24 * 8).reshape(24, 2, 2, 2) * .137)
        y = np.asarray(["a", "b", "c"] * 8)
        saved = TensorABESSOvR(core_timeout=None).fit(X, y)
        with tempfile.TemporaryDirectory() as temporary:
            saved.save(temporary)
            with threadpool_limits(limits=2), patch("tensor_abess_ovr.model._fit_one", side_effect=AssertionError("must reject before core fitting")):
                with self.assertRaisesRegex(RuntimeError, "CLI"):
                    TensorABESSOvR(core_timeout=None).fit(X, y)
                with self.assertRaisesRegex(RuntimeError, "CLI"):
                    TensorABESSOvR.load(temporary)

    def test_public_fit_rejects_profile_fingerprint_drift(self):
        import numpy as np
        import tensor_abess_ovr.identity as identity
        from tensor_abess_ovr.model import TensorABESSOvR
        original = identity.sha256

        def changed_profile(path):
            return "0" * 64 if Path(path).name == "core_profile.json" else original(path)

        with patch("tensor_abess_ovr.identity.sha256", side_effect=changed_profile), patch("tensor_abess_ovr.model._fit_one", side_effect=AssertionError("must reject before core fitting")):
            with self.assertRaisesRegex(RuntimeError, "core_profile.json"):
                TensorABESSOvR(core_timeout=None).fit(np.arange(48.).reshape(6, 2, 2, 2), ["a", "b", "c"] * 2)


if __name__ == "__main__":
    unittest.main()
