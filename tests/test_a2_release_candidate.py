"""Review-only A2 snapshot invariants; opt-in fit bridge runs after cold timing."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "doc/algorithm_versions/v5.1-bcd-fix/src"
FINAL = ROOT / "doc/algorithm_versions/v5.2-search-final/src"
BCD = Path("tensor_abess/tensor/bcd.py")


def text(path):
    return path.read_text(encoding="utf-8")


def grid_function(source):
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_candidate_size_triples")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<grid>", "exec"), namespace)
    return namespace["_candidate_size_triples"]


def outside_grid(raw):
    node = next(n for n in ast.parse(raw.decode("utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "_candidate_size_triples")
    lines = raw.splitlines(keepends=True)
    return b"".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


class SnapshotTests(unittest.TestCase):
    def candidate(self):
        self.assertTrue((FINAL / BCD).is_file(), "Review-only A2 snapshot missing")
        return text(FINAL / BCD)

    def test_final_full_grid_contains_legacy_omitted_size_three(self):
        legacy = grid_function(text(BASE / BCD))
        self.assertNotIn((3,3,3), legacy(6,6,7,280))
        candidate = grid_function(self.candidate())
        self.assertIn((3,3,3), candidate(6,6,7,280))
        self.assertEqual(len(candidate(6,6,7,280)),252)

    def test_full_grid_is_ordered_unique_and_capacity_filtered(self):
        candidate = grid_function(self.candidate())
        expected = [(a,b,c) for a in range(1,5) for b in range(1,4) for c in range(1,6) if a*b*c <= 30]
        self.assertEqual(candidate(4,3,5,30), expected)
        self.assertEqual(expected, sorted(set(expected)))
        self.assertEqual(candidate(1,1,1,0), [])
        self.assertEqual(candidate(0,3,4,10), [])

    def test_only_grid_function_differs_from_v51_raw_sources(self):
        self.candidate()
        base = {p.relative_to(BASE):p for p in BASE.rglob("*.py")}
        final = {p.relative_to(FINAL):p for p in FINAL.rglob("*.py")}
        self.assertEqual(set(base),set(final))
        self.assertEqual(len(final),32)
        changed = [name for name in base if base[name].read_bytes()!=final[name].read_bytes()]
        self.assertEqual(changed,[BCD])
        self.assertEqual(outside_grid(base[BCD].read_bytes()),outside_grid(final[BCD].read_bytes()))

    def test_fixed_adaptive_refine_and_public_signatures_are_unchanged(self):
        candidate = ast.parse(self.candidate())
        baseline = ast.parse(text(BASE / BCD))
        names = ("tensor_bcd","tensor_bcd_adaptive","_refine_msic_global_candidates")
        for name in names:
            base = next(node for node in baseline.body if isinstance(node,ast.FunctionDef) and node.name==name)
            final = next(node for node in candidate.body if isinstance(node,ast.FunctionDef) and node.name==name)
            self.assertEqual(ast.dump(base,include_attributes=False),ast.dump(final,include_attributes=False))
        self.assertEqual((BASE/"tensor_abess/tensor/abess.py").read_bytes(),(FINAL/"tensor_abess/tensor/abess.py").read_bytes())

    def test_sources_parse_with_python39_and_manifest_matches_all_files(self):
        self.candidate()
        version = json.loads((FINAL.parent/"VERSION.json").read_text(encoding="utf-8"))
        self.assertTrue(version["research_only"])
        self.assertFalse(version["release_approved"])
        self.assertEqual(version["changed_core_files"],[BCD.as_posix()])
        paths = sorted(FINAL.rglob("*.py"))
        self.assertEqual(len(version["source_sha256"]),32)
        for path in paths:
            ast.parse(text(path),feature_version=(3,9))
            name = path.relative_to(FINAL).as_posix()
            self.assertEqual(version["source_sha256"][name],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(version["source_canonical_lf_sha256"][name],hashlib.sha256(text(path).encode("utf-8")).hexdigest())


@unittest.skipUnless(os.environ.get("A2_RUN_BRIDGE") == "1", "Fit bridge waits until independent cold timing finishes")
class FitBridgeTests(unittest.TestCase):
    def test_default_final_matches_research_explicit_A2(self):
        import numpy as np
        rng = np.random.default_rng(2052011)
        X = rng.normal(size=(400,6,6,7))
        y = 2*X[:,0,0,0] + X[:,1,2,3] + rng.normal(size=400)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            np.savez(temp/"data.npz",X=X,y=y)
            script = '''import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
from tensor_abess.tensor.abess import tensor_abess
with np.load(sys.argv[2],allow_pickle=False) as data:
    kw=dict(c=3.,criterion="bic",k_max=1,max_cycles=2,n_lambdas=3,cv_folds=2,max_iter=80,tol=1e-6,eps=1e-8,adaptive_criterion="msic",msic_refine_alpha=1.5,msic_refine_gamma=1.,msic_refine_one_se_scale=1.)
    if sys.argv[4]=="research": kw.update(cap_policy="legacy",grid_policy="full",candidate_budget=2000)
    result=tensor_abess(data["X"],data["y"],**kw)
payload={k:getattr(result,k) for k in ("loss","rss","sic","s1","s2","s3","s_max","loss_history","sic_history")}
payload.update(n_cycles=result.bcd_result.n_cycles,core_path=str(Path(sys.modules["tensor_abess"].__file__).resolve()))
np.savez(sys.argv[3]+".npz",B=result.B,S1=result.S1,S2=result.S2,S3=result.S3)
Path(sys.argv[3]+".json").write_text(json.dumps(payload,allow_nan=False),encoding="utf-8")
'''
            environment = {**os.environ, **{key:"1" for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS")}}
            for label, source in (("final",FINAL),("research",ROOT/"doc/algorithm_versions/v5.2-candidate-search/src")):
                subprocess.run([sys.executable,"-c",script,str(source),str(temp/"data.npz"),str(temp/label),label],cwd=ROOT,env=environment,check=True,capture_output=True)
            final = json.loads((temp/"final.json").read_text())
            research = json.loads((temp/"research.json").read_text())
            self.assertTrue(Path(final.pop("core_path")).is_relative_to(FINAL))
            self.assertTrue(Path(research.pop("core_path")).is_relative_to(ROOT/"doc/algorithm_versions/v5.2-candidate-search/src"))
            self.assertEqual(final,research)
            self.assertTrue(3 < min(final["s_max"]))
            with np.load(temp/"final.npz",allow_pickle=False) as a,np.load(temp/"research.npz",allow_pickle=False) as b:
                for key in ("B","S1","S2","S3"):
                    np.testing.assert_array_equal(a[key],b[key])


if __name__ == "__main__":
    unittest.main()
