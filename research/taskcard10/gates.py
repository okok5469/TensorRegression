"""Run unchanged Cold regression and isolated CV/control/audit contracts."""
import importlib.util
import inspect
import io
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
import unittest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parents[2]
    output = root/'doc/taskcard10/CORRECTNESS_GATES.json'
    assert not output.exists(), 'Preserve the original gate result; choose an explicitly new research run after failures'
    source = root/'doc/algorithm_versions/v5.2-search-final/src'
    sys.path[:0] = [str(source),str(root)]
    import tensor_abess.tensor.bcd as bcd
    assert Path(bcd.__file__).resolve().is_relative_to(source)
    adapter = load(root/'scripts/taskcard8/warm_audit.py','task10_cold_adapter')
    core = adapter.source_manifest(source/'tensor_abess')
    assert len(core) == 32
    suite = unittest.TestSuite()
    plain = adapter.load_plain_tests(root/'tests/test_tensor_bcd.py','task10_original_fixed')
    original_count = 0
    for name,fn in vars(plain).items():
        if name.startswith('test_') and inspect.isfunction(fn):
            suite.addTest(unittest.FunctionTestCase(fn,description=name))
            original_count += 1
    original = load(root/'tests/test_bcd_fix_regression.py','task10_original_guards')
    original_suite = unittest.defaultTestLoader.loadTestsFromModule(original)
    original_count += original_suite.countTestCases()
    suite.addTests(original_suite)
    historical = load(root/'tests/test_numerical_simulation.py','task10_historical_original')
    suite.addTest(historical.NumericalSimulationTests('test_full_profile_fixed_size_bcd_can_recover_from_marginal_warm_start'))
    original_count += 1
    assert original_count == 16
    text = io.StringIO()
    started = perf_counter()
    result = unittest.TextTestRunner(stream=text,verbosity=2).run(suite)
    separate = []
    for name in ('test_cv','test_audit','test_experiment'):
        process = subprocess.run([sys.executable,'-B','-m','unittest','research.taskcard10.'+name],
                                 cwd=root,capture_output=True,text=True,timeout=180)
        log = process.stdout + process.stderr
        match = re.search(r'Ran (\d+) tests?',log)
        separate.append({'module':name,'returncode':process.returncode,
                         'tests_run':int(match.group(1)) if match else 0})
        text.write('\n'+name+' (fresh isolated process)\n'+log)
    actual = {name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
              if (name == 'tensor_abess' or name.startswith('tensor_abess.')) and getattr(module,'__file__',None)}
    assert all(Path(path).is_relative_to(source) for path in actual.values())
    assert core == adapter.source_manifest(source/'tensor_abess')
    passed = result.wasSuccessful() and all(r['returncode'] == 0 and r['tests_run'] > 0 for r in separate)
    body = {'passed':passed,'tests_run':result.testsRun + sum(r['tests_run'] for r in separate),'unchanged_cold_checks':original_count,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'isolated_module_runs':separate,
            'elapsed_seconds':perf_counter()-started,'core_sha256':core,'actual_imports':actual,
            'source_and_tests_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*sorted((root/'research/taskcard10').glob('*.py')),
                          root/'scripts/taskcard8/warm_audit.py',root/'tests/test_tensor_bcd.py',
                          root/'tests/test_bcd_fix_regression.py',root/'tests/test_numerical_simulation.py']},
            'threads':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'scope':'Original Cold16 plus new CV, cache, leak, protocol and audit tests; no Warm tests or scientific Pilot'}
    with output.open('x',encoding='utf-8') as handle:
        json.dump(body,handle,indent=2)
    with output.with_suffix('.log').open('x',encoding='utf-8') as handle:
        handle.write(text.getvalue())
    print(json.dumps({k:body[k] for k in ('passed','tests_run','unchanged_cold_checks','failures','errors','elapsed_seconds')}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
