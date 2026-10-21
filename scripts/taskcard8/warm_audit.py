"""Run unchanged BCD assertions in one explicitly isolated source tree.

Plain pytest-style files here need no fixtures and only contain an unused
``import pytest``. The AST adapter removes that exact unused import, retaining
all test bodies and assertions. Any actual pytest use is rejected explicitly.
"""
from __future__ import annotations

import ast
import hashlib
import types
import argparse
import importlib.util
import inspect
import io
import json
import sys
import time
import unittest
from pathlib import Path


def source_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def load_plain_tests(path: Path, name: str) -> types.ModuleType:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports = [node for node in tree.body if isinstance(node, ast.Import)
               and len(node.names) == 1 and node.names[0].name == "pytest"
               and node.names[0].asname is None]
    if any(isinstance(node, ast.Name) and node.id == "pytest" for node in ast.walk(tree)):
        raise ValueError("Actual pytest usage requires the real pytest runner")
    tree.body = [node for node in tree.body if node not in imports]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def summarize_result(result: unittest.TestResult) -> dict:
    passed = result.wasSuccessful() and result.testsRun > 0
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped),
        "correctness_passed": passed, "performance_allowed": passed,
        "failure_details": [{"test": test.id(), "traceback": trace}
                            for test, trace in result.failures],
        "error_details": [{"test": test.id(), "traceback": trace}
                          for test, trace in result.errors],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = arguments.source.resolve()
    sys.path[:0] = [str(source), str(root)]
    import tensor_abess.tensor.bcd as bcd
    assert Path(bcd.__file__).resolve() == source / "tensor_abess/tensor/bcd.py"
    manifest = source_manifest(source / "tensor_abess")
    assert len(manifest) == 32, "The audit requires all 32 core Python files"
    loaded_modules = {name: str(Path(module.__file__).resolve())
                      for name, module in sys.modules.items()
                      if name.startswith("tensor_abess") and getattr(module, "__file__", None)}
    assert all(Path(path).is_relative_to(source) for path in loaded_modules.values())
    suite = unittest.TestSuite()
    for path, name in (
        (root / "tests/test_tensor_bcd.py", "original_fixed_bcd"),
        (root / "doc/algorithm_versions/v5.3-warm-start-research/incoming/test_tensor_bcd.py", "uploaded_fixed_bcd"),
    ):
        module = load_plain_tests(path, name)
        functions = [(name, value) for name, value in vars(module).items()
                     if name.startswith("test_") and inspect.isfunction(value)]
        assert all(not inspect.signature(function).parameters for _, function in functions)
        suite.addTests(unittest.FunctionTestCase(function, description=f"{name}.{test_name}")
                       for test_name, function in functions)
    for path, name in (
        (root / "tests/test_bcd_fix_regression.py", "fixed_regression"),
        (root / "scripts/taskcard8/test_warm_audit.py", "audit_checks"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    # Invoke the exact historical method and untouched data via its original module.
    spec = importlib.util.spec_from_file_location("historical_simulation", root / "tests/test_numerical_simulation.py")
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    suite.addTest(historical.NumericalSimulationTests(
        "test_full_profile_fixed_size_bcd_can_recover_from_marginal_warm_start"))
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    loaded_modules = {name: str(Path(module.__file__).resolve())
                      for name, module in sys.modules.items()
                      if name.startswith("tensor_abess") and getattr(module, "__file__", None)}
    assert all(Path(path).is_relative_to(source) for path in loaded_modules.values())
    assert source_manifest(source / "tensor_abess") == manifest, "Core sources changed during the audit"
    report = summarize_result(result)
    report.update({
        "source": str(source), "bcd_import": str(Path(bcd.__file__).resolve()),
        "loaded_core_modules": loaded_modules,
        "core_file_count": len(manifest), "sha256": manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "executable": sys.executable, "python_version": sys.version,
        "test_adapter": "standard-library unittest; only unused standalone pytest import omitted in memory; assertions/data unchanged",
        "protected_inputs": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (root / "tests/test_tensor_bcd.py", root / "tests/test_bcd_fix_regression.py",
                                          root / "tests/test_numerical_simulation.py",
                                          root / "doc/algorithm_versions/v5.3-warm-start-research/incoming/bcd.py",
                                          root / "doc/algorithm_versions/v5.3-warm-start-research/incoming/test_tensor_bcd.py")},
    })
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "correctness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (arguments.output / "unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("tests_run", "failures", "errors", "correctness_passed", "elapsed_seconds")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
