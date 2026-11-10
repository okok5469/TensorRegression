"""File-content identity for source and installed release verification."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_root(name):
    module = importlib.import_module(name)
    if not getattr(module, "__file__", None):
        raise RuntimeError(f"package missing: {name}")
    return Path(module.__file__).resolve().parent


def runtime_identity():
    """Measure the actual imported numerical runtime and reject lock drift."""
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("Python lock mismatch: this application requires CPython 3.12")
    if platform.python_implementation() != "CPython":
        raise RuntimeError("Python lock mismatch: CPython is required")
    locked = {"numpy": "1.26.4", "scipy": "1.16.1", "scikit-learn": "1.5.2", "joblib": "1.5.2", "threadpoolctl": "3.6.0"}
    names = {"scikit-learn": "sklearn"}
    actual = {}
    imported = {}
    for distribution, expected in locked.items():
        module = importlib.import_module(names.get(distribution, distribution))
        actual[distribution] = importlib.metadata.version(distribution)
        imported[distribution] = str(Path(module.__file__).resolve())
        if actual[distribution] != expected or getattr(module, "__version__", None) != expected:
            raise RuntimeError(f"dependency lock mismatch: {distribution}; installed={actual[distribution]}, imported={getattr(module, '__version__', None)}, required={expected}")
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    if not pools or any(pool.get("num_threads") != 1 for pool in pools):
        raise RuntimeError(f"numerical thread lock mismatch: expected all active threadpools=1, actual={pools}. Restart through the CLI or set the six thread environment variables to 1 before importing any numerical library; an already initialized multithreaded runtime is not accepted.")
    for name in ("tensor_abess", "tensor_abess_ovr"):
        module = importlib.import_module(name)
        imported[name] = str(Path(module.__file__).resolve())
    return {"python": platform.python_version(), "python_implementation": platform.python_implementation(), "python_executable": str(Path(sys.executable).resolve()), "dependencies": actual, "actual_import_files": imported, "threadpools": pools}


def code_identity():
    roots = {name: package_root(name) for name in ("tensor_abess", "tensor_abess_ovr")}
    return {
        "application_version": "0.1.0",
        "core_version": "v5.2-search-final",
        "packages": {name: {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))} for name, root in roots.items()},
        "core_profile_sha256": sha256(roots["tensor_abess_ovr"] / "core_profile.json"),
    }


def verify_release():
    runtime = runtime_identity()
    wrapper = package_root("tensor_abess_ovr")
    manifest_path = wrapper / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("RELEASE_MANIFEST.json missing; build/install the release before verify-release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = {"core_sha256": package_root("tensor_abess"), "wrapper_sha256": wrapper}
    counts = {}
    for field, root in roots.items():
        expected = manifest[field]
        actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
        expected_names = {name for name in expected if name.endswith(".py")}
        if actual_names != expected_names:
            raise RuntimeError(f"{field} file inventory mismatch: missing={expected_names-actual_names}, extra={actual_names-expected_names}")
        for relative, digest in expected.items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha256(target) != digest:
                raise RuntimeError(f"release identity mismatch: {field}/{relative}")
        counts[field] = len(expected_names)
    if counts["core_sha256"] != 32:
        raise RuntimeError("frozen core must have exactly 32 Python files")
    return {"status": "PASS", "counts": counts, "import_paths": {field: str(root) for field, root in roots.items()}, "platform": platform.platform(), "python": platform.python_version(), "runtime": runtime, "verified_sha256": {field: manifest[field] for field in roots}}
