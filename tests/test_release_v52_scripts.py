"""Release deployment contracts: no network calls in local tests."""
import importlib.util
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release_v52/release_control.py"
BASH = Path("C:/Program Files/Git/bin/bash.exe")


class ReleaseScriptsTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), "release control contract is missing")
        spec = importlib.util.spec_from_file_location("release_control", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_inventory_and_canonical_verification_detect_tampering(self):
        control = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tensor_abess/a.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"a = 1\r\n")
            manifest = {"source_canonical_lf_sha256": {"src/tensor_abess/a.py": control.canonical(source)}}
            control.verify_files(root, manifest)
            source.write_bytes(b"a = 2\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                control.verify_files(root, manifest)

    def test_sync_plan_only_changes_bcd_and_stops_unknown_source(self):
        control = self.load()
        manifest = json.loads((ROOT / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        remote = control.inventory(ROOT)
        bcd = "src/tensor_abess/tensor/bcd.py"
        remote["files"][bcd]["canonical_sha256"] = manifest["base_source_canonical_lf_sha256"][bcd]
        plan = control.sync_plan(ROOT, manifest, remote)
        self.assertEqual([entry["path"] for entry in plan["updates"] if entry["path"].startswith("src/")], [bcd])
        remote["files"]["src/tensor_abess/tensor/ops.py"]["canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "conflict"):
            control.sync_plan(ROOT, manifest, remote)

    def test_deployment_identity_rejects_modified_launch_script(self):
        control = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/tensor_abess/a.py"
            source.parent.mkdir(parents=True)
            source.write_text("a=1\n", encoding="utf-8")
            launch = root / "run_v52_search_final_full.sh"
            launch.write_text("verified\n", encoding="utf-8")
            manifest = {"source_canonical_lf_sha256": {"src/tensor_abess/a.py": control.canonical(source)},
                        "deployment_canonical_lf_sha256": {launch.name: control.canonical(launch)}}
            launch.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                control.verify_files(root, manifest)

    def test_known_frozen_previous_launch_can_be_upgraded(self):
        control = self.load()
        manifest = json.loads((ROOT / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        old = control.canonical(ROOT / "doc/team_sync/v5.2-search-final/run_v52_search_final_full.sh")
        manifest["base_deployment_canonical_lf_sha256"] = {"run_v52_search_final_full.sh": old}
        remote = control.inventory(ROOT)
        remote["files"]["run_v52_search_final_full.sh"]["canonical_sha256"] = old
        try:
            plan = control.sync_plan(ROOT, manifest, remote)
        except ValueError as error:
            self.fail("Recognized frozen previous launcher must be backed up and updated: " + str(error))
        self.assertIn("run_v52_search_final_full.sh", [entry["path"] for entry in plan["updates"]])

    def test_sync_dry_run_accepts_chinese_spaces_without_network(self):
        self.load()
        with tempfile.TemporaryDirectory(prefix="release 中文 空格 ") as directory:
            remote_file = Path(directory) / "inventory.json"
            remote_file.write_text(json.dumps(self.load().inventory(ROOT)), encoding="utf-8")
            result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(ROOT / "scripts/release_v52/sync_to_cluster.ps1"), "-SshHost", "user@cluster",
                "-LocalRoot", str(ROOT), "-InventoryPath", str(remote_file)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY_RUN", result.stdout)

    def test_cluster_shell_syntax_and_existing_environment(self):
        self.load()
        paths = [ROOT / "run_v52_search_final_full.sh", *list((ROOT / "scripts/release_v52").glob("*.sh"))]
        for path in paths:
            result = subprocess.run([str(BASH), "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        shell = paths[0].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-699%64", shell)
        self.assertNotIn("source ", shell)
        self.assertIn("status)", shell)
        self.assertIn("--methods adaptive oracle lasso glasso", shell)
        self.assertIn("#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v52_search_final_full_logs/%A_%a.out", shell)
        self.assertIn("#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v52_search_final_full_logs/%A_%a.err", shell)
        self.assertIn('task_repo="${V52_REMOTE_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main}"', shell)

    def test_site_module_activates_conda_without_extra_source(self):
        with tempfile.TemporaryDirectory(prefix="cluster39 ") as directory:
            temp = Path(directory)
            startup = temp / "startup.sh"
            startup.write_text('module(){ conda(){ printf "%s" "$V52_UNSET_SITE_OPTION" >/dev/null; }; }\npython(){ :; }\n', encoding="utf-8")
            environment = {**os.environ, "BASH_ENV": startup.as_posix(),
                           "V52_REMOTE_ROOT": temp.as_posix()}
            environment.pop("V52_UNSET_SITE_OPTION", None)
            environment.pop("V52_CONDA_INIT", None)
            result = subprocess.run([str(BASH), str(ROOT / "run_v52_search_final_full.sh"), "status"],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, "Module-provided conda must activate without a hardcoded source dependency: " + result.stderr)

    def test_prepare_creates_slurm_log_parent_before_failing_validation(self):
        with tempfile.TemporaryDirectory(prefix="cluster-log-parent ") as directory:
            temp = Path(directory)
            startup = temp / "startup.sh"
            startup.write_text("module(){ conda(){ :; }; }\npython(){ return 42; }\n", encoding="utf-8")
            environment = {**os.environ, "BASH_ENV": startup.as_posix(), "V52_REMOTE_ROOT": temp.as_posix()}
            result = subprocess.run([str(BASH), str(ROOT / "run_v52_search_final_full.sh"), "prepare"],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 42)
            self.assertTrue((temp / "numerical_simulation/v52_search_final_full_logs").is_dir(),
                            "Log parent must exist before verification can stop preparation")

    def test_job_record_accepts_relative_worker_log_without_changing_result(self):
        control = self.load()
        with tempfile.TemporaryDirectory(prefix="slurm-log-path ") as directory:
            root = Path(directory).resolve()
            output = root / "numerical_simulation/outputs_v52_search_final_full"
            relative_log = Path("numerical_simulation/outputs_v52_search_final_full/slurm/task_000000_123_999.log")
            log = root / relative_log
            log.parent.mkdir(parents=True)
            log.write_text("task_id=0 completed\n", encoding="utf-8")
            before = log.read_bytes()
            fake = SimpleNamespace(read_task=lambda *args: {"record_sha256": "existing-result"},
                                   commit_exclusive=lambda path, data: path.write_bytes(data),
                                   json_bytes=lambda body: json.dumps(body).encode("utf-8"),
                                   digest=lambda body: "unit-test-checksum")
            environment = {"SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "123",
                           "SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_COUNT": "1"}
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(control, "checked_plan", return_value=(fake, {"plan_id": "existing-plan"})), patch.dict(os.environ, environment):
                    try:
                        record = control.job_record(root, output, relative_log, "completed")
                    except ValueError as error:
                        self.fail("Relative worker log must resolve against the absolute output: " + str(error))
            finally:
                os.chdir(previous)
            self.assertEqual(record["log"], "slurm/" + log.name)
            self.assertEqual(record["record_sha256"], "existing-result")
            self.assertTrue(log.with_suffix(".json").is_file())
            self.assertEqual(log.read_bytes(), before)

    def test_actual_package_backup_and_hash_locked_apply_in_staging(self):
        control = self.load()
        manifest = json.loads((ROOT / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="release remote 中文 ") as directory:
            remote = Path(directory)
            paths = list(manifest["source_sha256"]) + list(manifest["simulation_sha256"]) + control.DEPLOY
            for name in paths:
                destination = remote / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            bcd = "src/tensor_abess/tensor/bcd.py"
            shutil.copy2(ROOT / "doc/algorithm_versions/v5.1-bcd-fix/src/tensor_abess/tensor/bcd.py", remote / bcd)
            plan = control.sync_plan(ROOT, manifest, control.inventory(remote))
            archive = remote / ".v52-release-staging/reviewed/upload.tar"
            archive.parent.mkdir(parents=True)
            control.package(ROOT, plan, archive)
            original_resolve = Path.resolve
            alias_target = remote / "history/bcd.py"
            def redirected(path, *args, **kwargs):
                return alias_target if path == remote / bcd else original_resolve(path, *args, **kwargs)
            with patch.object(Path, "resolve", autospec=True, side_effect=redirected):
                with self.assertRaisesRegex(ValueError, "Unsafe remote path"):
                    control.apply_package(remote, archive)
            self.assertEqual(control.sha(remote / bcd), manifest["base_source_sha256"][bcd])
            try:
                result = control.apply_package(remote, archive)
            except ValueError as error:
                self.fail("Reviewed package inside its designated staging must apply: " + str(error))
            self.assertEqual(result["status"], "APPLIED")
            self.assertEqual(control.sha(remote / bcd), manifest["source_sha256"][bcd])
            self.assertEqual(control.sha(Path(result["backup"]) / bcd), manifest["base_source_sha256"][bcd])

    def test_powershell_apply_passes_literal_sftp_path_with_spaces(self):
        control = self.load()
        with tempfile.TemporaryDirectory(prefix="sync 中文 空格 ") as directory:
            temp = Path(directory)
            inventory = temp / "inventory.json"
            inventory.write_text(json.dumps(control.inventory(ROOT)), encoding="utf-8")
            wrapper = temp / "mock.ps1"
            wrapper.write_text('''param($ScriptPath,$LocalRoot,$InventoryPath,$Capture)
function global:ssh { $global:LASTEXITCODE=0; if (($args -join ' ') -match 'p.mkdir') { '/remote space/.v52-release-staging/test' } else { 'APPLIED' } }
function global:scp { [IO.File]::WriteAllText($Capture,($args | ConvertTo-Json -Compress)); $global:LASTEXITCODE=0 }
& $ScriptPath -SshHost user@cluster -LocalRoot $LocalRoot -RemoteRoot '/remote space' -InventoryPath $InventoryPath -Apply
''', encoding="utf-8")
            capture = temp / "args.json"
            result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
                                     str(ROOT / "scripts/release_v52/sync_to_cluster.ps1"), str(ROOT), str(inventory), str(capture)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = json.loads(capture.read_text(encoding="utf-8-sig"))
            self.assertEqual(arguments[-1], "user@cluster:/remote space/.v52-release-staging/test/upload.tar")

    def test_download_does_not_accept_partial_merge_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            wrapper = temp / "mock-download.ps1"
            wrapper.write_text('''param($ScriptPath,$Destination)
function global:scp { [IO.File]::WriteAllText($args[-1],'{"task_count":1,"method_result_count":4,"integrity_checks_passed":true}'); $global:LASTEXITCODE=0 }
& $ScriptPath -SshHost user@cluster -Destination $Destination -Apply
''', encoding="utf-8")
            result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
                                     str(ROOT / "scripts/release_v52/download_results.ps1"), str(temp / "new-results")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, "Partial 1/700 summaries must not be accepted")


if __name__ == "__main__":
    unittest.main()
