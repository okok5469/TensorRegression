# Install and upgrade without overwriting unrelated files

Record your current import path and a SHA256 inventory first:

```bash
python -B -c "import tensor_abess; print(tensor_abess.__file__)"
```

Use your original Python environment. Core source imports require NumPy/SciPy.
requirements.txt lists runtime names only; it is not an instruction to upgrade.
If dependencies are absent, stop and agree an environment setup with the project
owner. Do not run an unreviewed pip/conda upgrade inside an empirical project.

## A. Existing project

Inspect local changes and preserve old results. The commands below are manual,
explicit actions, not an automatically executed upgrade script. Replace the
package path with the extracted directory. Use a fresh, nonexisting backup path.

PowerShell, from your project root:

```powershell
$package = 'C:/path/to/Tensor-ABESS-v5.2-search-final'
$oldCore = Join-Path $PWD 'src/tensor_abess'
$backup = Join-Path $PWD 'src/tensor_abess.before-v52'
if (Test-Path -LiteralPath $backup) { throw 'Backup already exists; stop' }
python -B "$package/scripts/check_version.py" --root "$package" --import-check
# Execute only after recording local changes and explicitly deciding to upgrade:
Move-Item -LiteralPath $oldCore -Destination $backup
Copy-Item -LiteralPath "$package/src/tensor_abess" -Destination $oldCore -Recurse
$env:PYTHONPATH = "$PWD/src"
python -B -c "import tensor_abess; print(tensor_abess.__file__)"
# Verify the target directly against the delivered manifest, then test your adapter:
python -B "$package/scripts/check_version.py" --source "$PWD/src/tensor_abess" --import-check
```

Linux, from your project root:

```bash
package='/path/to/Tensor-ABESS-v5.2-search-final'
test ! -e src/tensor_abess.before-v52 || exit 1
python -B "$package/scripts/check_version.py" --import-check
# Execute only after reviewing local changes and deciding to upgrade:
mv -- src/tensor_abess src/tensor_abess.before-v52
cp -a -- "$package/src/tensor_abess" src/tensor_abess
export PYTHONPATH="$PWD/src"
python -B -c "import tensor_abess; print(tensor_abess.__file__)"
python -B "$package/scripts/check_version.py" --source "$PWD/src/tensor_abess" --import-check
```

Never combine these actions with deletion of previous results or a forced Git
reset. A complete source replacement supports old versions without assuming a
particular patch baseline, but preserve project-specific modifications and review
their compatibility first. Changes outside src/tensor_abess are not included.

## B. Independent project

Keep the extracted package separate and point PYTHONPATH to its absolute src
directory. Confirm the actual import path, run the included checks, then use
your unchanged adapter/configuration. Python -I deliberately ignores PYTHONPATH;
the packaged smoke/check scripts support -I by explicitly binding their own src.

Always use a new version-labelled output directory. Keep the original response,
confounds, features and validation folds, refit all ABESS folds, and return complete
OOF plus reproducibility identities. No HCP/NKI dataset-specific CLI is invented.
