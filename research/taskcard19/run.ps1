$ErrorActionPreference = 'Stop'
python -m tensor_abess_ovr.cli @args
exit $LASTEXITCODE
