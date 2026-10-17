param([string]$Python = (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv-agent\Scripts\python.exe'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path -LiteralPath $Python)) { throw 'Install the dedicated environment first or pass -Python explicitly.' }
Push-Location $root
try {
    . (Join-Path $PSScriptRoot 'use-d-storage.ps1')
    & $Python tools/check_repository.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository checks failed.' }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Whitespace checks failed.' }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency checks failed.' }
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Software tests failed.' }
} finally {
    Pop-Location
}
