param([string]$StorageRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'output\local'))
$ErrorActionPreference = 'Stop'
$absolute = [IO.Path]::GetFullPath($StorageRoot)
if (-not $absolute.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Local downloads, caches and temporary files must use D:\.'
}
$env:PIP_CACHE_DIR = Join-Path $absolute 'pip-cache'
$env:HF_HOME = Join-Path $absolute 'huggingface'
$env:TORCH_HOME = Join-Path $absolute 'torch'
$env:TEMP = Join-Path $absolute 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
foreach ($directory in @($env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_HOME, $env:TEMP)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
Write-Host "This shell now uses D storage: $absolute"
