<#
.SYNOPSIS
  Install Windows x64 CUDA stills: a GPU venv, diffusers, and a public safetensors image model.
.DESCRIPTION
  For native Windows 11 AMD64 with a discrete NVIDIA GPU (tested target: RTX 4090 Laptop, 16 GB).
  This is not the Windows on Arm / RTX Spark installer. That path stays scripts\setup-windows-images.ps1.
  Nothing here opens a port. Studio keeps listening on 127.0.0.1:7870. Ports 8080 and 1420 are not used.
  -Rollback removes only %LOCALAPPDATA%\ImaginationStudio\x64.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model auto -PlanOnly
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model auto
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Rollback
#>
[CmdletBinding()]
param(
  [ValidateSet('auto', 'z-image-turbo', 'sd15')]
  [string]$Model = 'auto',
  [switch]$PlanOnly,
  [switch]$NoDownload,
  [switch]$Rollback
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
$Helper = Join-Path $Repo 'app\windows_x64_setup.py'
$Requirements = Join-Path $Repo 'app\requirements-windows-x64.txt'

function Get-SetupPython {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"
    if ($LASTEXITCODE -eq 0) { return @('py', '-3.13') }
  }
  foreach ($name in @('python', 'python3')) {
    if (Get-Command $name -ErrorAction SilentlyContinue) {
      & $name -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"
      if ($LASTEXITCODE -eq 0) { return @($name) }
    }
  }
  throw 'Python 3.13 is required (py -3.13). Nothing was downloaded.'
}

function Invoke-Python([string[]]$Prefix, [string[]]$Arguments) {
  if ($Prefix.Length -gt 1) { & $Prefix[0] $Prefix[1] @Arguments }
  else { & $Prefix[0] @Arguments }
  return $LASTEXITCODE
}
function Invoke-Helper([string[]]$Prefix, [string[]]$HelperArgs) {
  $code = Invoke-Python $Prefix (@('-X', 'utf8', $Helper) + $HelperArgs)
  if ($code -ne 0) { throw 'Windows x64 image setup stopped. Nothing outside ImaginationStudio\x64 was changed.' }
}

$Launcher = Get-SetupPython
if ($PlanOnly -and -not $Rollback) {
  Invoke-Helper $Launcher @('--plan', '--model', $Model)
  exit 0
}

if ($Rollback) {
  $flag = if ($PlanOnly) { '--rollback-plan' } else { '--rollback' }
  Invoke-Helper $Launcher @($flag)
  if (-not $PlanOnly) { Write-Host 'Removed the Windows x64 stills venv and weights (%LOCALAPPDATA%\ImaginationStudio\x64).' }
  exit 0
}

if ($env:OS -ne 'Windows_NT') {
  throw 'This CUDA stills installer runs on native Windows AMD64. Nothing was downloaded.'
}
if ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64' -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
  throw 'This PC is Windows on Arm. Use scripts\setup-windows-images.ps1. The x64 installer did not change anything.'
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
  throw "This installer requires native Windows AMD64. PROCESSOR_ARCHITECTURE is '$($env:PROCESSOR_ARCHITECTURE)'."
}

$Root = Join-Path $env:LOCALAPPDATA 'ImaginationStudio\x64'
$Venv = Join-Path $Root 'gpu-venv'
$GpuPython = Join-Path $Venv 'Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
if (-not (Test-Path -LiteralPath $GpuPython)) {
  $code = Invoke-Python $Launcher @('-m', 'venv', $Venv)
  if ($code -ne 0) { throw 'Could not create the x64 GPU venv. Nothing was downloaded.' }
}
& $GpuPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip could not be upgraded in the x64 GPU venv.' }
# cu128 publishes a Windows AMD64 wheel for CPython 3.13. RTX 4090 (sm_89) runs that wheel.
& $GpuPython -m pip install 'torch==2.9.1+cu128' torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'The Windows x64 CUDA PyTorch wheel did not install. The model was not downloaded.' }
& $GpuPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'diffusers and its dependencies did not install. The model was not downloaded.' }
& $GpuPython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw 'PyTorch imported but CUDA is not available. Check the NVIDIA driver. Weights were not downloaded. -Rollback removes this venv.'
}
if ($NoDownload) {
  Write-Host 'GPU venv is ready. -NoDownload left the weights untouched.'
  exit 0
}
Invoke-Helper @($GpuPython) @('--download', '--model', $Model)
Write-Host 'Windows x64 stills are installed. Restart Studio, then open http://127.0.0.1:7870'
Write-Host 'A generation test belongs to the queue. This installer did not render and did not open a port.'
