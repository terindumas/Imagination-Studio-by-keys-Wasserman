<#
.SYNOPSIS
  Download LTX-Video 2B or Wan 2.1 1.3B into the existing Windows x64 GPU venv.
.DESCRIPTION
  For a discrete NVIDIA GPU of 16 GB on native Windows AMD64. FastH3 and MiniMax H3 are not installed.
  Reuses %LOCALAPPDATA%\ImaginationStudio\x64\gpu-venv. Does not open a port.
  -Rollback removes only that model's folder under x64\models. Stills and the venv stay.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-video.ps1 -Model ltx -PlanOnly
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-video.ps1 -Model ltx
#>
[CmdletBinding()]
param(
  [ValidateSet('auto', 'ltx', 'wan13')]
  [string]$Model = 'auto',
  [switch]$PlanOnly,
  [switch]$Rollback
)
$ErrorActionPreference = 'Stop'
# tqdm and Hugging Face write progress to stderr. That is not a failed command.
$PSNativeCommandUseErrorActionPreference = $false
$Repo = Split-Path $PSScriptRoot -Parent
$Helper = Join-Path $Repo 'app\windows_x64_video.py'
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
  # A function's stdout is its return value. The helper's JSON must not be mixed into the exit code,
  # and a pipeline would replace the native exit code, so the log goes to a file first.
  $PSNativeCommandUseErrorActionPreference = $false
  $log = Join-Path $env:TEMP ("studio-x64-" + [guid]::NewGuid().ToString() + ".log")
  try {
    if ($Prefix.Length -gt 1) { & $Prefix[0] $Prefix[1] @Arguments *> $log }
    else { & $Prefix[0] @Arguments *> $log }
    $code = $LASTEXITCODE
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log | Write-Host }
    if ($null -eq $code) { return 1 }
    return $code
  } finally {
    Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue
  }
}
function Invoke-Helper([string[]]$Prefix, [string[]]$HelperArgs) {
  $code = Invoke-Python $Prefix (@('-X', 'utf8', $Helper) + $HelperArgs)
  if ($code -ne 0) { throw 'Windows x64 video setup stopped. Stills and the GPU venv were not removed.' }
}

$Launcher = Get-SetupPython
if ($PlanOnly -and -not $Rollback) {
  Invoke-Helper $Launcher @('--plan', '--model', $Model)
  exit 0
}
if ($Rollback) {
  $flag = if ($PlanOnly) { '--rollback-plan' } else { '--rollback' }
  Invoke-Helper $Launcher @($flag, '--model', $Model)
  if (-not $PlanOnly) { Write-Host 'Removed that video model only. The x64 venv and stills weights are still there.' }
  exit 0
}
if ($env:OS -ne 'Windows_NT') { throw 'This video installer runs on native Windows AMD64. Nothing was downloaded.' }
if ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64' -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
  throw 'This PC is Windows on Arm. FastH3 stays on that installer. Nothing was downloaded.'
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
  throw "This installer requires native Windows AMD64. PROCESSOR_ARCHITECTURE is '$($env:PROCESSOR_ARCHITECTURE)'."
}

$GpuPython = Join-Path $env:LOCALAPPDATA 'ImaginationStudio\x64\gpu-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $GpuPython)) {
  & (Join-Path $PSScriptRoot 'setup-windows-x64-images.ps1') -NoDownload
  if ($LASTEXITCODE -ne 0) { throw 'The x64 GPU venv was not created. No video weights were downloaded.' }
}
$code = Invoke-Python @($GpuPython) @('-m', 'pip', 'install', '-r', $Requirements)
if ($code -ne 0) { throw 'The video Python packages did not install. No weights were downloaded.' }
$code = Invoke-Python @($GpuPython) @('-X', 'utf8', $Helper, '--check-tokenizer')
if ($code -ne 0) { throw 'The tokenizer packages did not import. No weights were downloaded.' }
Invoke-Helper @($GpuPython) @('--download', '--model', $Model)
Write-Host 'Windows x64 video weights are installed. Restart Studio, then open http://127.0.0.1:7870'
Write-Host 'FastH3 and MiniMax H3 are not supported on a 16 GB GPU. The page offers LTX-Video or Wan 2.1 instead.'
