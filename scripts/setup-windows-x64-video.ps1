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
$Repo = Split-Path $PSScriptRoot -Parent
$Helper = Join-Path $Repo 'app\windows_x64_video.py'
$Requirements = Join-Path $Repo 'app\requirements-windows-x64.txt'

function Format-ProcessArg([string]$Text) {
  # CommandLineToArgvW rules. Windows PowerShell 5.1 re-quotes a single argument string,
  # and the cmd /s switch strips one quote (that ran ""py").
  if ($Text -notmatch '[\s"]') { return $Text }
  $escaped = $Text -replace '(\\*)"', '$1$1\"'
  $escaped = $escaped -replace '(\\+)$', '$1$1'
  return '"' + $escaped + '"'
}
function Invoke-Python([string[]]$Prefix, [string[]]$Arguments) {
  # Windows PowerShell 5.1 turns a native stderr redirect into a terminating error while
  # $ErrorActionPreference is Stop. The process is started with redirected streams and
  # no window. Its ExitCode is the only failure signal. stderr is printed, not trusted.
  $preference = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $exe = $Prefix[0]
    $command = Get-Command -Name $exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source) { $exe = $command.Source }
    $argParts = @()
    if ($Prefix.Length -gt 1) { $argParts += $Prefix[1..($Prefix.Length - 1)] }
    if ($Arguments) { $argParts += $Arguments }
    $quoted = foreach ($part in $argParts) { Format-ProcessArg ([string]$part) }
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.Arguments = ($quoted -join ' ')
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $info
    if (-not $proc.Start()) { return 1 }
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    if ($stdout) { Write-Host $stdout }
    if ($stderr) { Write-Host $stderr }
    return [int]$proc.ExitCode
  } finally {
    $ErrorActionPreference = $preference
  }
}
function Get-SetupPython {
  $probe = @('-c', 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)')
  if (Get-Command py -ErrorAction SilentlyContinue) {
    if ((Invoke-Python @('py', '-3.13') $probe) -eq 0) { return @('py', '-3.13') }
  }
  foreach ($name in @('python', 'python3')) {
    if (Get-Command $name -ErrorAction SilentlyContinue) {
      if ((Invoke-Python @($name) $probe) -eq 0) { return @($name) }
    }
  }
  throw 'Python 3.13 is required (py -3.13). Nothing was downloaded.'
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
