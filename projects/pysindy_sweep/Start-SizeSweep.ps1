param(
    [ValidateSet('Clean', 'Pilot', 'Scalars', 'Systems')]
    [string]$Stage = 'Clean',
    [string]$OutputDir,
    [switch]$PlanOnly,
    [switch]$ValidateData,
    [switch]$Resume,
    [int]$LimitJobs = 0
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$taskRunner = Join-Path $PSScriptRoot 'run_size_sweep.py'
$taskSummary = Join-Path $PSScriptRoot 'summarize_size_sweep.py'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create projects/pysindy_sweep/.venv using the README instructions first.'
}
if (-not $OutputDir) {
    $taskDate = Get-Date -Format 'yyyyMMdd'
    $OutputDir = Join-Path $PSScriptRoot "outputs/size_$($Stage.ToLower())_$taskDate"
}
$taskArguments = @($taskRunner, '--output-dir', $OutputDir)
if ($Stage -eq 'Pilot') {
    $taskArguments += @('--grid', (Join-Path $PSScriptRoot 'size_pilot_grid.json'))
}
if ($Stage -eq 'Scalars') {
    $taskArguments += @('--datasets', 'ode_data.npy', 'vdp_data.npy', 'burgers_data.mat',
        'burgers_sln_100_data.csv', 'ac_data.npy', 'kdv_data.mat', 'kdv_periodic_data.npy',
        'wave_data.csv', 'pde_compound_data.npy', 'pde_divide_data.npy', 'ks_data.mat',
        'ODE_simple_discovery')
}
if ($Stage -eq 'Systems') {
    $taskArguments += @('--datasets', 'lotka_data.npy', 'lorenz_data.npy', 'ns_data.mat')
}
if ($PlanOnly) { $taskArguments += '--plan' }
if ($ValidateData) { $taskArguments += '--validate-data' }
if ($Resume) { $taskArguments += '--resume' }
if ($LimitJobs -gt 0) { $taskArguments += @('--limit-jobs', "$LimitJobs") }
& $taskPython @taskArguments
$taskExitCode = $LASTEXITCODE
if (-not $PlanOnly -and $taskExitCode -in @(0, 1, 130)) {
    & $taskPython $taskSummary $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Result aggregation failed; checkpoints are retained.' }
}
if ($taskExitCode -ne 0) { throw "Sweep exited with code $taskExitCode; see run_status.json and job status files." }
