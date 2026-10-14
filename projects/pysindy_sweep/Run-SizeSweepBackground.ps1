param([Parameter(Mandatory=$true)][string]$OutputDir)

$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskOutput = [System.IO.Path]::GetFullPath($OutputDir)
$taskAllowed = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
if (-not $taskOutput.StartsWith($taskAllowed, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must stay within projects/pysindy_sweep.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'manifest.json'))) {
    throw 'A prepared manifest is required.'
}

# Hold an OS file lock for the entire runner/postprocessing lifecycle.
# The lock is released automatically if the helper process exits or is killed.
$taskLock = [System.IO.File]::Open((Join-Path $taskOutput 'background.lock'),
    [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None)
$taskStatusFile = Join-Path $taskOutput 'background_job.json'
$taskStatus = [ordered]@{
    experiment = (Split-Path -Leaf $taskOutput)
    helper_pid = $PID
    runner_pid = $null
    summary_pid = $null
    state = 'starting'
    started_utc = [DateTime]::UtcNow.ToString('o')
    updated_utc = $null
    runner_exit_code = $null
    summary_exit_code = $null
    error = $null
}

function Save-TaskStatus {
    $taskStatus.updated_utc = [DateTime]::UtcNow.ToString('o')
    $taskTemporary = "$taskStatusFile.tmp"
    $taskStatus | ConvertTo-Json | Set-Content -LiteralPath $taskTemporary -Encoding UTF8
    Move-Item -LiteralPath $taskTemporary -Destination $taskStatusFile -Force
}

try {
    Save-TaskStatus
    $taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $taskRunner = Join-Path $PSScriptRoot 'run_size_sweep.py'
    $taskArguments = @('-u', ('"' + $taskRunner + '"'), '--resume', '--output-dir', ('"' + $taskOutput + '"'))
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'runner.log') `
        -RedirectStandardError (Join-Path $taskOutput 'runner_error.log')
    $taskStatus.runner_pid = $taskProcess.Id
    $taskStatus.state = 'running'
    Save-TaskStatus
    $taskProcess.WaitForExit()
    $taskStatus.runner_exit_code = $taskProcess.ExitCode

    $taskStatus.state = 'aggregating'
    Save-TaskStatus
    $taskSummary = Join-Path $PSScriptRoot 'summarize_size_sweep.py'
    $taskSummaryArguments = @('-u', ('"' + $taskSummary + '"'), ('"' + $taskOutput + '"'))
    $taskPost = Start-Process -FilePath $taskPython -ArgumentList $taskSummaryArguments -WorkingDirectory $taskRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'summary.log') `
        -RedirectStandardError (Join-Path $taskOutput 'summary_error.log')
    $taskStatus.summary_pid = $taskPost.Id
    Save-TaskStatus
    $taskPost.WaitForExit()
    $taskStatus.summary_exit_code = $taskPost.ExitCode

    $taskRunState = Get-Content -LiteralPath (Join-Path $taskOutput 'run_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskStatus.runner_exit_code -eq 0 -and $taskRunState.state -eq 'completed' -and $taskStatus.summary_exit_code -eq 0) {
        $taskStatus.state = 'completed'
    } elseif ($taskStatus.runner_exit_code -eq 1 -and $taskRunState.state -eq 'completed_with_errors' -and $taskStatus.summary_exit_code -eq 0) {
        $taskStatus.state = 'completed_with_errors'
    } elseif ($taskStatus.runner_exit_code -eq 130) {
        $taskStatus.state = 'interrupted'
    } else {
        $taskStatus.state = 'failed'
    }
    Save-TaskStatus
} catch {
    $taskStatus.state = 'failed'
    $taskStatus.error = $_.Exception.Message
    Save-TaskStatus
    throw
} finally {
    $taskLock.Dispose()
}
