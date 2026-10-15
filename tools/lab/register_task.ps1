<#
.SYNOPSIS
  Register (or remove) the "Spireline fix lab" scheduled task: the fix lab's runner, started when the PC is idle.

.DESCRIPTION
  The task runs tools\lab\runner.py from the fork's main checkout with the venv's pythonw.exe (no console window),
  for the CURRENT user only: logon type Interactive, run level Limited, so it runs only while that user is logged
  on and no password is ever stored. Trigger: the PC goes idle. Settings: run only if idle for 10 minutes, stop when
  the idle ends, restart at the next idle, priority 7 (below normal), no wake-to-run, not on battery, one instance,
  12 hours at most. The runner also checks for user input itself (60 s) and exits 0, so both layers hand the PC back.

  Nothing is registered unless you run it without -WhatIf. -WhatIf prints the definition and changes nothing (it is a plain switch here, so no module noise).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1 -WhatIf
  powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1
  powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1 -Unregister
#>
param(
    [switch]$WhatIf,
    [switch]$Unregister,
    [string]$TaskName = 'Spireline fix lab',
    [string]$VenvScripts = 'D:\spireline\src\sts2\.venv\Scripts',
    [string]$Fork = 'D:\spireline\fork\sts2',
    [string]$Lab = 'D:\spireline\referee\lab',
    [string]$Tree = 'D:\spireline\fork\wt-lab',
    [string]$Game = 'D:\spireline\referee\game-lab',
    [int]$IdleMinutes = 10,
    [int]$ExecutionHours = 12
)
$ErrorActionPreference = 'Stop'

if ($Unregister) {
    if ($WhatIf) { Write-Output "What if: would unregister the scheduled task '$TaskName'. Nothing changed."; return }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Unregistered '$TaskName'."
    return
}

$python = Join-Path $VenvScripts 'pythonw.exe'
if (-not (Test-Path $python)) { $python = Join-Path $VenvScripts 'python.exe' }
$arguments = 'tools\lab\runner.py --lab "{0}" --tree "{1}" --game "{2}" --exit-with-parent' -f $Lab, $Tree, $Game
$user = '{0}\{1}' -f $env:USERDOMAIN, $env:USERNAME

$action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $Fork

$idleClass = Get-CimClass -ClassName MSFT_TaskIdleTrigger -Namespace Root/Microsoft/Windows/TaskScheduler
$trigger = New-CimInstance -CimClass $idleClass -ClientOnly
$trigger.Enabled = $true

$settings = New-ScheduledTaskSettingsSet `
    -RunOnlyIfIdle `
    -IdleDuration (New-TimeSpan -Minutes $IdleMinutes) `
    -IdleWaitTimeout (New-TimeSpan -Hours $ExecutionHours) `
    -RestartOnIdle `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionHours) `
    -Priority 7
$settings.DisallowStartIfOnBatteries = $true   # not on battery
# StopOnIdleEnd is the default (we do not pass -DontStopOnIdleEnd); no -WakeToRun, so the PC is never woken.

$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Write-Output "Task:       $TaskName"
Write-Output "Runs as:    the current user, Interactive, Limited (only while logged on; no password stored)"
Write-Output "Action:     $python $arguments"
Write-Output "Start in:   $Fork"
Write-Output "Trigger:    on idle"
$fmt = 'Settings:   RunOnlyIfIdle={0} IdleDuration={1} StopOnIdleEnd={2} RestartOnIdle={3} Priority={4} WakeToRun={5} DisallowStartIfOnBatteries={6} ExecutionTimeLimit={7} MultipleInstances={8}'
Write-Output ($fmt -f $settings.RunOnlyIfIdle, $settings.IdleSettings.IdleDuration, $settings.IdleSettings.StopOnIdleEnd,
    $settings.IdleSettings.RestartOnIdle, $settings.Priority, $settings.WakeToRun, $settings.DisallowStartIfOnBatteries,
    $settings.ExecutionTimeLimit, $settings.MultipleInstances)

if ($WhatIf) {
    Write-Output "What if: would register the scheduled task '$TaskName' as above. Nothing changed."
} else {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
        -Description 'Spireline fix lab: aimed real-game fights, shrinking and grouping of divergences while the PC is idle. See fork tools/lab/README.md.' | Out-Null
    Write-Output "Registered '$TaskName'. Stop it at any time: Stop-ScheduledTask -TaskName '$TaskName'; pause it: create <lab>\PAUSE."
}
