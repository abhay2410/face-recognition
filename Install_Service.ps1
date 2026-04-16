<#
.SYNOPSIS
    Face Access System — Production Service Installer (Windows 11)
    Registers the .exe to run automatically in the background on startup.
#>

$ExePath = Join-Path $PSScriptRoot "FaceAccessSystem.exe"
$ServiceName = "FaceAccessSystem"
$DisplayName = "Face Access System Backend"

if (-not (Test-Path $ExePath)) {
    Write-Host "CRITICAL: FaceAccessSystem.exe not found in $PSScriptRoot" -ForegroundColor Red
    exit
}

# Check for Administrator privileges
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Please run this script as Administrator!" -ForegroundColor Red
    exit
}

Write-Host "--- Face Access System Release Installer ---" -ForegroundColor Cyan

# 1. Register as a Scheduled Task (Most reliable for Python web servers)
# Runs as SYSTEM, starts on boot, hidden from user.
Write-Host "[1/2] Creating Scheduled Task..."
$Action = New-ScheduledTaskAction -Execute $ExePath -WorkingDirectory $PSScriptRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 365)
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

try {
    Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $ServiceName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force
    Write-Host "SUCCESS: Task created. It will start automatically on next reboot." -ForegroundColor Green
} catch {
    Write-Host "FAILED: Could not create scheduled task ($($_.Exception.Message))" -ForegroundColor Red
}

# 2. Start it right now
Write-Host "[2/2] Starting application in background..."
Start-ScheduledTask -TaskName $ServiceName

Write-Host "`nInstallation Complete!" -ForegroundColor Cyan
Write-Host "Dashboard: http://localhost:8000"
Write-Host "Logs:      $PSScriptRoot/fastapi_access.log"
