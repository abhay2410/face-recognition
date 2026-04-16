<#
.SYNOPSIS
    Face Access System — Production Service Uninstaller
#>

$ServiceName = "FaceAccessSystem"

# Check for Administrator privileges
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Please run this script as Administrator!" -ForegroundColor Red
    exit
}

Write-Host "--- Face Access System Release Uninstaller ---" -ForegroundColor Cyan

# 1. Stop current instance
Write-Host "[1/2] Stopping task..."
Stop-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue

# 2. Unregister from system
Write-Host "[2/2] Removing task from Windows..."
Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false -ErrorAction SilentlyContinue

# 3. Terminate any stray processes
Write-Host "[3/3] Ensuring process is closed..."
Get-Process FaceAccessSystem -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "`nUninstallation Complete!" -ForegroundColor Green
