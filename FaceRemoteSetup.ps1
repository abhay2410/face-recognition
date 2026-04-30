<#
FaceRemoteSetup.ps1 (v2.0 - Combined & Persistent)
==================================================
- Sets SQL to 'pc_control = 0' for safety.
- Installs the background listener into C:\FaceSystem.
- Creates a hidden Scheduled Task to handle remote shutdown.
#>

# 1. Self-Elevate to Administrator
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs; exit
}

# 2. DATA DISCOVERY
$user = $env:USERNAME
$computer = $env:COMPUTERNAME
$adapter = Get-NetAdapter | Where-Object { $_.Status -eq "Up" -and $_.MediaType -eq "802.3" }
$mac = ($adapter.MacAddress).Replace("-",":")
$ip = (Get-NetIPAddress -InterfaceAlias $adapter.InterfaceAlias -AddressFamily IPv4).IPAddress

$results = @()
$results += "--- FACE SYSTEM REMOTE CLIENT SETUP LOG ---"
$results += "USER: $user | COMPUTER: $computer"
$results += "----------------------------------------------"
$results += "IP:  $ip"
$results += "MAC: $mac"
$results += "----------------------------------------------"
$results += "COPY AND PASTE THIS INTO SSMS:"
$results += "----------------------------------------------"

# THE MAGIC SQL QUERY (Defaulted to pc_control = 0 for safety)
$sqlQuery = "UPDATE employees SET pc_mac = '$mac', pc_ip = '$ip', pc_control = 0 WHERE name LIKE '%$user%';"
$results += $sqlQuery
$results += "----------------------------------------------"

# 3. Enable WoL
Write-Host "Configuring Wake on LAN..." -ForegroundColor Cyan
if ($adapter) { 
    Set-NetAdapterAdvancedProperty -Name $adapter.Name -DisplayName "Wake on Magic Packet" -DisplayValue "Enabled" -ErrorAction SilentlyContinue
}

# 4. Firewall Rule (UDP Port 9999)
Write-Host "Opening Firewall Port 9999..." -ForegroundColor Cyan
$ruleName = "FaceSystem_Remote_Control"
if (!(Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol UDP -LocalPort 9999 -Description "Allows server to trigger remote shutdown."
}

# 5. KILL OLD LISTENERS & WRITING SCRIPT
Write-Host "Cleaning up old listeners..." -ForegroundColor Cyan
Get-Process | Where-Object { $_.Path -like "*powershell*" } | ForEach-Object {
    $p = $_
    $netstat = netstat -ano | Select-String ":9999"
    if ($netstat) {
        $pidOnPort = ($netstat.ToString().Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries))[-1]
        if ($p.Id -eq $pidOnPort) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}

$listenerDir = "C:\FaceSystem"
$listenerFile = Join-Path $listenerDir "listener.ps1"
if (!(Test-Path $listenerDir)) { New-Item -Path $listenerDir -ItemType Directory | Out-Null }

$listenerCode = @'
$logFile = "C:\FaceSystem\listener_log.txt"
"--- Listener Started at $(Get-Date) ---" | Out-File $logFile
while ($true) {
    $udp = $null
    try {
        $udp = New-Object System.Net.Sockets.UdpClient(9999)
        $remoteEndpoint = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, 0)
        
        # Wait for signal
        $bytes = $udp.Receive([ref]$remoteEndpoint)
        $data = [System.Text.Encoding]::ASCII.GetString($bytes)
        "$(Get-Date): Received signal [$data] from $($remoteEndpoint.Address)" | Out-File $logFile -Append
        
        if ($data -eq "SHUTDOWN_NOW") {
            $udp.Close()
            "$(Get-Date): Executing SHUTDOWN" | Out-File $logFile -Append
            Stop-Computer -Force
        }
        elseif ($data -eq "LOCK_NOW") {
            "$(Get-Date): Executing LOCK" | Out-File $logFile -Append
            rundll32.exe user32.dll,LockWorkStation
        }
    } catch {
        "$(Get-Date): Error: $($_.Exception.Message)" | Out-File $logFile -Append
        Start-Sleep -Seconds 5
    } finally {
        if ($udp) { $udp.Close() }
    }
    Start-Sleep -Seconds 1
}
'@

Set-Content -Path $listenerFile -Value $listenerCode -Force
Write-Host "Listener script saved to $listenerFile" -ForegroundColor Green

# 6. REGISTER SCHEDULED TASK (SILENT BACKGROUND RUN)
Write-Host "Registering Background Task..." -ForegroundColor Cyan
$taskName = "FaceSystem_Remote_Listener"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$listenerFile`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -GroupId "Users" -RunLevel Highest

try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    
    # NEW: Start the task immediately so we don't have to reboot
    Start-ScheduledTask -TaskName $taskName
    
    Write-Host "✅ Persistent Task Created & STARTED." -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create task. You may need to run this as Admin manually." -ForegroundColor Red
}

# 7. SAVE LOG TO DESKTOP
$fileName = "$($user)_Face_Remote_PC_Setup.txt"
$logPath = Join-Path ([Environment]::GetFolderPath("Desktop")) $fileName
$results | Out-File -FilePath $logPath
Write-Host "`n✅ Success! Resulting SQL saved to: $logPath" -ForegroundColor Green

# Final Pause
Write-Host "`nPress [ENTER] to close and finish..." -ForegroundColor White
Read-Host
