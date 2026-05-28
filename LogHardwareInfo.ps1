# LogHardwareInfo.ps1 - Logs CPU, Memory, GPU information to a timestamped file
# Strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Determine the directory of this script and the log file path
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "hardware_log.txt"

# Helper function to append text to the log file with a newline
function Write-Log {
    param([string]$Message)
    $Message | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

# Timestamp header
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Log "--- Hardware Log: $timestamp ---"

try {
    # CPU Information
    $cpuInfo = Get-CimInstance -ClassName Win32_Processor | Select-Object -Property Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
    Write-Log "CPU Information:"
    foreach ($cpu in $cpuInfo) {
        Write-Log "  Name                : $($cpu.Name)"
        Write-Log "  Cores               : $($cpu.NumberOfCores)"
        Write-Log "  Logical Processors : $($cpu.NumberOfLogicalProcessors)"
        Write-Log "  Max Clock Speed (MHz): $($cpu.MaxClockSpeed)"
    }

    # Memory (Physical RAM) Information
    $memModules = Get-CimInstance -ClassName Win32_PhysicalMemory | Select-Object -Property Manufacturer,Capacity,Speed,SerialNumber
    $totalCapacity = ($memModules | Measure-Object -Property Capacity -Sum).Sum
    $totalGB = [math]::Round($totalCapacity / 1GB, 2)
    Write-Log "Memory Information:"
    Write-Log "  Total Physical RAM   : $totalGB GB"
    foreach ($module in $memModules) {
        $capGB = [math]::Round($module.Capacity / 1GB, 2)
        Write-Log "  Module: Manufacturer=$($module.Manufacturer), Capacity=${capGB}GB, Speed=$($module.Speed)MHz, Serial=$($module.SerialNumber)"
    }

    # Operating System Memory Stats (available/total)
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $totalPhys = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freePhys = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    Write-Log "OS Memory Stats:"
    Write-Log "  Total Visible Memory : $totalPhys MB"
    Write-Log "  Free Physical Memory : $freePhys MB"

    # GPU / Video Controller Information
    $gpuInfo = Get-CimInstance -ClassName Win32_VideoController | Select-Object -Property Name,DriverVersion,AdapterRAM,VideoProcessor
    Write-Log "GPU Information:"
    foreach ($gpu in $gpuInfo) {
        $ramGB = if ($gpu.AdapterRAM) { [math]::Round($gpu.AdapterRAM / 1GB, 2) } else { "N/A" }
        Write-Log "  Name          : $($gpu.Name)"
        Write-Log "  Processor     : $($gpu.VideoProcessor)"
        Write-Log "  Driver Version: $($gpu.DriverVersion)"
        Write-Log "  Adapter RAM   : $ramGB GB"
    }

    Write-Log "--- End of Log ---`n"
}
catch {
    Write-Log "Error occurred while collecting hardware information: $_"
    exit 1
}

exit 0
