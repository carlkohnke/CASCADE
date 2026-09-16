[CmdletBinding()]
param(
    [ValidateSet("Auto", "CPU", "GPU")]
    [string]$Mode = "Auto",

    [string]$InstallRoot,

    [string]$Python,

    [string]$DesktopShortcutDirectory,

    [string]$StartMenuShortcutDirectory,

    [string]$LogDirectory,

    [string]$PipCacheDirectory,

    [switch]$NoPathUpdate,

    [switch]$NoShortcuts,

    [switch]$NoSelfTest,

    [switch]$NonInteractive
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:TranscriptStarted = $false
$script:InstallSucceeded = $false
$script:InstallerLog = $null

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    Write-Host ("+ {0} {1}" -f $FilePath, ($ArgumentList -join " "))
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath"
    }
}

function Test-Python312 {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$PrefixArguments = @()
    )

    try {
        $probe = & $FilePath @PrefixArguments -c "import struct,sys; assert sys.version_info[:2] == (3,12); assert struct.calcsize('P') == 8; print('CASCADE_PYTHON_OK')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) {
            return [pscustomobject]@{
                FilePath = $FilePath
                PrefixArguments = [string[]]$PrefixArguments
                Executable = $FilePath
            }
        }
    } catch {
        return $null
    }
    return $null
}

function Resolve-Python312 {
    param([string]$RequestedPython)

    if ($RequestedPython) {
        $resolved = [System.IO.Path]::GetFullPath(
            (Resolve-Path -LiteralPath $RequestedPython).ProviderPath
        )
        $candidate = Test-Python312 -FilePath $resolved
        if ($null -eq $candidate) {
            throw "The selected Python is not 64-bit CPython 3.12: $resolved"
        }
        return $candidate
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = Test-Python312 -FilePath $launcher.Source -PrefixArguments @("-3.12")
        if ($null -ne $candidate) {
            return $candidate
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidate = Test-Python312 -FilePath $pythonCommand.Source
        if ($null -ne $candidate) {
            return $candidate
        }
    }

    throw @"
CASCADE requires 64-bit Python 3.12. Install it from https://www.python.org/downloads/windows/
and then double-click 'Install CASCADE for Windows.cmd' again. Python does not
need to be added to the global PATH when the standard Python launcher is installed.
"@
}

function Test-NvidiaGpu {
    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
        if ($controllers | Where-Object { $_.Name -match "NVIDIA" }) {
            return $true
        }
    } catch {
        Write-Warning "Windows GPU inventory was unavailable: $($_.Exception.Message)"
    }

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    return $null -ne $nvidiaSmi
}

function Add-UserPathEntry {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $resolvedDirectory = [System.IO.Path]::GetFullPath($Directory).TrimEnd("\")
    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    if ($current) {
        $entries = @($current.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries))
    }
    foreach ($entry in $entries) {
        if ($entry.Trim().TrimEnd("\").Equals($resolvedDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
    }

    $updatedEntries = @($entries) + @($resolvedDirectory)
    [Environment]::SetEnvironmentVariable("Path", ($updatedEntries -join ";"), "User")
    return $true
}

function New-CascadeShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $shortcutPath = Join-Path $Directory "CASCADE Studio.lnk"
    if (-not ("Cascade.WindowsShortcut" -as [type])) {
        Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace Cascade
{
    [ComImport]
    [Guid("00021401-0000-0000-C000-000000000046")]
    internal class ShellLink
    {
    }

    [ComImport]
    [Guid("000214F9-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int maximum, IntPtr findData, uint flags);
        void GetIDList(out IntPtr itemIdList);
        void SetIDList(IntPtr itemIdList);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder description, int maximum);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string description);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder directory, int maximum);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string directory);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder arguments, int maximum);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string arguments);
        void GetHotkey(out short hotkey);
        void SetHotkey(short hotkey);
        void GetShowCmd(out int showCommand);
        void SetShowCmd(int showCommand);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder iconPath, int maximum, out int iconIndex);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string iconPath, int iconIndex);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr window, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string path);
    }

    public static class WindowsShortcut
    {
        public static void Create(string shortcutPath, string targetPath, string workingDirectory)
        {
            IShellLinkW link = (IShellLinkW)new ShellLink();
            try
            {
                link.SetPath(targetPath);
                link.SetWorkingDirectory(workingDirectory);
                link.SetDescription("CASCADE Studio");
                link.SetIconLocation(targetPath, 0);
                ((IPersistFile)link).Save(shortcutPath, true);
            }
            finally
            {
                Marshal.FinalReleaseComObject(link);
            }
        }
    }
}
"@
    }
    [Cascade.WindowsShortcut]::Create($shortcutPath, $TargetPath, $WorkingDirectory)
    return $shortcutPath
}

function Show-CompletionMessage {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Message,
        [bool]$IsError = $false
    )

    if ($NonInteractive) {
        return
    }
    try {
        $shell = New-Object -ComObject WScript.Shell
        $icon = if ($IsError) { 16 } else { 64 }
        $null = $shell.Popup($Message, 0, $Title, $icon)
    } catch {
        Write-Warning "Could not display the completion dialog: $($_.Exception.Message)"
    }
}

try {
    if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
        throw "This installer must be run with native Windows PowerShell."
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "CASCADE requires 64-bit Windows."
    }

    $sourceRoot = [System.IO.Path]::GetFullPath(
        (Resolve-Path (Join-Path $PSScriptRoot "..\..")).ProviderPath
    )
    if ([System.IO.Path]::GetPathRoot($sourceRoot).StartsWith("\\")) {
        throw @"
The downloaded CASCADE source is on a network or WSL UNC path: $sourceRoot
Extract it to a normal Windows folder, such as Downloads\CASCADE, and run the installer there.
"@
    }
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "pyproject.toml") -PathType Leaf)) {
        throw "pyproject.toml was not found beside the installer source: $sourceRoot"
    }

    if (-not $InstallRoot) {
        $InstallRoot = Join-Path $env:LOCALAPPDATA "Programs\CASCADE"
    }
    $InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
    if (-not $LogDirectory) {
        $LogDirectory = Join-Path $env:LOCALAPPDATA "cascade\Logs"
    }
    $LogDirectory = [System.IO.Path]::GetFullPath($LogDirectory)
    if (-not $DesktopShortcutDirectory) {
        $DesktopShortcutDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    }
    if (-not $StartMenuShortcutDirectory) {
        $programs = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
        $StartMenuShortcutDirectory = Join-Path $programs "CASCADE"
    }

    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $script:InstallerLog = Join-Path $LogDirectory "cascade-install-$stamp.log"
    Start-Transcript -Path $script:InstallerLog -Force | Out-Null
    $script:TranscriptStarted = $true

    Write-Host "CASCADE native Windows installer" -ForegroundColor Green
    Write-Host "Source:  $sourceRoot"
    Write-Host "Install: $InstallRoot"
    Write-Host "Log:     $script:InstallerLog"

    Write-Stage "Locating 64-bit Python 3.12"
    $pythonCommand = Resolve-Python312 -RequestedPython $Python
    Write-Host "Python: $($pythonCommand.Executable)"

    $selectedMode = $Mode
    if ($selectedMode -eq "Auto") {
        if (Test-NvidiaGpu) {
            $selectedMode = "GPU"
        } else {
            $selectedMode = "CPU"
        }
    }
    Write-Host "Acceleration profile: $selectedMode"

    $runtimeDirectory = Join-Path $InstallRoot "runtime"
    $runtimePython = Join-Path $runtimeDirectory "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
        Write-Stage "Creating CASCADE's private Python environment"
        $venvArguments = @($pythonCommand.PrefixArguments) + @("-m", "venv", $runtimeDirectory)
        Invoke-Checked -FilePath $pythonCommand.FilePath -ArgumentList $venvArguments
    } else {
        $existing = Test-Python312 -FilePath $runtimePython
        if ($null -eq $existing) {
            throw "The existing CASCADE runtime is not 64-bit Python 3.12: $runtimePython"
        }
        Write-Host "Reusing the existing private Python environment."
    }

    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PYTHONUTF8 = "1"
    $cacheArguments = @("--no-cache-dir")
    if ($PipCacheDirectory) {
        $PipCacheDirectory = [System.IO.Path]::GetFullPath($PipCacheDirectory)
        New-Item -ItemType Directory -Path $PipCacheDirectory -Force | Out-Null
        $env:PIP_CACHE_DIR = $PipCacheDirectory
        $cacheArguments = @()
    }

    Write-Stage "Preparing the packaging tools"
    $packagingArguments = @("-m", "pip", "install", "--upgrade") +
        $cacheArguments + @("pip", "setuptools", "wheel")
    Invoke-Checked -FilePath $runtimePython -ArgumentList $packagingArguments

    Write-Stage "Building CASCADE from the downloaded source"
    $packageDirectory = Join-Path $InstallRoot "packages\$stamp"
    New-Item -ItemType Directory -Path $packageDirectory -Force | Out-Null
    $wheelArguments = @("-m", "pip", "wheel", "--no-deps") +
        $cacheArguments + @("--wheel-dir", $packageDirectory, $sourceRoot)
    Invoke-Checked -FilePath $runtimePython -ArgumentList $wheelArguments
    $wheel = Get-ChildItem -LiteralPath $packageDirectory -Filter "cascade_vascular-*.whl" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $wheel) {
        throw "The CASCADE wheel was not produced in $packageDirectory"
    }

    $extras = if ($selectedMode -eq "GPU") { "gui,gpu-cu13" } else { "gui" }
    $wheelRequirement = "$($wheel.FullName)[$extras]"
    Write-Stage "Installing CASCADE and binary dependencies"
    $installArguments = @("-m", "pip", "install", "--upgrade", "--only-binary=:all:") +
        $cacheArguments + @($wheelRequirement)
    Invoke-Checked -FilePath $runtimePython -ArgumentList $installArguments

    # pip treats a wheel with the same public version as already satisfied, even
    # when it was rebuilt from a newer release commit. Install dependencies from
    # the extras above, then make the selected wheel itself authoritative.
    $exactWheelArguments = @("-m", "pip", "install", "--force-reinstall", "--no-deps") +
        $cacheArguments + @($wheel.FullName)
    Invoke-Checked -FilePath $runtimePython -ArgumentList $exactWheelArguments

    $runtimeScripts = Join-Path $runtimeDirectory "Scripts"
    $cascadeCli = Join-Path $runtimeScripts "cascade.exe"
    $cascadeGui = Join-Path $runtimeScripts "cascade-gui.exe"
    if (-not (Test-Path -LiteralPath $cascadeCli -PathType Leaf)) {
        throw "CASCADE's CLI launcher was not installed: $cascadeCli"
    }
    if (-not (Test-Path -LiteralPath $cascadeGui -PathType Leaf)) {
        throw "CASCADE Studio's launcher was not installed: $cascadeGui"
    }

    Write-Stage "Verifying the installed package"
    Invoke-Checked -FilePath $cascadeCli -ArgumentList @("--version")
    $verifiedMode = $selectedMode
    if ($selectedMode -eq "GPU") {
        try {
            Invoke-Checked -FilePath $cascadeCli -ArgumentList @("doctor", "--require-gpu")
            if (-not $NoSelfTest) {
                Invoke-Checked -FilePath $cascadeCli -ArgumentList @("self-test", "--require-gpu")
            }
        } catch {
            if ($Mode -eq "GPU") {
                throw
            }
            Write-Warning "NVIDIA acceleration did not pass qualification; CASCADE will use CPU-compatible automatic paths."
            Write-Warning $_.Exception.Message
            $verifiedMode = "CPU fallback"
            Invoke-Checked -FilePath $cascadeCli -ArgumentList @("doctor", "--no-gpu-probe")
            if (-not $NoSelfTest) {
                Invoke-Checked -FilePath $cascadeCli -ArgumentList @("self-test")
            }
        }
    } else {
        Invoke-Checked -FilePath $cascadeCli -ArgumentList @("doctor", "--no-gpu-probe")
        if (-not $NoSelfTest) {
            Invoke-Checked -FilePath $cascadeCli -ArgumentList @("self-test")
        }
    }

    Write-Stage "Creating the stable CLI and Studio shortcut"
    $binDirectory = Join-Path $InstallRoot "bin"
    New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null
    $stableCli = Join-Path $binDirectory "cascade.exe"
    Copy-Item -LiteralPath $cascadeCli -Destination $stableCli -Force

    $pathAdded = $false
    if (-not $NoPathUpdate) {
        $pathAdded = Add-UserPathEntry -Directory $binDirectory
        $env:Path = "$binDirectory;$env:Path"
    }

    $shortcuts = @()
    if (-not $NoShortcuts) {
        $shortcuts += New-CascadeShortcut -Directory $DesktopShortcutDirectory -TargetPath $cascadeGui -WorkingDirectory $env:USERPROFILE
        if (-not $StartMenuShortcutDirectory.Equals($DesktopShortcutDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
            $shortcuts += New-CascadeShortcut -Directory $StartMenuShortcutDirectory -TargetPath $cascadeGui -WorkingDirectory $env:USERPROFILE
        }
    }

    $version = (& $runtimePython -c "import cascade; print(cascade.__version__)" | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) {
        throw "The installed CASCADE version could not be read."
    }
    $manifest = [ordered]@{
        schema_version = 1
        installed_at = (Get-Date).ToString("o")
        version = [string]$version
        source = $sourceRoot
        install_root = $InstallRoot
        runtime_python = $runtimePython
        cli = $stableCli
        gui = $cascadeGui
        requested_mode = $Mode
        selected_mode = $selectedMode
        verified_mode = $verifiedMode
        user_path_added = [bool]$pathAdded
        shortcuts = [string[]]$shortcuts
        log = $script:InstallerLog
    }
    $manifestPath = Join-Path $InstallRoot "install-manifest.json"
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    $script:InstallSucceeded = $true
    Write-Host ""
    Write-Host "CASCADE installation completed." -ForegroundColor Green
    Write-Host "Studio: $cascadeGui"
    Write-Host "CLI:    $stableCli"
    if (-not $NoPathUpdate) {
        Write-Host "Open a new PowerShell or Command Prompt and run: cascade --version"
    }
    if ($shortcuts.Count -gt 0) {
        Write-Host "Shortcut: $($shortcuts[0])"
    }
    Write-Host "Acceleration: $verifiedMode"
    Write-Host "Installer log: $script:InstallerLog"

    Show-CompletionMessage -Title "CASCADE installation complete" -Message @"
CASCADE Studio is ready.

Acceleration: $verifiedMode

Open CASCADE Studio from the new shortcut.
"@
} catch {
    Write-Host ""
    Write-Host "CASCADE installation failed." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($script:InstallerLog) {
        Write-Host "Installer log: $script:InstallerLog"
    }
    Show-CompletionMessage -Title "CASCADE installation failed" -Message $_.Exception.Message -IsError $true
    exit 1
} finally {
    if ($script:TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            # A transcript failure must not change an otherwise valid install result.
        }
    }
}

if (-not $script:InstallSucceeded) {
    exit 1
}
exit 0
