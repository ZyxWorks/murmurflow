# MurmurFlow - one-command install for a Windows PC with nothing on it.
#
#   irm https://raw.githubusercontent.com/ZyxWorks/murmurflow/main/install.ps1 | iex
#
# The Windows twin of install.sh, and it does the same four things: get the two binaries, get the
# package, download a model, register the listener at logon. It differs in exactly one place, and
# that place is whisper: macOS has `brew install whisper-cpp` and Windows has no package for it at
# all, so the official whisper.cpp release build is downloaded and unzipped instead.
#
# Nothing here needs Administrator. Everything lands under the user's own profile, PATH is set with
# `setx` at user scope, and the listener runs as an ordinary logon task - a client who cannot get
# admin rights on their work laptop can still install this, which on a German company machine is
# the normal case and not an edge one.

$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "!!  $msg" -ForegroundColor Red; exit 1 }

# `$ErrorActionPreference = "Stop"` governs PowerShell's OWN errors and says nothing about the exit
# code of a native .exe. Without checking it, a network drop halfway through the 1.6 GB model
# download still ends in a cheerful "Done." - install.sh gets this right, because `set -eu` aborts
# on an external command by itself.
function Check($what) { if ($LASTEXITCODE -ne 0) { Die "$what failed (exit $LASTEXITCODE)." } }

$Source = if ($env:MURMURFLOW_SOURCE) { $env:MURMURFLOW_SOURCE } else { "git+https://github.com/ZyxWorks/murmurflow" }
$BinDir = Join-Path $env:LOCALAPPDATA "MurmurFlow\bin"

# --- 1. winget, which is how ffmpeg and uv arrive -----------------------------------------------
# Shipped with Windows 10 1809+ and Windows 11. Older than that is a manual install, and saying so
# beats failing four steps later with a message about a missing command.
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Die "winget not found. Install 'App Installer' from the Microsoft Store, then re-run this."
}

Say "Installing ffmpeg and uv"
winget install --id Gyan.FFmpeg --source winget --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Null
winget install --id astral-sh.uv --source winget --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Null

# winget puts new commands on the PATH of the NEXT shell only. Without this, every later step in
# this same script cannot find what the step above it just installed.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

# These two are checked by OUTCOME rather than by exit code, deliberately: winget returns a
# non-zero code for "already installed", so a re-run of this script would abort on a machine where
# nothing is wrong. What matters is whether the command is there afterwards.
foreach ($tool in "ffmpeg", "uv") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Die "$tool is still missing after winget. Install it by hand, then re-run this."
    }
}

# --- 2. whisper.cpp, which has no Windows package ------------------------------------------------
# The plain x64 build, deliberately: `whisper-blas-bin` and the two `cublas` builds are faster on
# the machines that have the hardware for them and simply do not start on the machines that do not.
# A dictation tool that fails to launch on a client's laptop is worse than one that transcribes a
# second slower, and the model choice is a far bigger lever than the BLAS build anyway.
if (-not (Get-Command whisper-server -ErrorAction SilentlyContinue)) {
    Say "Installing whisper.cpp (no package exists, so this is its own release build)"
    $zip = Join-Path $env:TEMP "whisper-bin-x64.zip"
    $release = "https://github.com/ggml-org/whisper.cpp/releases/latest/download/whisper-bin-x64.zip"
    Invoke-WebRequest -Uri $release -OutFile $zip -UseBasicParsing
    $staging = Join-Path $env:TEMP "murmurflow-whisper"
    Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip -DestinationPath $staging -Force
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    # The zip nests everything under Release\, and the DLLs beside the exes are not optional -
    # whisper-server.exe will not start without ggml.dll and the ggml-cpu-*.dll next to it.
    Copy-Item -Path (Join-Path $staging "Release\*") -Destination $BinDir -Recurse -Force
    Remove-Item $zip, $staging -Recurse -Force -ErrorAction SilentlyContinue

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$BinDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDir", "User")
    }
    $env:Path = "$env:Path;$BinDir"
}

# --- 3. MurmurFlow itself ------------------------------------------------------------------------
Say "Installing murmurflow"
uv tool install --force --python 3.13 $Source
Check "uv tool install"
uv tool update-shell 2>&1 | Out-Null
# `murmurflow install` re-installs the package from its source before it does anything else, so
# that a pull plus `murmurflow install` is the whole update. We have just done that, so tell it
# not to do it twice.
$env:MURMURFLOW_RESYNCED = "1"

$mf = (Get-Command murmurflow -ErrorAction SilentlyContinue)
if (-not $mf) { $mf = Join-Path $env:USERPROFILE ".local\bin\murmurflow.exe" } else { $mf = $mf.Source }
if (-not (Test-Path $mf)) { Die "murmurflow installed but could not be found. Open a new terminal and run: murmurflow doctor" }

# --- 4. The model, then the listener --------------------------------------------------------------
Say "Downloading the transcription models (~2.1 GB, once)"
& $mf setup
Check "murmurflow setup (the model download)"

Say "Registering the listener to start at logon"
& $mf install
Check "murmurflow install"

Write-Host ""
Say "Done. Open a NEW terminal and run: murmurflow doctor"
Write-Host "    Then double-tap left Control, say something, and tap once to stop."
