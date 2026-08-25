#!/bin/sh
# MurmurFlow — one-command install for a Mac with nothing on it.
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/ZyxWorks/murmurflow/main/install.sh)"
#
# That invocation, rather than `curl | sh`, is deliberate: `curl | sh` hands the pipe to the
# script's stdin, and both Homebrew's installer and macOS's `sudo` need the real terminal to ask
# you anything. Piped, they fail with nothing useful on screen.
#
# Everything below is idempotent — running it twice is a no-op plus an upgrade.

set -eu

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
die() { printf '\n\033[31m[!] %s\033[0m\n' "$1" >&2; exit 1; }

# Where the package comes from. Overridable so the installer itself can be tested against a local
# checkout before it is trusted against the internet.
SOURCE="${MURMURFLOW_SOURCE:-git+https://github.com/ZyxWorks/murmurflow}"

[ "$(uname -s)" = "Darwin" ] || die "MurmurFlow is macOS-only (this is $(uname -s))."

# --- 1. Homebrew, which is how the two binaries and the installer arrive -------------------------
if ! command -v brew >/dev/null 2>&1; then
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        [ -x "$candidate" ] && eval "$("$candidate" shellenv)" && break
    done
fi

if ! command -v brew >/dev/null 2>&1; then
    say "Installing Homebrew (macOS has no package manager of its own)"
    echo "It will ask for your Mac password. That is macOS, not us."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        [ -x "$candidate" ] && eval "$("$candidate" shellenv)" && break
    done
    command -v brew >/dev/null 2>&1 || die "Homebrew installed but is not on PATH. Open a new terminal and re-run this."
    # A fresh Homebrew puts itself on PATH for THIS shell only. Without this line the next terminal
    # cannot find brew, and every later instruction in the README looks broken.
    profile="$HOME/.zprofile"
    line="eval \"\$($(command -v brew) shellenv)\""
    grep -qF "$line" "$profile" 2>/dev/null || printf '\n%s\n' "$line" >>"$profile"
fi

# --- 2. The two binaries, and uv (which brings its own Python) ----------------------------------
# macOS ships Python 3.9; MurmurFlow needs 3.11+. uv downloads a private one, so nothing on the
# system Python is touched and nothing to break on the next macOS upgrade.
say "Installing whisper-cpp, ffmpeg and uv"
brew install --quiet whisper-cpp ffmpeg uv

# --- 3. MurmurFlow itself -----------------------------------------------------------------------
say "Installing MurmurFlow"
uv tool install --force --python 3.13 "$SOURCE"
uv tool update-shell || true  # puts ~/.local/bin on PATH for the NEXT terminal

MF="$HOME/.local/bin/murmurflow"
[ -x "$MF" ] || MF="$(command -v murmurflow || true)"
[ -n "$MF" ] && [ -x "$MF" ] || die "murmurflow installed but could not be found. Open a new terminal and run: murmurflow doctor"

# --- 4. The speech model, then the login agent --------------------------------------------------
say "Downloading the speech model (~1.6 GB, once)"
"$MF" setup

say "Turning dictation on"
"$MF" install

say "Done"
echo "Open a NEW terminal for the 'murmurflow' command; dictation itself is already live."
