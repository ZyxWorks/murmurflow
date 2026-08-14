"""The launchd user agent: dictation is live after every login, with nothing to remember.

One plist, one label, written to ``~/Library/LaunchAgents`` and (un)loaded with ``launchctl``. This
is macOS-only and does not pretend otherwise — the rest of the tool is macOS-only too, because
``avfoundation`` capture and the CGEvent key polling both are.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "ai.murmurflow.listen"

# launchd runs agents with a MINIMAL PATH that does not include /opt/homebrew/bin or /usr/local/bin,
# so an agent that shells out to ffmpeg and whisper-server silently fails with no useful error. Every
# rendered plist therefore declares its own PATH. This is the single most common way a working
# dictation setup breaks the moment it is installed rather than run by hand.
_AGENT_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def is_macos() -> bool:
    return sys.platform == "darwin"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _executable() -> list[str]:
    """The argv that starts the listener, resolved absolutely because launchd has no PATH.

    Prefers the installed ``murmurflow`` console script; falls back to ``python -m murmurflow`` at
    this interpreter, which is what makes an editable/checkout install work without a second step.
    """
    found = shutil.which("murmurflow")
    if found:
        return [found, "listen"]
    return [sys.executable, "-m", "murmurflow", "listen"]


def render_plist() -> bytes:
    """The launchd job. ``KeepAlive`` because a dictation daemon that quietly died is worse than one
    that never started — you find out by holding the key and getting nothing back.

    ``ProcessType: Interactive`` is LOAD-BEARING, not a nicety. Without it launchd applies its
    default throttle to this agent and to the ffmpeg it spawns, and a throttled process cannot drain
    a CoreAudio input buffer at real time. Measured on an M4 Pro: a 5.0s hold captured **1.27s** of
    audio under the installed agent and **4.05s** running the identical code from a terminal. The
    microphone opens fine either way (~300ms), the level is healthy either way — three quarters of
    every sentence is simply never recorded, reaches whisper as a short clip of clipped words, and
    comes back as confident nonsense. If you ever see "it works when I run it by hand and it is
    useless once installed", this line is why.

    ``ThrottleInterval`` so a machine that has not been granted microphone access backs off instead
    of hot-looping on the device.
    """
    env = {"PATH": _AGENT_PATH}
    home = os.environ.get("MURMURFLOW_HOME", "")
    if home:  # keep a relocated home across the login boundary, or the agent reads the wrong one
        env["MURMURFLOW_HOME"] = home
    job: dict[str, object] = {
        "Label": LABEL,
        "ProgramArguments": _executable(),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "EnvironmentVariables": env,
        "StandardOutPath": str(Path.home() / ".murmurflow" / "listen.log"),
        "StandardErrorPath": str(Path.home() / ".murmurflow" / "listen.log"),
    }
    return plistlib.dumps(job)


def _launchctl(*args: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["launchctl", *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stderr or proc.stdout or "").strip()


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install() -> tuple[bool, str]:
    """Write the plist and load it. Idempotent: an already-loaded agent is booted out first.

    The bootout is unconditional and its failure is ignored on purpose — "was not loaded" is the
    expected answer on a first install, and treating it as an error would make the happy path look
    broken.
    """
    if not is_macos():
        return False, "launchd is macOS-only"
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    (Path.home() / ".murmurflow").mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_plist())
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # ignore: not-loaded is the normal first case
    ok, detail = _launchctl("bootstrap", _domain(), str(path))
    if not ok:
        # Older macOS, and the fallback that still works everywhere.
        ok, detail = _launchctl("load", "-w", str(path))
    return ok, detail


def uninstall() -> tuple[bool, str]:
    """Unload the agent and delete its plist. Succeeds when the end state is "not installed"."""
    if not is_macos():
        return False, "launchd is macOS-only"
    path = plist_path()
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    _launchctl("unload", str(path))
    path.unlink(missing_ok=True)
    return True, ""


def running() -> bool:
    """True if launchd currently has the agent loaded."""
    if not is_macos():
        return False
    ok, _ = _launchctl("print", f"{_domain()}/{LABEL}")
    return ok
