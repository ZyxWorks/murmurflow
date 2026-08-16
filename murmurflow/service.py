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
import sysconfig
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


APP_NAME = "MurmurFlow"
BUNDLE_ID = "ai.murmurflow"


def app_path() -> Path:
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def build_app() -> Path | None:
    """Put the listener inside an .app bundle and return its executable, or None if that failed.

    THE PERMISSION ROW IS THE POINT. macOS names a Privacy row after the executable that asked, and
    for a Python tool that is the interpreter: the row reads ``python3.13``. Nobody looking for
    ``murmurflow`` finds it, and switching it on grants Accessibility and the microphone to EVERY
    other tool sharing that interpreter — which on a Mac with `uv` is all of them. A bundle gives
    this one program its own identity, so the row reads MurmurFlow and the grant stops there.

    The executable is a COPY of the interpreter rather than a script or a symlink, because both of
    those hand the identity straight back: a ``#!`` script runs as ``/bin/sh``, and macOS resolves
    a symlink before it looks. The copy keeps its ad-hoc signature, so the grant survives a
    reinstall; ``PYTHONHOME``/``PYTHONPATH`` in the job (below) are what let a 17 MB interpreter
    sitting outside its own prefix still find its standard library and this package.

    This interpreter is CPython from python-build-standalone as uv ships it, which links no
    ``libpython`` — checked with ``otool -L`` before relying on it. A build that did would need its
    dylib copied alongside, and this returns None rather than install something that cannot start.
    """
    real = Path(sys.executable).resolve()
    macos_dir = app_path() / "Contents" / "MacOS"
    executable = macos_dir / APP_NAME
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        # No Dock icon and no menu bar: this is a background listener, not an application anybody
        # switches to. Without it, dictation would put a bouncing icon in the Dock at every login.
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "MurmurFlow records while you hold the trigger key and transcribes on this Mac. "
            "Nothing is uploaded."
        ),
    }
    try:
        macos_dir.mkdir(parents=True, exist_ok=True)
        (app_path() / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
        # Replace rather than overwrite: writing into a running binary is refused (ETXTBSY), and
        # the reinstall path always has the old listener still up when this runs.
        staged = executable.with_suffix(".new")
        shutil.copy2(real, staged)
        staged.replace(executable)
    except OSError:
        return None
    return executable


def remove_app() -> bool:
    """Delete the bundle. True if there was one. Uninstalling must not leave an app behind."""
    if not app_path().is_dir():
        return False
    try:
        shutil.rmtree(app_path())
    except OSError:
        return False
    return True


def bundle_env() -> dict[str, str]:
    """The two variables the bundled interpreter cannot work out for itself.

    It is a copy living outside its own prefix, so the path-based search that finds the standard
    library in place finds nothing here. This is the whole price of the bundle.
    """
    return {
        "PYTHONHOME": str(Path(sys.executable).resolve().parent.parent),
        "PYTHONPATH": sysconfig.get_paths()["purelib"],
    }


def app_accessibility_trusted() -> bool | None:
    """Ask the BUNDLE whether IT has Accessibility. None when there is no bundle to ask.

    The CLI asking on its own behalf was the wrong question and gave a confidently wrong answer:
    the grant belongs to whichever binary asks for it, the daemon is the bundle, and the two are
    different identities on purpose. A diagnostic that reports the wrong process's permission is
    worse than one that says it does not know.
    """
    executable = app_path() / "Contents" / "MacOS" / APP_NAME
    if not executable.is_file():
        return None
    env = {**os.environ, **bundle_env()}
    code = "from murmurflow import hotkey; print(hotkey.accessibility_trusted())"
    try:
        proc = subprocess.run(
            [str(executable), "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = proc.stdout.strip()
    return answer == "True" if answer in {"True", "False"} else None


def _executable() -> list[str]:
    """The argv that starts the listener, resolved absolutely because launchd has no PATH.

    The .app bundle first, so the Privacy rows carry this tool's name. Then the installed
    ``murmurflow`` console script, then ``python -m murmurflow`` at this interpreter, which is what
    makes an editable/checkout install work without a second step.
    """
    bundled = build_app()
    if bundled is not None:
        return [str(bundled), "-m", "murmurflow", "listen"]
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
    argv = _executable()
    if argv[0].startswith(str(app_path())):
        env.update(bundle_env())
    job: dict[str, object] = {
        "Label": LABEL,
        "ProgramArguments": argv,
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
