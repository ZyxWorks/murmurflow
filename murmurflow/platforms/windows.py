"""The Windows backend: dshow, ``GetAsyncKeyState``, ``keybd_event``, Task Scheduler.

Written because **the client desktop in Germany is Windows**, and dictation is the first thing a
client touches. A macOS-only dictation tool caps the practice at macOS-only clients, which is a
sales constraint rather than a nice-to-have.

The same four seams as macOS, and three of them are smaller here:

* **capture** — ffmpeg ships a ``dshow`` input on Windows exactly as it ships ``avfoundation`` on
  macOS. The one real difference is that dshow addresses a microphone **by name** and avfoundation
  by index, which is why :func:`capture_args` takes an opaque id from :func:`list_inputs` and
  nothing above the seam ever reads it.
* **keys** — ``GetAsyncKeyState`` is one call per key with no framework to find, and it is BETTER
  than the Mac: Windows reports left and right modifiers apart, so ``right_command`` is a trigger
  that can actually fire here. (On the MacBooks the right-side virtual keycodes never go true.)
* **inject** — ``keybd_event`` for ^V, clipboard through ``user32``/``kernel32``.
* **service** — Task Scheduler at logon, from an XML definition so the task can restart itself.

**There is no TCC, so there is no permission to grant and no ``.app`` bundle to build.** That is a
whole class of macOS failure that does not exist here: no Accessibility row, no Microphone row, no
identity to defend, and no grant to lose to a reinstall. ``murmurflow doctor`` still prints a row
for it — see :func:`doctor_rows` — because a row that vanishes reads as a check that was forgotten.

**Honest ceilings, all three from Windows itself and none of them worked around:**

* UIPI. A process at normal integrity cannot send input to, or read keys pressed inside, a window
  running elevated. Dictating into an admin console needs MurmurFlow running elevated too.
* The clipboard restore keeps **text only**. macOS can hold every pasteboard flavour in one
  AppleScript value; Windows cannot round-trip arbitrary clipboard handles safely, so an image on
  the clipboard is lost when a dictation lands. Named here rather than discovered.
* ``fn`` is not a trigger. On most Windows laptops Fn is handled in the keyboard firmware and never
  reaches the OS at all, so it cannot be polled — it is simply absent from :func:`trigger_names`.

ponytail: ``keybd_event`` rather than ``SendInput``. It is four calls against ~40 lines of INPUT
struct-and-union ctypes, and it has been "superseded" since 2001 without ever being removed. The
ceiling is a target that filters injected input by ``LLKHF_INJECTED`` source; move to ``SendInput``
if one ever turns up.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import config

NAME = "Windows"

# Loaded at import and only where they exist, so this module can be IMPORTED on a Mac — which is
# what lets the seam contract be tested from the machine the port is written on. `type: ignore` is
# the honest annotation: `ctypes.WinDLL` genuinely is not there when mypy is checking for darwin,
# and pinning mypy to win32 would hide every macOS-side error instead.
_windows = sys.platform.startswith("win")
_user32 = ctypes.WinDLL("user32", use_last_error=True) if _windows else None  # type: ignore[attr-defined]
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if _windows else None  # type: ignore[attr-defined]

#: Calls that hand back a POINTER or a HANDLE. ctypes cannot guess a return type and its guess is a
#: 32-bit ``int``, so on 64-bit Windows — every Windows this ships to — the top half of the address
#: is cut off before Python ever sees it. The truncated value then goes straight into
#: ``ctypes.memmove`` in :func:`clipboard_set`: a crash on the first paste, and a clipboard restore
#: that fails without saying so. macos.py declares a restype for every pointer-returning call it
#: makes; this file declared none, which is why the whole paste path on Windows was broken.
_POINTER_RETURNS = {
    "kernel32": ("GlobalAlloc", "GlobalLock", "GlobalFree"),
    "user32": ("GetClipboardData", "SetClipboardData"),
}


def _declare_signatures(user32: Any, kernel32: Any) -> None:
    """Give ctypes the return types it cannot guess. Called once, at import, on Windows only."""
    for lib, key in ((user32, "user32"), (kernel32, "kernel32")):
        for name in _POINTER_RETURNS[key]:
            getattr(lib, name).restype = ctypes.c_void_p
    # A SHORT, not an int. Only bit 0x8000 is ever read, so this one was never wrong in practice —
    # it is declared so the file has no undeclared return left to be wrong about later.
    user32.GetAsyncKeyState.restype = ctypes.c_short


if _user32 is not None and _kernel32 is not None:
    _declare_signatures(_user32, _kernel32)

# --- capture --------------------------------------------------------------------------------


def capture_available() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not found — winget install Gyan.FFmpeg"
    return True, ""


def capture_args(device: str) -> list[str]:
    """ffmpeg's input flags for ``device``, a dshow device NAME.

    dshow has no ``:default`` spelling — there is no token meaning "whatever the system picked", so
    an empty device is resolved to the first audio input by :func:`default_input` before it gets
    here. If that found nothing there is no microphone to record from and the caller says so.
    """
    if not device:
        return []
    return ["-f", "dshow", "-i", f"audio={device}"]


# ffmpeg prints dshow devices as `"Name" (audio)`, one per line, on stderr. Older builds omit the
# `(audio)` suffix and separate the two kinds with a `DirectShow audio devices` header instead, so
# both shapes are read — a device list that silently comes back empty is indistinguishable from a
# machine with no microphone, and they need completely different answers.
_DSHOW_LINE = re.compile(r'"([^"]+)"\s*(\(audio\))?')


def list_inputs() -> list[tuple[str, str]]:
    """Every dshow AUDIO input as ``(name, name)``; ``[]`` if ffmpeg is missing or errors.

    The id and the label are the same string because dshow is addressed by name. ffmpeg exits
    non-zero listing devices (there is no input file), so that is not treated as failure.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[tuple[str, str]] = []
    in_audio = False
    for line in (proc.stderr or "").splitlines():
        lowered = line.lower()
        if "video devices" in lowered:
            in_audio = False
            continue
        if "audio devices" in lowered:
            in_audio = True
            continue
        if 'alternative name "' in lowered:
            # The `@device_cm_{GUID}` twin of the line above it. Stable across renames, and
            # unreadable — a device picker showing GUIDs is a device picker nobody can use.
            continue
        match = _DSHOW_LINE.search(line)
        if not match:
            continue
        if match.group(2) or in_audio:
            name = match.group(1)
            if name and (name, name) not in devices:
                devices.append((name, name))
    return devices


def default_input() -> str:
    """The device id to record from when the config names none: the first audio input there is."""
    inputs = list_inputs()
    return inputs[0][0] if inputs else ""


# --- keys -----------------------------------------------------------------------------------

# Virtual-Key codes (winuser.h). Unlike the Mac, the side-specific ones are real: Windows keeps
# left and right modifier state apart and reports it honestly, so `right_command` is a usable
# trigger here and is not on macOS.
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F13 = 0x7C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4  # left Alt
VK_RMENU = 0xA5  # right Alt = AltGr on a German layout

#: The canonical trigger vocabulary, unchanged from macOS (see :func:`hotkey.canonical_trigger`):
#: ``option`` IS Alt and ``command`` IS the Windows key, which is what `alt`/`win`/`super`/`meta`
#: already alias to. Keeping one vocabulary across both platforms is what makes a config file, a
#: doctor row and a README sentence mean the same thing on a client's machine as on ours.
#:
#: A name maps to the codes that must ALL be held. A single-element tuple is one key; a
#: two-element one is a combo; ``command`` is two codes meaning EITHER, which is the one case that
#: needs its own table below.
KEY_TRIGGERS: dict[str, tuple[int, ...]] = {
    "left_control": (VK_LCONTROL,),
    "right_control": (VK_RCONTROL,),
    "control": (VK_CONTROL,),
    "left_shift": (VK_LSHIFT,),
    "right_shift": (VK_RSHIFT,),
    "shift": (VK_SHIFT,),
    "left_option": (VK_LMENU,),
    "right_option": (VK_RMENU,),
    "option": (VK_MENU,),
    "left_command": (VK_LWIN,),
    "right_command": (VK_RWIN,),
    "f13": (VK_F13,),
    "control_option": (VK_CONTROL, VK_MENU),
    "control_command": (VK_CONTROL, VK_LWIN),
    "command_option": (VK_LWIN, VK_MENU),
    "control_shift": (VK_CONTROL, VK_SHIFT),
}

#: Names where ANY of the codes counts, rather than all of them. Only the bare Windows key: it has
#: no combined virtual code the way Control, Shift and Alt do.
_EITHER_TRIGGERS: dict[str, tuple[int, ...]] = {"command": (VK_LWIN, VK_RWIN)}


def trigger_names() -> frozenset[str]:
    """No ``fn``: on most Windows laptops it never reaches the OS, so it cannot be polled."""
    return frozenset(KEY_TRIGGERS) | frozenset(_EITHER_TRIGGERS)


def keys_unavailable() -> str:
    if _user32 is None:
        return "user32 unavailable — is this Windows?"
    return ""


def _held(code: int) -> bool:
    # Bit 15 of the return is "down right now". Bit 0 is "pressed since the last call", which is
    # deliberately NOT used for the trigger: the poll loop asks 60 times a second and would consume
    # that flag before anything else could read it.
    if _user32 is None:
        return False
    return bool(_user32.GetAsyncKeyState(ctypes.c_int(code)) & 0x8000)


def is_down(name: str) -> bool:
    either = _EITHER_TRIGGERS.get(name)
    if either is not None:
        return any(_held(code) for code in either)
    codes = KEY_TRIGGERS.get(name)
    if not codes:
        return False
    return all(_held(code) for code in codes)


# The keys a person actually chords WITH, plus the mouse buttons. Scanned at poll rate to answer
# "did the user do something else while holding the trigger", which is the chord guard's whole
# input.
#
# `GetLastInputInfo` is the obvious call and it is the WRONG one: it counts mouse MOVEMENT and the
# modifier press itself, so it resets continuously while the trigger is held and every recording
# would abort. macOS answers the right question directly
# (`CGEventSourceSecondsSinceLastEventType` on key-down and clicks, which modifier changes do not
# raise); Windows has no equivalent, so the equivalent is reconstructed from a scan.
#
# ponytail: ~80 GetAsyncKeyState calls per 60 Hz tick, which is under 5k cheap calls a second and
# invisible. The ceiling is that a key outside this set is not seen as a chord; the fix is to widen
# the range, not to add a hook — a low-level keyboard hook needs a message pump and a thread, and
# that is a service to keep alive rather than a function to call.
_CHORD_KEYS: tuple[int, ...] = (
    VK_LBUTTON,
    VK_RBUTTON,
    VK_MBUTTON,
    VK_BACK,
    VK_TAB,
    VK_RETURN,
    VK_ESCAPE,
    VK_SPACE,
    VK_DELETE,
    *range(0x21, 0x2E),  # PageUp..Insert, the arrows and Home/End live in here
    *range(0x30, 0x3A),  # 0-9
    *range(0x41, 0x5B),  # A-Z
    *range(0x60, 0x70),  # numpad
    *range(0x70, 0x7C),  # F1-F12
    *range(0xBA, 0xC1),  # OEM punctuation
    *range(0xDB, 0xE0),  # brackets, backslash, quote
)

_last_other_input = 0.0


def seconds_since_input() -> float:
    """Seconds since any non-modifier key or mouse button was last down.

    Stateful on purpose: there is no Windows call that answers this, so the answer is accumulated
    by the same 60 Hz poll that reads the trigger. The first call has nothing to remember and says
    "forever ago", which fails OPEN — a chord guard that fires on its own first tick would abort
    the very first dictation after every restart.
    """
    global _last_other_input
    if any(_held(code) for code in _CHORD_KEYS):
        _last_other_input = time.monotonic()
    if _last_other_input == 0.0:
        return float("inf")
    return time.monotonic() - _last_other_input


# --- inject ---------------------------------------------------------------------------------

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_KEYEVENTF_KEYUP = 0x0002
_VK_V = 0x56


def input_blocked() -> str:
    """Windows has no Secure Input mode. The one thing that swallows injected keys is UIPI, and it
    cannot be detected without knowing the foreground window's integrity level — which is a
    diagnostic worth having only once somebody reports it, not a guess to make on every paste.
    """
    return ""


def _clipboard_text() -> str | None:
    """Whatever text is on the clipboard, or ``None`` for "nothing to put back"."""
    if _user32 is None or _kernel32 is None:
        return None
    if not _user32.OpenClipboard(None):
        return None
    try:
        handle = _user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return None
        pointer = _kernel32.GlobalLock(ctypes.c_void_p(handle))
        if not pointer:
            return None
        try:
            return ctypes.c_wchar_p(pointer).value
        finally:
            _kernel32.GlobalUnlock(ctypes.c_void_p(handle))
    finally:
        _user32.CloseClipboard()


def clipboard_set(text: str) -> bool:
    """Put ``text`` on the clipboard as CF_UNICODETEXT. ``True`` on success.

    The buffer is ``GlobalAlloc(GMEM_MOVEABLE)`` and is deliberately NOT freed: ``SetClipboardData``
    takes ownership of the handle, and freeing it afterwards is a use-after-free the clipboard
    viewer pays for, not us.
    """
    if _user32 is None or _kernel32 is None:
        return False
    payload = text + "\0"
    size = len(payload) * ctypes.sizeof(ctypes.c_wchar)
    if not _user32.OpenClipboard(None):
        return False
    try:
        _user32.EmptyClipboard()
        handle = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, ctypes.c_size_t(size))
        if not handle:
            return False
        pointer = _kernel32.GlobalLock(ctypes.c_void_p(handle))
        if not pointer:
            _kernel32.GlobalFree(ctypes.c_void_p(handle))
            return False
        ctypes.memmove(pointer, ctypes.create_unicode_buffer(payload), size)
        _kernel32.GlobalUnlock(ctypes.c_void_p(handle))
        if not _user32.SetClipboardData(_CF_UNICODETEXT, ctypes.c_void_p(handle)):
            _kernel32.GlobalFree(ctypes.c_void_p(handle))
            return False
        return True
    finally:
        _user32.CloseClipboard()


def inject(text: str, settle: float) -> tuple[bool, str, str]:
    """Clipboard-swap + synthetic ^V, then put the previous TEXT back.

    ``(ok, problem, note)``. The note is the same diagnostic clause the Mac writes into the daemon
    log — how much of the transcript survived the copy, and whether it was still there when the
    restore came round. There is no cheap answer here for WHICH app got the keystroke (that needs
    ``GetForegroundWindow`` + a PID walk), so this reports the two lengths and no target: half a
    diagnostic that is true beats a name that would have to be guessed at.

    Paste rather than typing the string a character at a time, for the same two reasons as macOS:
    it is O(1) instead of O(chars), and it is immune to the keyboard-layout mangling that synthetic
    per-character events hit — worse here than there, because ``keybd_event`` carries a virtual-key
    code that the target re-maps through ITS OWN layout, so an umlaut typed key-by-key on a German
    layout arrives as whatever sits at that position on the reader's.

    Only text is restored. See the module docstring: an image on the clipboard is lost.
    """
    if _user32 is None:
        return False, "user32 unavailable — is this Windows?", ""
    saved = _clipboard_text()
    if not clipboard_set(text):
        return False, "could not put the text on the clipboard", ""
    held = len(_clipboard_text() or "")
    try:
        _user32.keybd_event(VK_CONTROL, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, _KEYEVENTF_KEYUP, 0)
        _user32.keybd_event(VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)
    except OSError as exc:
        return False, f"paste failed: {exc}. The text is on your clipboard — press Ctrl-V.", ""
    # The same race the Mac has, and the same answer: the restore is what ENDS the window in which
    # the target may read the clipboard, so it waits for the text rather than a fixed tick.
    time.sleep(settle)
    still = len(_clipboard_text() or "")
    if saved is not None:
        clipboard_set(saved)
    note = ""
    if held != len(text):
        note = f"COPIED ONLY {held}/{len(text)}"
    elif still != held:
        note = f"CLIPBOARD CHANGED UNDER THE PASTE ({still}/{len(text)})"
    return True, "", note


def input_permitted() -> bool:
    """There is no permission to grant. ``True``, because "not required" and "granted" are the same
    answer to a caller deciding whether to warn somebody.
    """
    return True


def permission_hint() -> str:
    return ""


# --- typing ---------------------------------------------------------------------------------


def type_text(text: str) -> str:
    """The whole string back, so Windows keeps the clipboard paste it has always used.

    The macOS version exists because the clipboard round trip is half the streaming cycle THERE —
    an AppleScript that saves the pasteboard, sends Cmd-V, waits and restores. `SendInput` with
    `KEYEVENTF_UNICODE` is the equivalent and would be worth having, but it is a second ctypes
    surface to get right on a platform nobody here can test on, and the paste already works.
    """
    return text


# --- the one sound ------------------------------------------------------------------------------


def _beep(which: str) -> None:
    """One system sound, without blocking. ``winsound`` is stdlib here.

    ``MessageBeep`` and not a wav: Windows names its sounds by EVENT rather than by path, so there
    is no file to point at, and the alternative is shipping and caching one of our own for a tick.
    """
    with contextlib.suppress(Exception):
        winsound = importlib.import_module("winsound")
        winsound.MessageBeep(getattr(winsound, which))


def play_ready() -> None:
    """The microphone is live: start talking."""
    _beep("MB_OK")


def play_done() -> None:
    """The microphone just closed: stop talking."""
    _beep("MB_ICONASTERISK")


# --- service --------------------------------------------------------------------------------

TASK_NAME = "MurmurFlow"


def service_path() -> Path:
    """Task Scheduler keeps its own store, so the on-disk artefact is the launcher we write."""
    return config.home_root() / "listen.cmd"


def _log_path() -> Path:
    return config.log_path()


def _pythonw() -> str:
    """``pythonw.exe`` beside this interpreter if there is one, else this interpreter.

    ``pythonw`` has no console, which is the difference between a background dictation daemon and a
    black window that sits in the taskbar for the rest of the session.
    """
    executable = Path(sys.executable)
    windowless = executable.with_name(executable.name.replace("python", "pythonw", 1))
    return str(windowless) if windowless.is_file() else str(executable)


def _launcher(args: list[str], env: dict[str, str]) -> Path:
    """Write the .cmd the task runs, and return it.

    A launcher rather than putting the command straight in the task: ``schtasks`` truncates a long
    action, quoting a path with spaces through it is its own genre of bug, and neither the log
    redirect nor the environment fits in there at all. A file is also the thing somebody can open
    and read when it does not work.
    """
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["@echo off"]
    lines += [f'set "{key}={value}"' for key, value in env.items()]
    argv = " ".join(f'"{part}"' for part in [_pythonw(), "-m", "murmurflow", *args])
    lines.append(f'start "" /b {argv} >> "{_log_path()}" 2>&1')
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return path


# `KeepAlive` is launchd's word for it, and this is the Windows one. Plain `schtasks /create` flags
# cannot express "restart it if it dies" — only the XML form can — and a dictation daemon that
# quietly died is worse than one that never started, because you find out by holding the key and
# getting nothing back.
_TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>MurmurFlow press-to-talk dictation listener.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId></LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>{command}</Command></Exec>
  </Actions>
</Task>
"""


def task_xml(command: str, user: str) -> str:
    """The task definition. Split out so it can be asserted on without a Windows machine."""
    return _TASK_XML.format(command=command, user=user)


def _schtasks(*args: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["schtasks", *args], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stderr or proc.stdout or "").strip()


def service_install(args: list[str], env: dict[str, str]) -> tuple[bool, str]:
    """Register the logon task. Idempotent — ``/f`` overwrites an existing one."""
    _log_path().parent.mkdir(parents=True, exist_ok=True)
    launcher = _launcher(args, env)
    user = os.environ.get("USERNAME", "")
    domain = os.environ.get("USERDOMAIN", "")
    xml_path = launcher.with_name("listen.task.xml")
    # UTF-16 with a BOM: schtasks rejects a task XML in any other encoding, with an error that
    # names the file and not the reason.
    xml_path.write_text(
        task_xml(str(launcher), f"{domain}\\{user}" if domain else user), encoding="utf-16"
    )
    ok, detail = _schtasks("/create", "/tn", TASK_NAME, "/xml", str(xml_path), "/f")
    if not ok:
        return False, detail
    return _schtasks("/run", "/tn", TASK_NAME)


def service_uninstall() -> tuple[bool, str]:
    """Delete the task. Succeeds when the end state is "not installed"."""
    _schtasks("/end", "/tn", TASK_NAME)
    _schtasks("/delete", "/tn", TASK_NAME, "/f")
    with contextlib.suppress(OSError):
        service_path().unlink(missing_ok=True)
        service_path().with_name("listen.task.xml").unlink(missing_ok=True)
    return True, ""


def service_restart() -> bool:
    if not service_running():
        return False
    _schtasks("/end", "/tn", TASK_NAME)
    ok, _ = _schtasks("/run", "/tn", TASK_NAME)
    return ok


def service_running() -> bool:
    ok, detail = _schtasks("/query", "/tn", TASK_NAME)
    return ok and "running" in detail.lower()


def dictation_conflict(trigger: str) -> bool:
    """Windows puts its own dictation on Win+H — a CHORD, which this tool cannot bind, so the two
    can never fire on the same gesture. Nothing to warn about, and the row still prints.
    """
    return False
