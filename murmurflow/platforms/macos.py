"""The macOS backend: avfoundation, CoreGraphics, AppleScript, launchd.

Everything here was written against a Mac and lived in :mod:`murmurflow.dictate`,
:mod:`murmurflow.hotkey` and :mod:`murmurflow.service` before the Windows port needed a seam. The
comments came with it, because most of them record a measurement or a trap and neither survives a
paraphrase.

Nothing in this file is imported directly by anything above it — :mod:`murmurflow.platforms`
selects it — so a name that is not in that module's contract is private to macOS by construction.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import plistlib
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from .. import config

NAME = "macOS"

# --- capture --------------------------------------------------------------------------------


def capture_available() -> tuple[bool, str]:
    """avfoundation ships with ffmpeg on macOS, so this is only ever about ffmpeg itself."""
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not found — brew install ffmpeg"
    return True, ""


def capture_args(device: str) -> list[str]:
    """ffmpeg's input flags for ``device``, an avfoundation index as a string.

    ``:N`` is avfoundation's "no video, audio device N" spelling. An empty device is the system
    default, which avfoundation writes as ``:default``.
    """
    return ["-f", "avfoundation", "-i", f":{device or 'default'}"]


def default_input() -> str:
    """avfoundation has a real "whatever the system picked" token, so nothing has to be listed."""
    return "default"


_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def list_inputs() -> list[tuple[str, str]]:
    """Every avfoundation AUDIO input as ``(index, name)``; ``[]`` if ffmpeg is missing/errors.

    ffmpeg prints its device list to stderr and exits non-zero by design (there is no input file),
    so a non-zero return here is normal and is not treated as failure.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
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
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        # Strip ffmpeg's "[AVFoundation indev @ 0x...] " prefix before matching the "[N] Name" pair.
        stripped = re.sub(r"^\[AVFoundation indev @ [^\]]+\]\s*", "", line).strip()
        match = _DEVICE_LINE.match(stripped)
        if match:
            devices.append((match.group(1), match.group(2)))
    return devices


# --- keys -----------------------------------------------------------------------------------
#
# The obvious way to do this is a global-hotkey library. It is not needed: macOS exposes the two
# facts we want through CoreGraphics C functions that ctypes can call directly, so the listener is
# a POLL LOOP and not an event tap. A CGEventTap would require the Input Monitoring TCC grant, a
# CFRunLoop and ~150 lines of callback glue; polling requires none of it and, on macOS 26, prompted
# for nothing. Nothing is compiled either, so a client's Mac needs no Xcode, no signing and no
# notarization, and the TCC grant cannot be invalidated by a rebuild.

# kCGEventSourceStateHIDSystemState — real hardware state, not the synthetic/session view. This is
# what makes the poll see a physical key press rather than events we ourselves post.
_HID_STATE = 1

# CGEventFlags bits (CGEventTypes.h). Modifier-held flags are shared between left and right keys,
# so a side-specific trigger is read via CGEventSourceKeyState on the virtual keycode instead.
FLAG_SHIFT = 1 << 17
FLAG_CONTROL = 1 << 18
FLAG_OPTION = 1 << 19
FLAG_COMMAND = 1 << 20
FLAG_FN = 1 << 23

# Virtual keycodes (Events.h / HIToolbox). Side-specific, unlike the flag bits above.
LEFT_CONTROL = 0x3B
LEFT_SHIFT = 0x38
LEFT_OPTION = 0x3A
LEFT_COMMAND = 0x37
RIGHT_OPTION = 0x3D
RIGHT_COMMAND = 0x36
RIGHT_SHIFT = 0x3C
RIGHT_CONTROL = 0x3E
F13 = 0x69

# kCGEventKeyDown — the event type we ask "how long since one of these?" to detect a chord.
_EVENT_KEY_DOWN = 10
# kCGEventLeftMouseDown / kCGEventRightMouseDown. A CLICK inside the press is a chord too: ⌘-click
# opens a link in a new tab, and two of those in a row are otherwise indistinguishable from a
# deliberate double-tap.
_EVENT_LEFT_MOUSE_DOWN = 1
_EVENT_RIGHT_MOUSE_DOWN = 3

#: Side-AGNOSTIC trigger names -> the modifier FLAG bit they poll.
#:
#: These exist because side-specific detection is a hardware promise this Mac does not keep. On the
#: MacBooks, ``CGEventSourceKeyState`` reports the RIGHT Command key as the LEFT one and the right
#: Option as the left — the right-side virtual keycodes simply never go true, so a trigger bound to
#: ``right_command`` can never fire. The flag bits are shared between both sides BY DEFINITION, so
#: they are read from the state that is actually maintained rather than from one we hoped was.
#: (Windows has no such problem and reads both sides correctly — see that backend.)
FLAG_TRIGGERS: dict[str, int] = {
    "command": FLAG_COMMAND,
    "option": FLAG_OPTION,
    "control": FLAG_CONTROL,
    "shift": FLAG_SHIFT,
    "fn": FLAG_FN,
}

#: COMBINATION triggers: every one of these flags must be held at once, and nothing types.
COMBO_TRIGGERS: dict[str, tuple[int, ...]] = {
    "control_option": (FLAG_CONTROL, FLAG_OPTION),
    "control_command": (FLAG_CONTROL, FLAG_COMMAND),
    "command_option": (FLAG_COMMAND, FLAG_OPTION),
    "control_shift": (FLAG_CONTROL, FLAG_SHIFT),
}

#: Trigger names accepted in config -> the keycode they poll.
KEYCODE_TRIGGERS: dict[str, int] = {
    "left_control": LEFT_CONTROL,
    "left_shift": LEFT_SHIFT,
    "left_option": LEFT_OPTION,
    "left_command": LEFT_COMMAND,
    "right_option": RIGHT_OPTION,
    "right_command": RIGHT_COMMAND,
    "right_shift": RIGHT_SHIFT,
    "right_control": RIGHT_CONTROL,
    "f13": F13,
}


def trigger_names() -> frozenset[str]:
    return frozenset(KEYCODE_TRIGGERS) | frozenset(FLAG_TRIGGERS) | frozenset(COMBO_TRIGGERS)


class _Unavailable(RuntimeError):
    """CoreGraphics cannot be reached — not macOS, or the framework is missing."""


_LIB: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    """Bind CoreGraphics' key-state functions; raise :class:`_Unavailable` if impossible."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        raise _Unavailable("ApplicationServices framework not found (is this macOS?)")
    try:
        lib = ctypes.CDLL(path)
        lib.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
        lib.CGEventSourceKeyState.restype = ctypes.c_bool
        lib.CGEventSourceFlagsState.argtypes = [ctypes.c_int32]
        lib.CGEventSourceFlagsState.restype = ctypes.c_uint64
        lib.CGEventSourceSecondsSinceLastEventType.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        lib.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
        # ...and the three that TYPE. Same framework, same ctypes, no new dependency.
        lib.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
        lib.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        lib.CGEventKeyboardSetUnicodeString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        lib.CGEventKeyboardSetUnicodeString.restype = None
        lib.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.CGEventSetFlags.restype = None
        lib.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        lib.CGEventPost.restype = None
    except (OSError, AttributeError) as exc:
        raise _Unavailable(f"CoreGraphics key-state API unavailable: {exc}") from exc
    _LIB = lib
    return lib


def keys_unavailable() -> str:
    try:
        _load()
    except _Unavailable as exc:
        return str(exc)
    return ""


def is_down(name: str) -> bool:
    """Is the trigger called ``name`` down right now — flag-based, combo or keycode-based.

    ONE seam, so a listener never has to know which kind of trigger it was given.
    """
    try:
        lib = _load()
    except _Unavailable:
        return False
    combo = COMBO_TRIGGERS.get(name)
    if combo is not None:
        state = int(lib.CGEventSourceFlagsState(_HID_STATE))
        return all(state & bit for bit in combo)
    flag = FLAG_TRIGGERS.get(name)
    if flag is not None:
        return bool(int(lib.CGEventSourceFlagsState(_HID_STATE)) & flag)
    code = KEYCODE_TRIGGERS.get(name, LEFT_CONTROL)
    return bool(lib.CGEventSourceKeyState(_HID_STATE, ctypes.c_uint16(code)))


def seconds_since_input() -> float:
    """Seconds since the last real key-down OR mouse click anywhere on the system.

    The whole chord guard: if this is smaller than how long the trigger has been held, the user
    acted WHILE holding it — they are typing ⌘S or ⌘-clicking a link, not dictating.
    """
    try:
        lib = _load()
    except _Unavailable:
        return float("inf")
    return min(
        float(lib.CGEventSourceSecondsSinceLastEventType(_HID_STATE, event))
        for event in (_EVENT_KEY_DOWN, _EVENT_LEFT_MOUSE_DOWN, _EVENT_RIGHT_MOUSE_DOWN)
    )


# --- inject ---------------------------------------------------------------------------------


def input_blocked() -> str:
    """Why a synthetic keystroke will be swallowed right now, or ``""``.

    Secure Input: while some app has it on (a password field, some terminals) the OS refuses ALL
    synthetic keyboard events, so a paste silently does nothing — the single most confusing failure
    this feature can have ("it heard me but typed nothing").
    """
    try:
        path = ctypes.util.find_library("Carbon")
        if not path:
            return ""
        carbon = ctypes.CDLL(path)
        carbon.IsSecureEventInputEnabled.restype = ctypes.c_bool
        if not bool(carbon.IsSecureEventInputEnabled()):
            return ""
    except Exception:  # noqa: BLE001 — a diagnostic; never block a paste because it failed
        return ""
    return (
        "Secure Input is active (a password field or terminal has keyboard entry locked), so "
        "macOS is blocking synthetic paste."
    )


def clipboard_set(text: str) -> bool:
    """Put ``text`` on the clipboard. ``True`` on success."""
    binary = shutil.which("pbcopy")
    if not binary:
        return False
    try:
        proc = subprocess.run(
            binary, input=text, text=True, timeout=5, check=False, capture_output=True
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# The whole injection, in ONE AppleScript. Paste is chosen over typing the string
# character-by-character (`keystroke "<text>"`) because it is O(1) instead of O(chars) — a
# 300-character dictation types visibly, one letter at a time, over several seconds — and because
# it is immune to the keyboard-layout mangling that synthetic per-character events hit with German
# umlauts. The text arrives through `argv`, so the AppleScript quoting traps of an inlined string
# never apply either.
#
# Save and restore live INSIDE the script because `the clipboard as record` is the only thing that
# can hold what was there. It carries EVERY flavour the pasteboard offers — a copied image, a
# styled snippet, a file — where `pbpaste` sees text and nothing else. That gap was a real loss,
# not a theoretical one: copy a screenshot while a dictation is in flight and the transcript ate
# it, because a `pbpaste` that returned "" read as "nothing to put back".
#
# Order is the design: save, overwrite, paste, settle, restore. If the paste raises — Secure Input,
# no Accessibility grant — the script aborts BEFORE the restore, leaving your words on the
# clipboard for a manual Cmd-V, which is exactly what the error text promises.
#
# AND IT REPORTS WHAT IT DID. A dictation that lands in full and one that lands as its first eight
# words are the SAME `[OK]` line in the daemon log, because a zero exit from `osascript` only says
# the keystroke was posted. Three facts split that open, and the report they answer ("when I talk
# longer, only a few words paste") could not be diagnosed without them:
#
#   held    the clipboard's length right after the transcript is written to it. Short means the
#           COPY truncated, and the paste never had the words to give.
#   still   its length again after the settle, before the restore. Short means something took the
#           pasteboard mid-flight and the target read that instead.
#   target  the app the keystroke actually went to. "it drops in iTerm and never in Mail" is a
#           whole diagnosis, and it was unanswerable from this log.
#
# Tab-separated on stdout: a transcript can contain any other separator, but never a tab.
_INJECT_SCRIPT = """
on run argv
	set saved to missing value
	try
		set saved to (the clipboard as record)
	end try
	set the clipboard to (item 1 of argv)
	set held to length of (the clipboard as text)
	set target to "?"
	try
		tell application "System Events" to set target to name of first process whose frontmost is true
	end try
	tell application "System Events" to keystroke "v" using command down
	delay {settle}
	set still to -1
	try
		set still to length of (the clipboard as text)
	end try
	if saved is not missing value then
		try
			set the clipboard to saved
		end try
	end if
	return (held as text) & tab & (still as text) & tab & target
end run
"""


def _paste_note(stdout: str, sent: int) -> str:
    """The diagnostic clause for the daemon log, read off what the script reported.

    Always present on a successful paste, never only on a suspicious one: a line that appears only
    when something looks wrong is a line nobody has a baseline for, and the baseline is the whole
    point of writing it. Never raises — a diagnostic that can break a paste is worse than no
    diagnostic, so anything unparseable degrades to ``""`` and the log reads exactly as before.
    """
    parts = (stdout or "").strip("\n").split("\t")
    if len(parts) != 3:
        return ""
    try:
        held, still = int(parts[0]), int(parts[1])
    except ValueError:
        return ""
    note = f"→ {parts[2].strip() or '?'}"
    if held != sent:
        note += f" · COPIED ONLY {held}/{sent}"
    elif still != held:
        # -1 is "the clipboard was no longer text at all", which is the same story: our words were
        # not there to be read by the time the target got round to reading them.
        note += f" · CLIPBOARD CHANGED UNDER THE PASTE ({still}/{sent})"
    return note


def inject(text: str, settle: float) -> tuple[bool, str, str]:
    """Clipboard-swap + synthetic ⌘V, then put back whatever was on the clipboard.

    ``(ok, problem, note)``. The note is for the log and is never shown to anybody as an error.

    Requires the **Accessibility** TCC grant for whichever process runs this. That is the one
    permission dictation genuinely cannot avoid: it is what "type into another app" *means* here.
    """
    osascript = shutil.which("osascript")
    if not osascript:
        return False, "osascript missing — is this macOS?", ""
    try:
        proc = subprocess.run(
            [osascript, "-e", _INJECT_SCRIPT.format(settle=f"{settle:.2f}"), text],
            capture_output=True,
            text=True,
            # The restore re-materializes every pasteboard flavour, which on a large copied image
            # is seconds of work — generous enough that a big screenshot is never dropped.
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"paste failed: {exc}", ""
    if proc.returncode == 0:
        return True, "", _paste_note(proc.stdout, len(text))
    detail = (proc.stderr or "").strip().splitlines()
    hint = detail[-1] if detail else "unknown error"
    if "not allowed" in hint.lower() or "1002" in hint:
        hint = permission_hint()
    return False, f"paste failed: {hint}.", ""


# --- typing, for text that arrives while you are still talking ------------------------------------
#
# THE CLIPBOARD ROUND TRIP IS HALF THE STREAMING CYCLE. `inject` below saves the pasteboard, writes
# the text, sends Cmd-V, waits for the target to read it and puts the old contents back — measured
# at ~500ms, against ~420ms to actually decode the audio. So the words arrived in two-and-three-word
# lumps about once a second, and the report was the obvious one: "it is lagging behind".
#
# A unicode key event carries the characters ITSELF. Nothing touches the pasteboard, nothing has to
# settle, and there is no Cmd-V to be caught by a held modifier. Measured on the same machine:
# 2.6ms for a 25-character chunk, 190x cheaper than the paste it replaces.
#
# And it is safe on a German keyboard, which is the reason the clipboard was chosen in the first
# place. `keystroke "text"` in AppleScript sends KEYCODES, which the target re-maps through its own
# layout and mangles every umlaut. `CGEventKeyboardSetUnicodeString` sends the characters, so
# "Förderung" arrives as "Förderung" on any layout there is.
#
# ponytail: streamed chunks only. The FINAL transcript still goes through the clipboard, because it
# can be two thousand characters at once and because its paste reports back what the target actually
# received — see `_paste_note`, which is the whole diagnosis for "only half my sentence arrived".

#: kCGSessionEventTap — post into this login session, ahead of the app that has focus.
_SESSION_TAP = 1

#: CoreFoundation, for the one thing ctypes cannot do for us: releasing an event we created. Without
#: it every streamed chunk leaks a CFTypeRef, a hundred times a dictation, in a process that runs
#: from login to shutdown.
_CF = ctypes.CDLL(ctypes.util.find_library("CoreFoundation") or "CoreFoundation")
_CF.CFRelease.argtypes = [ctypes.c_void_p]
_CF.CFRelease.restype = None

#: UTF-16 units per event. CGEventKeyboardSetUnicodeString takes an arbitrary count, but long
#: strings are unreliable in practice, so the text is posted in small pieces.
_TYPE_CHUNK = 16


def _chunk_end(units: ctypes.Array[ctypes.c_uint16], start: int, total: int) -> int:
    """Where the piece beginning at ``start`` ends — never between the halves of one character.

    A non-BMP character (an emoji) is two UTF-16 units, and both have to reach
    ``CGEventKeyboardSetUnicodeString`` in the same string. Split across two events the app is
    handed half a character and drops it, silently, while everything reports success.
    """
    end = min(start + _TYPE_CHUNK, total)
    if end < total and end - 1 > start and 0xD800 <= units[end - 1] <= 0xDBFF:
        end -= 1  # a high surrogate at the seam: let its other half come with it, next time round
    return end


def type_text(text: str) -> str:
    """Type ``text`` into the focused app as unicode key events. Returns what did NOT go out.

    ``""`` means all of it landed; the whole string back means none of it did, and the caller falls
    back to the clipboard — this needs the same Accessibility grant a paste does, and a machine
    that cannot post events must still dictate.

    **The remainder, and not a bool.** The text goes out in pieces, so a failure halfway has
    already put some of it on screen. A caller told only "that did not work" would paste the whole
    chunk over the top of the half that landed, and the words in the middle would appear twice.
    """
    if not text:
        return ""
    try:
        lib = _load()
    except _Unavailable:
        return text
    units = text.encode("utf-16-le", errors="ignore")
    buffer = (ctypes.c_uint16 * (len(units) // 2)).from_buffer_copy(units)
    total = len(buffer)
    start = 0
    try:
        while start < total:
            end = _chunk_end(buffer, start, total)
            payload = (ctypes.c_uint16 * (end - start))(*buffer[start:end])
            for pressed in (True, False):
                event = lib.CGEventCreateKeyboardEvent(None, 0, pressed)
                if not event:
                    return units[start * 2 :].decode("utf-16-le", errors="ignore")
                # No modifiers, whatever the hands are doing. The characters are carried by the
                # event, so nothing here is a shortcut that a held Control could turn into one.
                lib.CGEventSetFlags(event, 0)
                lib.CGEventKeyboardSetUnicodeString(event, len(payload), payload)
                lib.CGEventPost(_SESSION_TAP, event)
                _CF.CFRelease(event)
            start = end
    except Exception:  # noqa: BLE001 — a failed keystroke falls back to the paste, never crashes
        return units[start * 2 :].decode("utf-16-le", errors="ignore")
    return ""


# --- the one sound ------------------------------------------------------------------------------

#: The Mac's own "ready" tick. A system sound rather than a generated tone: it is one people have
#: heard for twenty years, it needs no file shipped or cached, and it follows the alert-volume
#: slider, which nothing we synthesise can see.
READY_SOUND = "/System/Library/Sounds/Tink.aiff"


def play_ready() -> None:
    """Say the microphone is live, once, without blocking. Silence, never a crash, if it cannot."""
    player = shutil.which("afplay")
    if not player or not Path(READY_SOUND).is_file():
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.Popen(
            [player, READY_SOUND],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def input_permitted() -> bool:
    """Has THIS binary been granted Accessibility — the grant that lets the text be typed?

    ``AXIsProcessTrusted`` and never ``AXIsProcessTrustedWithOptions``: the options form is the one
    that raises a system dialog, and a diagnostic that puts up a modal every time it runs is worse
    than the question it answers.
    """
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        return False
    try:
        lib = ctypes.CDLL(path)
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except (OSError, AttributeError):
        return False


def permission_hint() -> str:
    """Naming the wrong process here wastes an afternoon.

    The grant belongs to whatever is RUNNING THE LISTENER, which under the launchd agent is the
    bundled ``MurmurFlow`` binary and not the terminal you happen to be looking at. Granting
    Terminal makes ``murmurflow listen`` work by hand while the installed agent stays silently
    blocked — exactly the confusing half-working state this message exists to prevent.
    """
    return (
        "not permitted to control this Mac. Accessibility must be granted to the process "
        f"running the listener — this one is {Path(sys.executable).name} at "
        f"{sys.executable}. Add it in System Settings > Privacy & Security > "
        "Accessibility (the apps you dictate INTO never need permission)"
    )


# --- sound ----------------------------------------------------------------------------------

# --- service --------------------------------------------------------------------------------

LABEL = "ai.murmurflow.listen"
APP_NAME = "MurmurFlow"
BUNDLE_ID = "ai.murmurflow"

# launchd runs agents with a MINIMAL PATH that does not include /opt/homebrew/bin or
# /usr/local/bin, so an agent that shells out to ffmpeg and whisper-server silently fails with no
# useful error. Every rendered plist therefore declares its own PATH. This is the single most
# common way a working dictation setup breaks the moment it is installed rather than run by hand.
_AGENT_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def service_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


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
    reinstall; ``PYTHONHOME``/``PYTHONPATH`` in the job are what let a 17 MB interpreter sitting
    outside its own prefix still find its standard library and this package.

    **Windows needs none of this** — there is no TCC, so no identity to defend and no bundle to
    build. That asymmetry is why ``build_app`` is a macOS name and not a seam name.
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
    different identities on purpose.
    """
    executable = app_path() / "Contents" / "MacOS" / APP_NAME
    if not executable.is_file():
        return None
    env = {**os.environ, **bundle_env()}
    code = "from murmurflow import platforms; print(platforms.input_permitted())"
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


def _launchd_argv(args: list[str]) -> list[str]:
    """The argv that starts the listener, resolved absolutely because launchd has no PATH.

    The .app bundle first, so the Privacy rows carry this tool's name. Then the installed
    ``murmurflow`` console script, then ``python -m murmurflow`` at this interpreter, which is what
    makes an editable/checkout install work without a second step.
    """
    bundled = build_app()
    if bundled is not None:
        return [str(bundled), "-m", "murmurflow", *args]
    found = shutil.which("murmurflow")
    if found:
        return [found, *args]
    return [sys.executable, "-m", "murmurflow", *args]


def render_plist(args: list[str], extra_env: dict[str, str]) -> bytes:
    """The launchd job. ``KeepAlive`` because a dictation daemon that quietly died is worse than
    one that never started — you find out by holding the key and getting nothing back.

    ``ProcessType: Interactive`` is LOAD-BEARING, not a nicety. Without it launchd applies its
    default throttle to this agent and to the ffmpeg it spawns, and a throttled process cannot
    drain a CoreAudio input buffer at real time. Measured on an M4 Pro: a 5.0s hold captured
    **1.27s** of audio under the installed agent and **4.05s** running the identical code from a
    terminal. The microphone opens fine either way (~300ms), the level is healthy either way —
    three quarters of every sentence is simply never recorded, reaches whisper as a short clip of
    clipped words, and comes back as confident nonsense. If you ever see "it works when I run it by
    hand and it is useless once installed", this line is why. (The daemon log now prints the
    captured length beside the hold, so the same regression can never again be invisible.)

    ``ThrottleInterval`` so a machine that has not been granted microphone access backs off instead
    of hot-looping on the device.
    """
    env = {"PATH": _AGENT_PATH, **extra_env}
    argv = _launchd_argv(args)
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
        "StandardOutPath": str(config.log_path()),
        "StandardErrorPath": str(config.log_path()),
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


def service_install(args: list[str], env: dict[str, str]) -> tuple[bool, str]:
    """Write the plist and load it. Idempotent: an already-loaded agent is booted out first.

    The bootout is unconditional and its failure is ignored on purpose — "was not loaded" is the
    expected answer on a first install, and treating it as an error would make the happy path look
    broken.
    """
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config.log_path().parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_plist(args, env))
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # ignore: not-loaded is the normal first case
    ok, detail = _launchctl("bootstrap", _domain(), str(path))
    if not ok:
        # Older macOS, and the fallback that still works everywhere.
        ok, detail = _launchctl("load", "-w", str(path))
    return ok, detail


def service_uninstall() -> tuple[bool, str]:
    """Unload the agent and delete its plist. Succeeds when the end state is "not installed"."""
    path = service_path()
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    _launchctl("unload", str(path))
    path.unlink(missing_ok=True)
    return True, ""


def service_restart() -> bool:
    """Bounce the agent so it re-reads its config. False if it was not running."""
    if not service_running():
        return False
    ok, _ = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    return ok


def service_running() -> bool:
    ok, _ = _launchctl("print", f"{_domain()}/{LABEL}")
    return ok


def dictation_conflict(trigger: str) -> bool:
    """True if macOS's OWN dictation shortcut would fire on the same gesture as ours.

    Apple puts dictation on a double-tap of Control by default. If that is still enabled and the
    user's trigger is a bare Control, both fire: Apple's microphone panel appears on top of this
    one and neither transcript is what was wanted. It is the single most confusing collision this
    tool has, and it was documented in prose nobody reads — so it is a checked row instead.
    """
    if trigger not in {"left_control", "right_control", "control"}:
        return False
    try:
        proc = subprocess.run(
            ["defaults", "read", "com.apple.assistant.support", "Dictation Enabled"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.stdout.strip() == "1"
