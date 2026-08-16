"""Global press-to-talk key detection on macOS, in stdlib :mod:`ctypes` and nothing else.

The obvious way to do this is a global-hotkey listener library and a synthetic-keystroke library,
both of them dependencies. Neither is needed. macOS exposes the two facts we want through
CoreGraphics C functions that :mod:`ctypes` can call directly:

* ``CGEventSourceFlagsState`` — which modifier keys are held right now.
* ``CGEventSourceKeyState``   — whether a given virtual keycode is held right now.

So the listener is a **poll loop**, not an event tap. That matters more than it sounds:

* A ``CGEventTap`` requires the **Input Monitoring** TCC grant, a ``CFRunLoop``, and ~150 lines of
  fragile callback glue. Polling requires none of it — these two functions read HID state and, in
  testing on macOS 26, prompted for nothing.
* An Automator/Shortcuts hotkey (the other dependency-free option) fires on key-DOWN only, with no
  key-UP callback, so it **structurally cannot** do hold-to-talk. Polling can: key down starts the
  recording, key up ends it.
* Nothing is compiled, so a client's Mac needs no Xcode, no code signing and no notarization — and
  the TCC grant cannot be invalidated by a rebuild (the trap that bites ad-hoc-signed helpers).

**Cost of the shortcut.** A poll loop cannot *consume* the keystroke, so the trigger must be a key
the frontmost app will not miss. The default is **left Control** (:data:`LEFT_CONTROL`) — where
macOS itself puts dictation, so the hand already knows it. A modifier types nothing on its own, and
the two guards in :func:`listen` (:data:`INTENT_DELAY`, then the chord abort) mean a ``⌃C`` never
starts the microphone at all.

Right *Option* is deliberately never the default, and that is not a preference: on the German layout
right Option is AltGr, the dead key for ``@ € \\ | ~ [ ] { }``. Anyone typing German all day would
fire the microphone on virtually every email address and code bracket.
``fn``/Globe is out too — macOS claims it system-wide for the emoji picker and dictation, and a poll
cannot suppress that. ``F13`` is not on a built-in MacBook keyboard.

ponytail: a 60 Hz poll of two C calls, not an event tap. The ceiling is that we observe rather than
intercept the key; upgrade to a ``CGEventTap`` only if a trigger key that must be swallowed is ever
required.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import time
from collections.abc import Callable

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

#: Side-AGNOSTIC trigger names -> the modifier FLAG bit they poll (``CGEventSourceFlagsState``).
#:
#: These exist because side-specific detection is a hardware promise this Mac does not keep. On the
#: MacBooks, ``CGEventSourceKeyState`` reports the RIGHT Command key as the LEFT one and
#: the right Option as the left — the right-side virtual keycodes simply never go true, so a
#: trigger bound to ``right_command`` can never fire, which is exactly the "doesn't work in any way
#: whatsoever" report. The flag bits are shared between both sides BY DEFINITION, so they are read
#: from the state that is actually maintained rather than from one we hoped was.
#:
#: The cost is honest and small: ``command`` means EITHER Command key. That is the correct trade —
#: a trigger that works on both sides beats a trigger that works on neither.
FLAG_TRIGGERS: dict[str, int] = {
    "command": FLAG_COMMAND,
    "option": FLAG_OPTION,
    "control": FLAG_CONTROL,
    "shift": FLAG_SHIFT,
    "fn": FLAG_FN,
}

#: COMBINATION triggers: every one of these flags must be held at once, and nothing types.
#:
#: This is the answer to the real cost of polling. A poll cannot CONSUME a key, so a single bare
#: modifier is shared with everything the user already does with it: ⌃C, ⌃←, ⌃-click. The chord
#: guard discards the audio, but the guard only has :data:`CHORD_GRACE` to notice, and a person who
#: holds Control and *then* reaches for the arrow key beats it — so the microphone opens and the
#: cue plays for a keystroke that was never dictation. Two modifiers together are bound to nothing
#: in macOS, are typed by nobody in the course of ordinary work, and need no setting turned off
#: first — unlike Fn (the emoji picker) or double-tap Control (Apple's own dictation).
COMBO_TRIGGERS: dict[str, tuple[int, ...]] = {
    "control_option": (FLAG_CONTROL, FLAG_OPTION),
    "control_command": (FLAG_CONTROL, FLAG_COMMAND),
    "command_option": (FLAG_COMMAND, FLAG_OPTION),
    "control_shift": (FLAG_CONTROL, FLAG_SHIFT),
}

#: Trigger names accepted in config (``trigger``) -> the keycode they poll.
TRIGGERS: dict[str, int] = {
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

# Control+Option held together. It used to be bare left Control, "where macOS itself puts
# dictation, so the hand already knows where it lives" — which is true, and was still the wrong
# default, because it is also the most CHORDED modifier on the machine.
#
# The chord guard does discard the audio from a ⌃C. It cannot discard the EXPERIENCE: the guard has
# only `CHORD_GRACE` to notice, and holding Control and then reaching for an arrow key (Mission
# Control's space switch, which everybody uses) beats it — so the microphone opens and a tone plays
# for a keystroke that was never dictation. A tool that chimes while you work gets uninstalled, and
# no amount of correct discarding fixes that.
#
# Two modifiers together are typed by nobody in the course of ordinary work, are bound to nothing
# in macOS, and — unlike Fn (the emoji picker) or double-tap Control (Apple's own dictation) —
# need no system setting turned off before they are safe. ⌃⌥ specifically because on a Mac
# keyboard they are adjacent, so one hand reaches both without moving.
#
# THE GESTURE DECIDES THE KEY, which is why there are two defaults and not one.
#
# A HOLD needs a combo, for the reason above: a hold starts the moment the key goes down, and on a
# bare modifier that moment is indistinguishable from the start of ⌃C.
#
# A DOUBLE-TAP does not, and this is the whole point — two deliberate taps of one key inside half a
# second is not a gesture anybody performs by accident, and a shortcut that could imitate it (⌃C
# then ⌃C) is thrown out by the chord guard in `listen_double_tap`. It is exactly why macOS itself
# puts dictation on double-tap Control. So double-tapping TWO keys at once, which is what a combo
# default forced, is awkward for no safety it was not already getting for free.
#
# Portability is the second reason to keep the tap default a single ordinary modifier: Control
# exists, is named the same, and is reachable on every keyboard there is. `control_option` is a
# macOS-shaped answer; `left_control` is not.
#
# The one conflict, and it is detected in `murmurflow doctor` rather than left as folklore: if
# macOS's own "press Control twice for dictation" is enabled, both fire.
# System Settings > Keyboard > Dictation > Shortcut.
DEFAULT_TRIGGER = "control_option"
DEFAULT_TAP_TRIGGER = "left_control"

#: Names for the same physical keys on a keyboard that is not Apple's. The config vocabulary should
#: not have to be relearned to move this to another OS, and `alt` is what the key is called
#: everywhere else. Aliases only — the canonical names above are what gets stored and printed.
TRIGGER_ALIASES: dict[str, str] = {
    "ctrl": "control",
    "alt": "option",
    "left_alt": "left_option",
    "right_alt": "right_option",
    "control_alt": "control_option",
    "ctrl_alt": "control_option",
    "ctrl_option": "control_option",
    "win": "command",
    "super": "command",
    "meta": "command",
}


def canonical_trigger(name: str) -> str:
    """One spelling per key, so `ctrl_alt` and `control_option` cannot behave differently."""
    key = str(name).strip().lower().replace("-", "_").replace("+", "_")
    return TRIGGER_ALIASES.get(key, key)


# DEAD TIME IS LOST WORDS. The recorder now starts on key-DOWN, immediately, and the chord guard
# DISCARDS the clip if a shortcut turns out to be what was happening. The reverse — wait, then start
# — is what shipped first, and it cost the user the first half-second of every sentence:
# 180ms of waiting PLUS the ~300ms CoreAudio needs to open the mic, ~480ms before a single sample
# lands. Transcripts came back truncated mid-thought ("Hold it, go in, give me this test of the
# first motor") while the identical audio recorded with a countdown transcribed perfectly.
#
# Starting first makes the guard window OVERLAP the warm-up instead of preceding it, so the dead
# time is the ~300ms CoreAudio floor and nothing more. The cost is real and accepted: a ⌃C now
# spawns an ffmpeg that lives ~100ms before the chord guard kills it. Mic churn is cheap; the first
# words of a sentence are not.
#
# ponytail: the floor is CoreAudio's device-open, not our code — a resident always-on recorder would
# reach zero, and is deliberately NOT built: a permanently hot microphone is the exact shape of the
# privacy incident this product exists to avoid.
CHORD_GRACE = 0.18

# DOUBLE-TAP (hands-free) mode. Two taps of the trigger within this window start recording; one tap
# stops it. This is what macOS's own dictation does, and on a heavily-chorded key like Control it is
# easier on the hand than holding. Hold stays the default because it is faster for one short
# sentence — you are already holding the key you pressed.
#
# This used to claim a chord "can NEVER look like a double-tap". It can, and does: ⌃C then ⌃C in a
# terminal is two short Control presses inside the window. The guard is in `listen_double_tap` — a
# press with a real keystroke inside it is a shortcut, not a tap — and it is what the claim was
# standing in for.
#
# Measured PRESS TO PRESS, and that is the whole reliability of the gesture. Release-to-release —
# what shipped first — makes the second tap's own duration count against the budget: a 200ms gap
# plus a 200ms press blows a 350ms window, so an ordinary double-tap read as two unrelated taps and
# nothing happened at all. Press-to-press is also how macOS measures a double-click, and 500ms is
# its default there.
DOUBLE_TAP_WINDOW = 0.50

# A tap is a press SHORTER than this. Longer and it is a hold, not a tap — which is what lets both
# modes coexist on one key rather than needing two. Generous on purpose: a tap that lingers is the
# common human miss, and reading it as a hold silently breaks the pair.
TAP_MAX = 0.50

# Virtual keycode of the LAST key macOS saw go down, used only to notice that the user pressed
# a real key while holding the trigger — i.e. typing ⌘S, not dictating. Any such chord aborts
# the recording, which is what makes a bare modifier safe to bind at all.
_ABORT_ON_CHORD = True

# Slack in the "was a real key pressed DURING this trigger press?" comparison. The poll runs at
# 60Hz and the two clocks are read a tick apart, so an exact comparison would miss a keystroke that
# landed in the same frame as the release. Small enough that a genuine tap a moment after typing is
# still a tap.
_CHORD_EPSILON = 0.02

# 60 Hz. Fast enough that press/release feels instant (16ms granularity is below the ~100ms a human
# perceives as lag) and cheap enough to be invisible: two C calls per tick is well under 1% of one
# core. Polling faster buys nothing a person can feel.
POLL_HZ = 60


class Unavailable(RuntimeError):
    """Raised when CoreGraphics cannot be reached — not macOS, or the framework is missing."""


# kCGEventKeyDown — the event type we ask "how long since one of these?" to detect a chord.
_EVENT_KEY_DOWN = 10
# kCGEventLeftMouseDown / kCGEventRightMouseDown. A CLICK inside the press is a chord too: ⌘-click
# opens a link in a new tab, and two of those in a row are otherwise indistinguishable from a
# deliberate double-tap. Only a key-down was checked while the trigger was a Control key, where
# nobody chords with the mouse; on Command it is the common case.
_EVENT_LEFT_MOUSE_DOWN = 1
_EVENT_RIGHT_MOUSE_DOWN = 3


def _load() -> ctypes.CDLL:
    """Bind CoreGraphics' key-state functions; raise :class:`Unavailable` if impossible."""
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        raise Unavailable("ApplicationServices framework not found (is this macOS?)")
    try:
        lib = ctypes.CDLL(path)
        lib.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
        lib.CGEventSourceKeyState.restype = ctypes.c_bool
        lib.CGEventSourceFlagsState.argtypes = [ctypes.c_int32]
        lib.CGEventSourceFlagsState.restype = ctypes.c_uint64
        lib.CGEventSourceSecondsSinceLastEventType.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        lib.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
    except (OSError, AttributeError) as exc:
        raise Unavailable(f"CoreGraphics key-state API unavailable: {exc}") from exc
    return lib


def accessibility_trusted() -> bool:
    """Has THIS binary been granted Accessibility — the grant that lets the text be typed?

    ``AXIsProcessTrusted`` and never ``AXIsProcessTrustedWithOptions``: the options form is the one
    that raises a system dialog, and a diagnostic that puts up a modal every time it runs is worse
    than the question it answers. The grant attaches to the EXECUTABLE, and the daemon and this CLI
    run the same interpreter, so asking on our own behalf also answers for the daemon.
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


def seconds_since_keydown(lib: ctypes.CDLL | None = None) -> float:
    """Seconds since the last real key-down OR mouse click anywhere on the system.

    The whole chord guard: if this is smaller than how long the trigger has been held, the user
    acted WHILE holding it — they are typing ⌘S or ⌘-clicking a link, not dictating. Clicks count
    as well as keys, because on a Command trigger the mouse is half of the shortcuts.
    """
    lib = lib or _load()
    return min(
        float(lib.CGEventSourceSecondsSinceLastEventType(_HID_STATE, event))
        for event in (_EVENT_KEY_DOWN, _EVENT_LEFT_MOUSE_DOWN, _EVENT_RIGHT_MOUSE_DOWN)
    )


def is_trigger_down(name: str, lib: ctypes.CDLL | None = None) -> bool:
    """Is the trigger called ``name`` down right now — flag-based or keycode-based.

    ONE seam, so a listener never has to know which kind of trigger it was given. A side-agnostic
    name (``command``) reads the modifier flag; a side-specific one (``left_control``) reads the
    virtual keycode, which is the only way to tell the two Controls apart — where the hardware
    reports them apart at all.
    """
    lib = lib or _load()
    key = canonical_trigger(name)
    combo = COMBO_TRIGGERS.get(key)
    if combo is not None:
        state = flags(lib)
        return all(state & bit for bit in combo)
    flag = FLAG_TRIGGERS.get(key)
    if flag is not None:
        return bool(flags(lib) & flag)
    return is_held(keycode(name), lib)


def available() -> bool:
    """True if global key state can be read on this machine."""
    try:
        _load()
    except Unavailable:
        return False
    return True


def keycode(name: str) -> int:
    """The virtual keycode for a configured trigger name.

    The fallback is LEFT_CONTROL and not ``TRIGGERS[DEFAULT_TRIGGER]``: the default is a combo now,
    and a combo has no single keycode. Nothing reaches here with a combo name — `is_trigger_down`
    handles those first — so this is only the answer for a name nobody recognises.
    """
    return TRIGGERS.get(canonical_trigger(name), LEFT_CONTROL)


def is_held(code: int, lib: ctypes.CDLL | None = None) -> bool:
    """Whether the key with virtual keycode ``code`` is physically held right now."""
    lib = lib or _load()
    return bool(lib.CGEventSourceKeyState(_HID_STATE, ctypes.c_uint16(code)))


def flags(lib: ctypes.CDLL | None = None) -> int:
    """The raw modifier-flags bitmask currently held (for diagnostics)."""
    lib = lib or _load()
    return int(lib.CGEventSourceFlagsState(_HID_STATE))


def listen(
    on_press: Callable[[], None],
    on_release: Callable[[], None],
    *,
    trigger: str = DEFAULT_TRIGGER,
    min_hold: float = 0.15,
    should_stop: Callable[[], bool] | None = None,
    on_abort: Callable[[], None] | None = None,
    chord_grace: float = CHORD_GRACE,
    poll_hz: int = POLL_HZ,
) -> None:
    """Block, calling ``on_press`` when the trigger goes down and ``on_release`` when it comes up.

    ``on_press`` fires IMMEDIATELY on key-down, so the microphone starts opening while the user
    is still deciding to talk — see :data:`CHORD_GRACE` for why waiting first cost him the opening
    words of every sentence.

    **Chord abort.** Holding ⌃ and pressing C is "interrupt", not "dictate". If a real key goes down
    while the trigger is held, the recording is abandoned via ``on_abort`` (which must DISCARD, not
    transcribe) and no release fires for that press. Without this guard, binding a bare modifier
    would dictate on every keyboard shortcut the user uses. ``chord_grace`` bounds how long
    after the press a key-down still counts as a chord; past it, a keystroke is someone typing in
    another window, not this gesture.

    ``min_hold`` swallows accidental brushes: a press shorter than this still fires the pair, and
    the caller decides what a too-short clip means — suppressing it here would strand the recording
    that ``on_press`` already started.

    ``should_stop`` is polled each tick so a daemon can shut down cleanly. A callback that raises is
    never allowed to kill the loop: the listener is the one process standing between the user
    and the dictation, so it keeps going rather than dying silently at 3am.
    """
    lib = _load()
    interval = 1.0 / max(1, poll_hz)
    held = False  # the trigger is physically down and on_press has fired
    aborted = False
    pressed_at = 0.0
    while True:
        if should_stop is not None and should_stop():
            if held and not aborted:  # never leave a recording running on shutdown
                _safe(on_release)
            return
        now_held = is_trigger_down(trigger, lib)
        elapsed = time.monotonic() - pressed_at
        # A key-down MORE RECENT than the trigger press is a shortcut, not speech. Bounded by
        # chord_grace so that typing in another window a minute into a long dictation cannot
        # retroactively cancel it.
        chord = (
            _ABORT_ON_CHORD
            and held
            and not aborted
            and elapsed <= chord_grace
            and seconds_since_keydown(lib) < elapsed
        )
        if now_held and not held:
            held, aborted = True, False
            pressed_at = time.monotonic()
            _safe(on_press)  # start the mic NOW; the guard below discards if this was a chord
        elif chord:
            aborted = True
            _safe(on_abort or on_release)
        elif not now_held and held:
            held = False
            if aborted:
                aborted = False
                continue  # a shortcut: already discarded, there is nothing to finish
            if elapsed < min_hold:
                time.sleep(min_hold)  # let the mic collect something before we cut it
            _safe(on_release)
        time.sleep(interval)


def listen_double_tap(
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
    *,
    trigger: str = DEFAULT_TRIGGER,
    should_stop: Callable[[], bool] | None = None,
    window: float = DOUBLE_TAP_WINDOW,
    tap_max: float = TAP_MAX,
    poll_hz: int = POLL_HZ,
    on_tap: Callable[[str], None] | None = None,
) -> None:
    """Hands-free mode: double-tap the trigger to start talking, tap once to stop.

    Why this exists beside :func:`listen`: on Control — the key the user actually wants, because
    it is where macOS puts dictation — every ``⌃C``/``⌃D``/``⌃R`` looks like the beginning of a
    hold. Hold mode handles that correctly (the chord guard discards the clip) but the user
    still SEES the machine react to a keystroke that was never meant for it. A double-tap is a
    deliberate gesture, so the interaction stays silent until it is genuinely asked for.

    ``on_tap`` (optional) is told about every press this loop DECIDES something about — ``"tap"``,
    ``"hold"``, ``"chord"``, ``"start"``, ``"stop"``. It exists because the alternative failure is
    unanswerable: a listener that reacts to nothing looks identical whether the key is never read,
    the taps are too far apart, or the chord guard is eating them. The user hit exactly that on
    a newly-bound key ("I can't hear any sound when I double click the right command"), and there
    was nothing anywhere to tell him which of the three it was.

    **One chord is not a start — but two in a row were.** This originally reasoned that "a shortcut
    is one press, and one press is never a start", and applied no chord guard at all. ⌃C then ⌃C in
    a terminal is two short Control presses inside :data:`DOUBLE_TAP_WINDOW`, so recording began
    behind the user's back and the next stray tap ended it as a too-short clip — failure cues
    arriving "out of nowhere" mid-work. A press with a real keystroke inside it is now
    discarded as the shortcut it was.
    """
    lib = _load()
    interval = 1.0 / max(1, poll_hz)
    held = False
    pressed_at = 0.0
    last_tap = -999.0
    recording = False

    def saw(what: str) -> None:
        if on_tap is not None:
            _safe(lambda: on_tap(what))

    while True:
        if should_stop is not None and should_stop():
            if recording:
                _safe(on_stop)
            return
        now = time.monotonic()
        now_held = is_trigger_down(trigger, lib)
        if now_held and not held:
            held, pressed_at = True, now
        elif not now_held and held:
            held = False
            if now - pressed_at > tap_max:
                last_tap = -999.0  # a long hold is not a tap; it cannot open a double-tap pair
                saw("hold")
                continue
            # A CHORD IS NOT A TAP. This module used to claim a chord "can NEVER look like a
            # double-tap" — false, and the user found it: ⌃C then ⌃C in a terminal is two short
            # Control presses inside the window, which is exactly the gesture. Recording started
            # unnoticed and the next stray tap ended it as a too-short clip, so what arrived was failure
            # cues "out of nowhere, maybe every 30 seconds". If a real key went down while the
            # trigger was held, this press belonged to a shortcut.
            if seconds_since_keydown(lib) <= (now - pressed_at) + _CHORD_EPSILON:
                last_tap = -999.0
                saw("chord")
                continue
            if recording:
                recording, last_tap = False, -999.0
                saw("stop")
                _safe(on_stop)
            elif pressed_at - last_tap <= window:
                # PRESS to PRESS. Comparing releases charged the second tap's own duration to the
                # window and made an ordinary double-tap miss.
                recording, last_tap = True, -999.0
                saw("start")
                _safe(on_start)
            else:
                last_tap = pressed_at
                saw("tap")
        time.sleep(interval)


def _safe(fn: Callable[[], None]) -> None:
    """Run a listener callback, swallowing anything it throws.

    A failed dictation must never take down the listener — the daemon is the one process standing
    between someone and their microphone, so it outlives any single clip.
    """
    with contextlib.suppress(Exception):
        fn()
