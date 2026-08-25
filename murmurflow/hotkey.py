"""Press-to-talk key detection and the GESTURE it makes, in stdlib and nothing else.

Everything here is portable. The two facts that are not — "is this key down right now" and "how
long since the user did something else" — come from :mod:`murmurflow.platforms`, which is the only
module that knows which operating system this is.

That split is the whole reason a Windows port is four files and not a fork. The gesture is the hard
part and it is written once: the intent delay, the chord abort, the press-to-press double-tap
window, the tap/hold distinction, and the rule that a callback which raises never kills the loop.
Every one of those is a decision made against a person's hand, not against an API.

**A poll loop, not an event tap**, and that is the design rather than a shortcut. Polling needs no
special grant, no run loop and no callback glue on either platform, and nothing is compiled — so a
client's machine needs no toolchain and no signing. The cost is honest: a poll cannot *consume* the
keystroke, so the trigger must be a key the frontmost app will not miss. That is what the two
guards below are for.

Right *Option* is deliberately never the default, and that is not a preference: on the German
layout right Option is AltGr, the dead key for ``@ € \\ | ~ [ ] { }``. Anyone typing German all day
would fire the microphone on virtually every email address and code bracket. This is true on both
platforms, which is exactly why the trigger vocabulary is shared and not per-OS.

ponytail: a 60 Hz poll of two cheap calls, not an event tap. The ceiling is that we observe rather
than intercept the key; upgrade only if a trigger key that must be swallowed is ever required.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from . import platforms

#: COMBINATION triggers: every one of these flags must be held at once, and nothing types.
#:
#: This is the answer to the real cost of polling. A poll cannot CONSUME a key, so a single bare
#: modifier is shared with everything the user already does with it: ⌃C, ⌃←, ⌃-click. The chord
#: guard discards the audio, but the guard only has :data:`CHORD_GRACE` to notice, and a person who
#: holds Control and *then* reaches for the arrow key beats it — so the microphone opens and the
#: cue plays for a keystroke that was never dictation. Two modifiers together are bound to nothing
#: in macOS, are typed by nobody in the course of ordinary work, and need no setting turned off
#: first — unlike Fn (the emoji picker) or double-tap Control (Apple's own dictation).
COMBO_TRIGGERS: tuple[str, ...] = (
    "control_option",
    "control_command",
    "command_option",
    "control_shift",
)


def trigger_names() -> frozenset[str]:
    """Every trigger this machine can actually poll — the platform's own answer, never a guess.

    macOS and Windows share the vocabulary but not the whole set: ``fn`` exists only on the Mac
    (Windows keyboards handle it in firmware and it never reaches the OS), and ``right_command``
    only works on Windows (the MacBooks report the right-side virtual keycodes as the left ones,
    so a trigger bound to one there can never fire). A config naming a key this platform cannot
    read is a trigger that silently does nothing, which is why the set is asked for rather than
    assumed.
    """
    return platforms.trigger_names()


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
#:
#: Substituted WORD BY WORD, not whole-name. A hand-written list of whole names is the same list
#: twice over — every alias times every side and every combo it appears in — and it shipped with
#: holes: `left_ctrl`, `ctrl_win`, `win_alt` and `ctrl_shift` were all rejected while their
#: canonical twins worked, against a README that promises the two spellings are interchangeable.
#: No canonical trigger name contains any of these words, so the substitution cannot collide.
TRIGGER_ALIASES: dict[str, str] = {
    "ctrl": "control",
    "alt": "option",
    "win": "command",
    "super": "command",
    "meta": "command",
}


def canonical_trigger(name: str) -> str:
    """One spelling per key, so `ctrl_alt` and `control_option` cannot behave differently."""
    key = str(name).strip().lower().replace("-", "_").replace("+", "_")
    return "_".join(TRIGGER_ALIASES.get(word, word) for word in key.split("_"))


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
# easier on the hand than holding — which is why this, and not hold, is the default. Hold is the
# opt-out (`config set doubleTap false`), and it is faster for one short sentence: you are already
# holding the key you pressed.
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
# core. Polling faster buys nothing a person can feel. (Windows scans a few dozen keys per tick
# for the chord guard instead of one call; still well under a percent.)
POLL_HZ = 60


def accessibility_trusted() -> bool:
    """Has this binary been granted whatever permission typing into another app needs.

    On macOS that is the Accessibility TCC grant, and it is the one permission dictation genuinely
    cannot avoid. On Windows there is no such concept, so the answer is ``True`` — "not required"
    and "granted" are the same answer to a caller deciding whether to warn somebody, and a
    diagnostic that says BLOCKED on a machine with nothing to block is worse than no diagnostic.
    """
    return platforms.input_permitted()


def seconds_since_keydown() -> float:
    """Seconds since the last real key-down OR mouse click anywhere on the system.

    The whole chord guard: if this is smaller than how long the trigger has been held, the user
    acted WHILE holding it — they are typing ⌘S or ⌘-clicking a link, not dictating. Clicks count
    as well as keys, because on a Command trigger the mouse is half of the shortcuts.
    """
    return platforms.seconds_since_input()


def is_trigger_down(name: str) -> bool:
    """Is the trigger called ``name`` down right now.

    ONE seam, so a listener never has to know whether the platform answered from a modifier flag,
    a side-specific keycode or a pair of them held together.
    """
    return platforms.is_down(canonical_trigger(name))


def available() -> bool:
    """True if global key state can be read on this machine."""
    return not platforms.keys_unavailable()


def unavailable_reason() -> str:
    """Why it cannot be, or ``""``. A listener that reads no keys must be able to say why."""
    return platforms.keys_unavailable()


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
    interval = 1.0 / max(1, poll_hz)
    held = False  # the trigger is physically down and on_press has fired
    aborted = False
    pressed_at = 0.0
    while True:
        if should_stop is not None and should_stop():
            if held and not aborted:  # never leave a recording running on shutdown
                _safe(on_release)
            return
        now_held = is_trigger_down(trigger)
        elapsed = time.monotonic() - pressed_at
        # A key-down MORE RECENT than the trigger press is a shortcut, not speech. Bounded by
        # chord_grace so that typing in another window a minute into a long dictation cannot
        # retroactively cancel it.
        chord = (
            _ABORT_ON_CHORD
            and held
            and not aborted
            and elapsed <= chord_grace
            and seconds_since_keydown() < elapsed
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
                # The REMAINDER, not the floor again: a press of `min_hold - 1ms` used to wait a
                # further full floor, so the shortest holds took nearly twice as long to answer.
                time.sleep(min_hold - elapsed)  # let the mic collect something before we cut it
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
        now_held = is_trigger_down(trigger)
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
            if seconds_since_keydown() <= (now - pressed_at) + _CHORD_EPSILON:
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
