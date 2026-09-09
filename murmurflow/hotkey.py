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

from collections.abc import Callable

from . import gesture, platforms

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


# THE GESTURE ITSELF IS SHARED — `murmurflow.gesture`, byte-identical with zyx's copy of it (see
# that module's own note). What stays HERE is the vocabulary: which triggers exist, what they are
# called on each keyboard, and which platform module answers for them. The loop is handed the two
# questions it needs as callables and never learns what a keycode is.
CHORD_GRACE = gesture.CHORD_GRACE
DOUBLE_TAP_WINDOW = gesture.DOUBLE_TAP_WINDOW
TAP_MAX = gesture.TAP_MAX
POLL_HZ = gesture.POLL_HZ


def listen(
    on_press: Callable[[], None],
    on_release: Callable[[], None],
    *,
    trigger: str = DEFAULT_TRIGGER,
    on_abort: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    min_hold: float = 0.15,
    chord_grace: float = CHORD_GRACE,
    poll_hz: int = POLL_HZ,
) -> None:
    """Hold-to-talk on ``trigger``. Blocks. See :func:`murmurflow.gesture.listen`."""
    gesture.listen(
        on_press,
        on_release,
        held=lambda: is_trigger_down(trigger),
        since_keydown=seconds_since_keydown,
        on_abort=on_abort,
        should_stop=should_stop,
        min_hold=min_hold,
        chord_grace=chord_grace,
        poll_hz=poll_hz,
    )


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
    is_recording: Callable[[], bool] | None = None,
) -> None:
    """Double-tap to start, tap once to stop. See :func:`murmurflow.gesture.listen_double_tap`."""
    gesture.listen_double_tap(
        on_start,
        on_stop,
        held=lambda: is_trigger_down(trigger),
        since_keydown=seconds_since_keydown,
        should_stop=should_stop,
        window=window,
        tap_max=tap_max,
        poll_hz=poll_hz,
        on_tap=on_tap,
        is_recording=is_recording,
    )
