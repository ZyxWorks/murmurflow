"""gesture — what a hand does with one key, and what it means. ONE COPY, TWO TOOLS.

**This file is byte-identical in MurmurFlow and in zyx, and that is enforced** — same rule and same
ritual as :mod:`core.speech`, which holds the other half (`make voice-sync`, a digest in both
repos, a test that fails the moment either copy is edited alone).

It is here rather than in `speech` because it is not about audio at all. It is the gesture: the
intent delay, the chord abort, the press-to-press double-tap window, the tap/hold distinction, and
the rule that a callback which raises never kills the loop. Every one of those is a decision made
against a person's hand, tuned by watching one, and every one of them was tuned twice — the
press-to-press window was measured in MurmurFlow and carried into zyx by hand three weeks later,
which is exactly the drift this file ends.

**What is NOT here, and must not be:** which key, what it is called, how this platform reads it,
and what the gesture is FOR. The two primitives are passed in — ``held()`` answers "is the trigger
down right now" and ``since_keydown()`` "how long since any real key went down" — so this file
knows nothing about CoreGraphics, dshow, virtual keycodes or trigger names, and neither tool has to
explain its own vocabulary to it.

Stdlib only, like everything it may be copied into — and it must satisfy the STRICTER of the two
repos' linters, which is MurmurFlow's (it runs the `S` and `SIM` rule sets that zyx does not). A
shared file that only passes in one repo is a file somebody edits in the other and cannot commit.

Licence: MIT, as MurmurFlow is. The copy in zyx is vendored under it — see
docs/legal/THIRD-PARTY.md.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

# DEAD TIME IS LOST WORDS. The recorder now starts on key-DOWN, immediately, and the chord guard
# DISCARDS the clip if a shortcut turns out to be what was happening. The reverse — wait, then start
# — is what shipped first, and it cost the operator the first half-second of every sentence:
# 180ms of waiting PLUS the ~300ms CoreAudio needs to open the mic, ~480ms before a single sample
# lands. His transcripts came back truncated mid-thought ("Hold it, go in, give me this test of the
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
# TAP-WINDOW-1 (operator, 2026-08-27): 0.35 -> 0.50. "The double command click just doesn't work
# sometimes, or works like after 10 seconds. You can't just open it and close it."
#
# 350ms is tighter than the gesture it is measuring. macOS's own double-CLICK default is 500ms and
# its accessibility slider goes far higher, so a hand tapping ⌘⌘ at an ordinary pace was landing
# OUTSIDE this window a good share of the time — and a missed pair is not a no-op: the second tap
# becomes the FIRST tap of a new pair, so the next single tap opens the panel. That is exactly the
# "it fires later, seemingly at random" he is describing, and it is a measurement error rather than
# a race.
#
# Widening it also widens the chord false-positive this comment is about — and that is fine here,
# because the chord GUARD is independent of this number: `listen_double_tap` discards any press
# with a real keystroke inside it, whatever the window says. This value only decides how patient
# the pairing is; `seconds_since_keydown` decides whether the pair was a gesture at all.
DOUBLE_TAP_WINDOW = 0.50

# A tap is a press SHORTER than this. Longer and it is a hold, not a tap — which is what lets both
# modes coexist on one key rather than needing two. Generous on purpose: a tap that lingers is the
# common human miss, and reading it as a hold silently breaks the pair. (0.35 until 2026-09-09,
# raised to MurmurFlow's own number with the press-to-press pairing below — the two were measured
# together on a real hand and neither works as well alone.)
TAP_MAX = 0.50

# Virtual keycode of the LAST key macOS saw go down, used only to notice that the operator pressed
# a real key while holding the trigger — i.e. he is typing ⌘S, not dictating. Any such chord aborts
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


def listen(
    on_press: Callable[[], None],
    on_release: Callable[[], None],
    *,
    held: Callable[[], bool],
    since_keydown: Callable[[], float],
    min_hold: float = 0.15,
    should_stop: Callable[[], bool] | None = None,
    on_abort: Callable[[], None] | None = None,
    chord_grace: float = CHORD_GRACE,
    poll_hz: int = POLL_HZ,
) -> None:
    """Block, calling ``on_press`` when the trigger goes down and ``on_release`` when it comes up.

    ``on_press`` fires IMMEDIATELY on key-down, so the microphone starts opening while the operator
    is still deciding to talk — see :data:`CHORD_GRACE` for why waiting first cost him the opening
    words of every sentence.

    **Chord abort.** Holding ⌃ and pressing C is "interrupt", not "dictate". If a real key goes down
    while the trigger is held, the recording is abandoned via ``on_abort`` (which must DISCARD, not
    transcribe) and no release fires for that press. Without this guard, binding a bare modifier
    would dictate on every keyboard shortcut the operator uses. ``chord_grace`` bounds how long
    after the press a key-down still counts as a chord; past it, a keystroke is someone typing in
    another window, not this gesture.

    ``min_hold`` swallows accidental brushes: a press shorter than this still fires the pair, and
    the caller decides what a too-short clip means — suppressing it here would strand the recording
    that ``on_press`` already started.

    ``should_stop`` is polled each tick so a daemon can shut down cleanly. A callback that raises is
    never allowed to kill the loop: the listener is the one process standing between a person
    and their microphone, so it keeps going rather than dying silently at 3am.
    """
    interval = 1.0 / max(1, poll_hz)
    down = False  # the trigger is physically down and on_press has fired
    aborted = False
    pressed_at = 0.0
    while True:
        if should_stop is not None and should_stop():
            if down and not aborted:  # never leave a recording running on shutdown
                _safe(on_release)
            return
        now_held = held()
        elapsed = time.monotonic() - pressed_at
        # A key-down MORE RECENT than the trigger press is a shortcut, not speech. Bounded by
        # chord_grace so that typing in another window a minute into a long dictation cannot
        # retroactively cancel it.
        chord = (
            _ABORT_ON_CHORD
            and down
            and not aborted
            and elapsed <= chord_grace
            and since_keydown() < elapsed
        )
        if now_held and not down:
            down, aborted = True, False
            pressed_at = time.monotonic()
            _safe(on_press)  # start the mic NOW; the guard below discards if this was a chord
        elif chord:
            aborted = True
            _safe(on_abort or on_release)
        elif not now_held and down:
            down = False
            if aborted:
                aborted = False
                continue  # a shortcut: already discarded, there is nothing to finish
            if elapsed < min_hold:
                # The REMAINDER, not the floor again: a press of `min_hold - 1ms` used to wait
                # a further full floor, so the shortest holds took nearly twice as long to answer.
                # Measured in MurmurFlow; zyx carried the doubled wait until the loop became one.
                time.sleep(min_hold - elapsed)  # let the mic collect something before we cut it
            _safe(on_release)
        time.sleep(interval)


def listen_double_tap(
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
    *,
    held: Callable[[], bool],
    since_keydown: Callable[[], float],
    should_stop: Callable[[], bool] | None = None,
    window: float = DOUBLE_TAP_WINDOW,
    tap_max: float = TAP_MAX,
    poll_hz: int = POLL_HZ,
    on_tap: Callable[[str], None] | None = None,
    pairs_only: bool = False,
    is_recording: Callable[[], bool] | None = None,
) -> None:
    """Hands-free mode: double-tap the trigger to start talking, tap once to stop.

    Why this exists beside :func:`listen`: on Control — the key the operator actually wants, because
    it is where macOS puts dictation — every ``⌃C``/``⌃D``/``⌃R`` looks like the beginning of a
    hold. Hold mode handles that correctly (the chord guard discards the clip) but the operator
    still SEES the machine react to a keystroke that was never meant for it. A double-tap is a
    deliberate gesture, so the interaction stays silent until he genuinely asks for it.

    ``is_recording`` (optional) is how the loop learns that a clip ended WITHOUT a tap. The
    ``recording`` flag below used to be the only record of whether anything was running, and the
    microphone can now close itself (:data:`core.dictate.SILENCE_STOP_SECONDS`). The clip was gone
    and this still believed it was running, so the next tap was spent being a STOP for a clip that
    had already stopped, and the double-tap only worked on the try after that. Asked once per poll,
    so the answer must be a cheap lookup and never work.

    ``on_tap`` (optional) is told ``"press"`` the instant the key goes down — before anything is
    known about what the gesture will turn out to be, which is what lets a caller open the
    microphone early — and then about every press this loop DECIDES something about — ``"tap"``,
    ``"hold"``, ``"chord"``, ``"start"``, ``"stop"``. It exists because the alternative failure is
    unanswerable: a listener that reacts to nothing looks identical whether the key is never read,
    the taps are too far apart, or the chord guard is eating them. The operator hit exactly that on
    a newly-bound key ("I can't hear any sound when I double click the right command"), and there
    was nothing anywhere to tell him which of the three it was.

    **One chord is not a start — but two in a row were.** This originally reasoned that "a shortcut
    is one press, and one press is never a start", and applied no chord guard at all. ⌃C then ⌃C in
    a terminal is two short Control presses inside :data:`DOUBLE_TAP_WINDOW`, so recording began
    behind the operator's back and the next stray tap ended it as a too-short clip — failure cues
    arriving "out of nowhere" while he worked. A press with a real keystroke inside it is now
    discarded as the shortcut it was.

    **``pairs_only`` — for a trigger that TOGGLES something rather than recording.** The floating
    chat pane is one gesture in and the same gesture out, and it was bound to this loop as if it
    were a microphone: the double-tap opened it (``on_start``), and the next single tap was read as
    the "stop" of a recording — a callback that does nothing for a pane, while it still CONSUMED
    the tap and left ``recording`` true with nothing recording. The count the operator measured is
    exactly that arithmetic: two taps open it, one dead tap, two more to fire again (PANE-TAP-1,
    operator, 2026-08-28: "it only opens after clicking four times the command and only closes
    after clicking four times the command").

    With it set there is no recording state at all: every completed pair fires ``on_start`` and a
    lone tap is only ever the first half of the next pair. ``on_stop`` is never called.
    """
    interval = 1.0 / max(1, poll_hz)
    down = False
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
        if recording and is_recording is not None and not is_recording():
            # It closed itself. Forget the clip rather than charging the next tap for it.
            recording, last_tap = False, -999.0
            saw("ended")
        now = time.monotonic()
        now_held = held()
        if now_held and not down:
            down, pressed_at = True, now
            saw("press")
        elif not now_held and down:
            down = False
            if now - pressed_at > tap_max:
                last_tap = -999.0  # a long hold is not a tap; it cannot open a double-tap pair
                saw("hold")
                continue
            # A CHORD IS NOT A TAP. This module used to claim a chord "can NEVER look like a
            # double-tap" — false, and the operator found it: ⌃C then ⌃C in a terminal is two short
            # Control presses inside the window, which is exactly the gesture. Recording started
            # behind his back and the next stray tap ended it as a too-short clip, so he got failure
            # cues "out of nowhere, maybe every 30 seconds". If a real key went down while the
            # trigger was held, this press belonged to a shortcut.
            if since_keydown() <= (now - pressed_at) + _CHORD_EPSILON:
                last_tap = -999.0
                saw("chord")
                continue
            if recording and not pairs_only:
                recording, last_tap = False, -999.0
                saw("stop")
                _safe(on_stop)
            elif pressed_at - last_tap <= window:
                # PRESS TO PRESS, never release to release. Comparing releases charges the second
                # tap's own duration to the window, so a 200ms press blew a 500ms window and an
                # ordinary double-tap read as two unrelated taps — nothing happened at all. It is
                # also how macOS measures its own double-click.
                recording, last_tap = not pairs_only, -999.0
                saw("start")
                _safe(on_start)
            else:
                last_tap = pressed_at
                saw("tap")
        time.sleep(interval)


def _safe(fn: Callable[[], None]) -> None:
    """Run a listener callback, swallowing anything it throws."""
    with contextlib.suppress(Exception):
        fn()
