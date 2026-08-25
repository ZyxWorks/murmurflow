"""The OS layer, and the only place in MurmurFlow that knows which operating system this is.

Four things a dictation tool cannot do portably, and they are the whole of this package:

======== ============================== ============================== =========================
seam     macOS                          Windows                        what it answers
======== ============================== ============================== =========================
capture  ffmpeg ``avfoundation``        ffmpeg ``dshow``               which microphone, how
keys     ``CGEventSourceFlagsState``    ``GetAsyncKeyState``           is the trigger down
inject   AppleScript ⌘V                 ``SendInput`` ^V               put the text at the cursor
service  launchd + an ``.app``          Task Scheduler                 start at login
======== ============================== ============================== =========================

Everything else — the gesture state machine, the recorder's lifecycle, the transcript cleanup, the
hallucination traps, the config vocabulary — is plain Python and already portable. That split is
why this package exists and why it is this small: **the port is four files, not a fork.** Writing
Windows support without the seam means every one of those decisions gets a second copy that drifts,
and the drift is invisible until a client's machine is the one that hits it.

**One backend is selected at import, by :data:`sys.platform`, and nothing above this line branches
on the OS again.** A platform with no backend gets :mod:`~murmurflow.platforms.unsupported`, whose
every function fails cleanly and says why — never an ``ImportError`` at start-up, because a tool
that cannot run should still be able to tell you that, and ``murmurflow doctor`` has to keep
working on the machine that most needs it.

ponytail: module dispatch on ``sys.platform``, not a class hierarchy. There is exactly one instance
of each backend and it never changes during a run, so a class would be a namespace with extra
steps. The ceiling is a machine that wants two backends at once (an X11 and a Wayland session in
one process); it does not exist, and if it ever does the fix is a function argument, not a factory.
"""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from . import macos as impl
elif sys.platform.startswith("win"):
    from . import windows as impl
else:  # linux, bsd, anything else — see the backlog's Wayland decision
    from . import unsupported as impl

#: Human name of the running platform, for a message or a doctor row.
NAME: str = impl.NAME

# --- capture ------------------------------------------------------------------------------------

#: ``(ok, reason)`` for whether audio capture can work at all here.
capture_available = impl.capture_available
#: ffmpeg's input flags for one microphone: ``["-f", "avfoundation", "-i", ":1"]``.
capture_args = impl.capture_args
#: Every audio input as ``(id, name)``. ``id`` is opaque and only :func:`capture_args` reads it.
list_inputs = impl.list_inputs
#: The id to record from when the config names no microphone.
default_input = impl.default_input

# --- keys ---------------------------------------------------------------------------------------

#: ``""`` if the trigger key can be polled, else why it cannot.
keys_unavailable = impl.keys_unavailable
#: Trigger names this platform can poll. :mod:`murmurflow.hotkey` validates config against it.
trigger_names = impl.trigger_names
#: Is the named trigger held down right now.
is_down = impl.is_down
#: Seconds since the last key press or mouse click ANYWHERE — the chord guard's whole input.
seconds_since_input = impl.seconds_since_input
#: True if the OS's OWN dictation would fire on the same gesture as this trigger.
dictation_conflict = impl.dictation_conflict

# --- inject -------------------------------------------------------------------------------------

#: ``""`` if synthetic keystrokes will be delivered, else why they are being swallowed.
input_blocked = impl.input_blocked
#: Put text on the clipboard. ``True`` on success.
clipboard_set = impl.clipboard_set
#: Type text into whatever has focus, restoring the clipboard. Never raises.
#: ``(ok, problem, note)`` — the note is a diagnostic for the daemon log (how much of the text
#: survived the copy, and where the keystroke went), never an error anybody is shown.
inject = impl.inject
#: Has this executable been granted whatever permission :func:`inject` needs. ``True`` when the
#: platform has no such concept, because "not required" and "granted" are the same answer to a
#: caller deciding whether to warn.
input_permitted = impl.input_permitted
#: One line telling the user how to grant it, or ``""`` when there is nothing to grant.
permission_hint = impl.permission_hint

# --- sound --------------------------------------------------------------------------------------

#: Play a wav, without blocking and without ever raising. A missing player is silence, not a crash.
play = impl.play
#: ``{kind: path}`` of the OS's own alert sounds, or ``{}`` where there are none to borrow.
system_cues = impl.system_cues

# --- service ------------------------------------------------------------------------------------

#: Install the always-on listener so it starts at login. ``(ok, detail)``.
service_install = impl.service_install
#: Remove it. ``(ok, detail)`` — succeeds when the end state is "not installed".
service_uninstall = impl.service_uninstall
#: Stop and start it. ``True`` if it came back.
service_restart = impl.service_restart
#: Is it loaded right now.
service_running = impl.service_running
#: Where its definition lives on disk, for a doctor row. May not exist.
service_path = impl.service_path
