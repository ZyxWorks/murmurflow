"""The backend for a platform MurmurFlow has not been ported to. Everything fails, politely.

This exists so that ``import murmurflow`` succeeds everywhere. The alternative — raising at import
on an unknown ``sys.platform`` — breaks the one command that matters on an unsupported machine:
``murmurflow doctor``, which is how somebody finds out *why* it will not run. A tool that cannot
work should still be able to say so.

Linux lives here for now, and that is a decision rather than an omission. Capture and injection are
both small (ffmpeg ``pulse``; ``xdotool``), but the key polling is not: X11 has ``XQueryKeymap``
and **Wayland has no global hotkey API at all**. Shipping X11-only, requesting a desktop portal, or
not shipping Linux are three different products, and that call is the operator's.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAME = f"{sys.platform} (unsupported)"

_WHY = (
    f"MurmurFlow has no backend for {sys.platform} yet — macOS and Windows are supported. "
    "On Linux the open question is the hotkey: Wayland exposes no global hotkey API."
)


def capture_available() -> tuple[bool, str]:
    return False, _WHY


def capture_args(device: str) -> list[str]:
    return []


def list_inputs() -> list[tuple[str, str]]:
    return []


def default_input() -> str:
    return ""


def keys_unavailable() -> str:
    return _WHY


def trigger_names() -> frozenset[str]:
    return frozenset()


def is_down(name: str) -> bool:
    return False


def seconds_since_input() -> float:
    # Not "0 seconds ago", which the chord guard reads as "the user just typed something" and would
    # abort every recording. There is no input to be near, so the honest answer is "forever ago".
    return float("inf")


def dictation_conflict(trigger: str) -> bool:
    return False


def input_blocked() -> str:
    return _WHY


def clipboard_set(text: str) -> bool:
    return False


def inject(text: str, settle: float) -> tuple[bool, str, str]:
    # `settle` was missing here while every caller passed it, so on Linux the one honest message
    # this backend exists to give ("MurmurFlow has no backend for this platform") was a TypeError.
    return False, _WHY, ""


def input_permitted() -> bool:
    return False


def permission_hint() -> str:
    return _WHY


def type_text(text: str) -> str:
    return text


def play_ready() -> None:
    return None


def service_install(args: list[str], env: dict[str, str]) -> tuple[bool, str]:
    return False, _WHY


def service_uninstall() -> tuple[bool, str]:
    return False, _WHY


def service_restart() -> bool:
    return False


def service_running() -> bool:
    return False


def service_path() -> Path:
    return Path("/nonexistent")
