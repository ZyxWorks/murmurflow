"""The always-on listener: dictation is live after every login, with nothing to remember.

launchd on macOS, Task Scheduler on Windows. Both live in :mod:`murmurflow.platforms`; this module
is the portable question they answer — *install it, remove it, bounce it, is it up, and who is it
running as* — so nothing above here has to know which one it got.

The environment is the one thing the caller must supply and neither platform can work out: a
relocated ``MURMURFLOW_HOME`` has to survive the login boundary, or the agent starts up reading a
different config than the one the user just edited.
"""

from __future__ import annotations

import os
import sys

from . import platforms

#: The service's name in whichever registry the platform keeps: a launchd label, a task name.
LABEL = "ai.murmurflow.listen"


def supported() -> bool:
    """Can an always-on listener be installed here at all."""
    return sys.platform == "darwin" or sys.platform.startswith("win")


def is_macos() -> bool:
    return sys.platform == "darwin"


def _env() -> dict[str, str]:
    home = os.environ.get("MURMURFLOW_HOME", "")
    return {"MURMURFLOW_HOME": home} if home else {}


def install() -> tuple[bool, str]:
    """Register the listener to start at login and start it now. ``(ok, detail)``."""
    if not supported():
        return False, f"no always-on listener for {sys.platform} yet"
    return platforms.service_install(["listen"], _env())


def uninstall() -> tuple[bool, str]:
    """Remove it. Succeeds when the end state is "not installed"."""
    if not supported():
        return False, f"no always-on listener for {sys.platform} yet"
    return platforms.service_uninstall()


def restart() -> bool:
    """Bounce it so it re-reads its config. False if it was not running.

    The listener reads ``trigger`` and ``doubleTap`` ONCE, when it binds, and then blocks in the
    poll loop forever — so ``config set doubleTap true`` changed the file and nothing else, and the
    honest report was "I set it and it still works the old way". Nobody should have to know a
    daemon is involved, let alone restart one.
    """
    return platforms.service_restart() if supported() else False


def running() -> bool:
    return platforms.service_running() if supported() else False


def installed() -> bool:
    """Is there a service definition on disk, whether or not it is up right now."""
    return supported() and platforms.service_path().is_file()


def where() -> str:
    """The service definition's path, for a doctor row."""
    return str(platforms.service_path()) if supported() else ""


def identity() -> str:
    """Who the listener runs AS, when that is a thing the platform cares about.

    macOS names a permission row after the EXECUTABLE that asked, so the listener is installed
    inside its own ``.app`` bundle and this returns its name and path — the row reads MurmurFlow
    rather than ``python3.13``, and the grant stops at this one program instead of reaching every
    tool sharing the interpreter.

    Windows has no such concept, so this is ``""`` and the caller prints the row differently rather
    than dropping it. A check that VANISHES on one platform reads as a check somebody forgot.
    """
    if not is_macos():
        return ""
    from .platforms import macos

    return f"{macos.APP_NAME}  ({macos.app_path()})" if macos.app_path().is_dir() else ""


def permission_trusted() -> bool | None:
    """Has the thing that will actually do the typing been granted permission to.

    ``None`` means "cannot be answered yet", which on macOS is the honest state before the bundle
    exists: the grant belongs to whichever binary asks for it, the daemon is the bundle, and a
    diagnostic that reports the CLI's permission instead is confidently wrong.
    """
    if not is_macos():
        return platforms.input_permitted()
    from .platforms import macos

    return macos.app_accessibility_trusted()


def remove_identity() -> bool:
    """Delete anything :func:`identity` created. Uninstalling must not leave an app behind."""
    if not is_macos():
        return False
    from .platforms import macos

    return macos.remove_app()
