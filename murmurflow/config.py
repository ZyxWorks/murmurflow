"""``~/.murmurflow/config.json`` — the whole settings surface, and it is one flat JSON object.

There is no settings UI and there is deliberately no schema: every key is optional, an unknown key
is ignored, and a corrupt file degrades to ``{}`` rather than wedging the hotkey daemon. The
daemon reads this on every key press, so it has to be cheap and it has to never raise.

Set ``MURMURFLOW_HOME`` to move the whole directory (config, models, scratch audio) — that is what
the test suite does, and it is the only isolation the tests need.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HOME_ENV = "MURMURFLOW_HOME"

# Recognized keys, and what they do. Kept here rather than in a schema because the only consumer
# that needs the list is `murmurflow config` printing it.
KEYS: dict[str, str] = {
    "trigger": "hold-to-talk key. Default control_option (two modifiers, so no shortcut can fire it). Also: command_option, control_command, control_shift, left_control, right_shift, f13, command, option",
    "doubleTap": "true = tap the trigger twice to start, twice to stop (instead of holding)",
    "model": "path to a ggml model file, overriding the best one found in ~/.murmurflow/models/",
    "language": "the spoken language, e.g. en / de. Default auto, which costs ~0.7s per clip",
    "inputName": "substring of the microphone name to record from. Default: system default",
    "vocabulary": "list of proper nouns to bias the transcriber toward (names, jargon, acronyms)",
    "cue": "tone preset: glass (default), soft, pure, wood, bell, or off for silence",
    "polishCommand": "shell command receiving the transcript on stdin and printing the cleaned text",
    "keepAudio": "true = keep the last clip at ~/.murmurflow/audio/last.wav for debugging",
    "port": "loopback port for the warm whisper-server. Default 8479",
}


def home_root() -> Path:
    """``~/.murmurflow`` (or ``$MURMURFLOW_HOME``), created on demand.

    Test the env var as a STRING, not as the Path it becomes: ``Path("")`` is ``Path(".")``, which
    is truthy, so an unset variable would silently resolve the whole home — config, models, scratch
    audio — to whatever directory the process happened to start in.
    """
    override = os.environ.get(HOME_ENV, "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".murmurflow"
    root.mkdir(parents=True, exist_ok=True)
    return root


def config_path() -> Path:
    return home_root() / "config.json"


def load() -> dict[str, Any]:
    """The config as a dict. ``{}`` on anything unreadable — never raises, by contract."""
    try:
        data = json.loads(config_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_TRUE = frozenset({"true", "yes", "on", "1"})


def flag(key: str, default: bool = False, *, cfg: dict[str, Any] | None = None) -> bool:
    """A boolean setting, tolerating the string forms a hand-edited JSON file ends up holding."""
    value = (load() if cfg is None else cfg).get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def set_value(key: str, value: Any) -> None:
    """Write one key. Read-modify-write of a small file the daemon may be reading concurrently, so
    it lands via a temp file + atomic rename — a half-written config is a daemon that stops.
    """
    data = load()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    path = config_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(path)


def coerce(raw: str) -> Any:
    """Turn a CLI string into the JSON value it obviously means, so ``config set doubleTap true``
    does not store the string ``"true"``. Falls back to the string itself.
    """
    text = raw.strip()
    if text.lower() in _TRUE:
        return True
    if text.lower() in {"false", "no", "off", "0"}:
        return False
    try:
        return json.loads(text)  # numbers, lists, objects
    except ValueError:
        return text
