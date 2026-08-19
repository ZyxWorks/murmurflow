"""The measurements MurmurFlow and zyx must not drift apart on — see voice-contract.json.

MurmurFlow was extracted from zyx's ``core.dictate`` and the two now ship separately, on purpose:
they do different jobs. MurmurFlow types what you say into the app you are in; zyx hands what you
say to its own runtime and answers out loud. Merging them back was measured and rejected, so the
CODE is forked and the MEASUREMENTS are not.

This asserts MurmurFlow's own code still matches the contract; zyx holds an identical copy of the
file and asserts the same of its own. It is deliberately about physics and protocol, never about
product decisions — where the two SHOULD differ (``tidy``, the hallucination list, ``quietFloor``)
the contract says so and nothing here checks it. A test forcing those together would be wrong.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from murmurflow import dictate

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "voice-contract.json").read_text("utf-8")
)


def test_the_contract_is_the_version_this_test_was_written_against() -> None:
    """Bumping it is a deliberate act in BOTH repos — see the file's own ``its_honest_limit``."""
    assert CONTRACT["version"] == 1


def test_every_threshold_still_holds_its_measured_value() -> None:
    for name, row in CONTRACT["thresholds"].items():
        assert getattr(dictate, name) == row["value"], (
            f"{name} drifted from the measurement in voice-contract.json. "
            f"Why it is {row['value']}: {row['why']}"
        )


def test_the_warm_request_still_asks_whisper_the_same_question() -> None:
    """Read off the source, because the alternative is a live whisper-server in the suite."""
    spec = CONTRACT["whisper_warm_request"]
    src = inspect.getsource(dictate.transcribe_warm)
    for field in spec["required_fields"]:
        assert f'"{field}"' in src, f"the warm request dropped `{field}`: {spec['why']}"
    assert f'"{spec["response_format"]}"' in src
    assert f'"temperature": "{spec["temperature"]}"' in src
    assert spec["endpoint"] in src


def test_the_warm_server_is_started_where_it_can_write() -> None:
    """THE BUG THAT SHIPPED IN BOTH COPIES.

    ``--convert`` shells out to ffmpeg, ffmpeg writes into the process CWD, and launchd runs agents
    at ``/`` — so started with no cwd the server answers every request with a 500.
    """
    spec = CONTRACT["whisper_server_flags"]
    # Read the SOURCE, not a call: `serve_command()` returns None without a resolvable binary
    # and model, so calling it would pass vacuously on any machine that has neither.
    src = inspect.getsource(dictate.serve_command)
    for flag in spec["must_contain"]:
        assert f'"{flag}"' in src, f"{flag} missing: {spec['why']}"
    if spec["must_run_with_cwd"]:
        assert "cwd=" in inspect.getsource(dictate.start_server), spec["why"]


def test_capture_still_stays_on_real_time() -> None:
    spec = CONTRACT["ffmpeg_capture"]
    src = inspect.getsource(dictate.start)
    for token in spec["must_contain"]:
        assert f'"{token}"' in src, f"capture dropped `{token}`: {spec['why']}"


def test_a_whisper_segment_break_never_survives_tidy() -> None:
    """The one thing NOT under ``deliberately_divergent`` even though ``tidy`` itself is.

    whisper emits one newline per SEGMENT and segments mid-clause, so a single sentence comes back
    broken across lines. Nobody can speak a newline, so flattening them is not an edit to what was
    said — it undoes a transport artefact, and both tools owe it whatever else their tidy does.
    """
    raw = " It seems like it's way better working here than in the\n other, you know,\n"
    assert "\n" not in dictate.tidy(raw)
