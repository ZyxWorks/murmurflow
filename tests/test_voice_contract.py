"""The measurements MurmurFlow's voice code must not drift from — see voice-contract.json.

The contract is the measured record another voice tool can check itself against. This asserts
MurmurFlow's own code still matches it. It is deliberately about physics and protocol, never about
product decisions — where a consumer MAY differ (``tidy``, the hallucination list, ``quietFloor``)
the contract says so and nothing here checks it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from murmurflow import dictate, speech

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "voice-contract.json").read_text("utf-8")
)


def test_the_contract_is_the_version_this_test_was_written_against() -> None:
    """Bumping it is a deliberate act — see the file's own ``its_honest_limit``."""
    assert CONTRACT["version"] == 2


def test_every_threshold_still_holds_its_measured_value() -> None:
    for name, row in CONTRACT["thresholds"].items():
        assert getattr(dictate, name) == row["value"], (
            f"{name} drifted from the measurement in voice-contract.json. "
            f"Why it is {row['value']}: {row['why']}"
        )


def test_the_warm_request_still_asks_whisper_the_same_question() -> None:
    """Read off the source, because the alternative is a live whisper-server in the suite."""
    spec = CONTRACT["whisper_warm_request"]
    src = inspect.getsource(speech.transcribe_warm)
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
    src = inspect.getsource(speech.serve_command)
    for flag in spec["must_contain"]:
        assert f'"{flag}"' in src, f"{flag} missing: {spec['why']}"
    if spec["must_run_with_cwd"]:
        assert "cwd=" in inspect.getsource(speech.start_server), spec["why"]


def test_capture_still_stays_on_real_time() -> None:
    spec = CONTRACT["ffmpeg_capture"]
    src = inspect.getsource(speech.start)
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


def test_a_segment_seam_inside_a_word_never_becomes_a_space() -> None:
    """The seam whisper leaves between two tokens of ONE word must close with nothing.

    Reported 2026-08-20 from a real dictation: a domain typed as "gith ub.io". Every tool that
    flattens whisper's per-segment newlines owns this half of it too.
    """
    spec = CONTRACT["transcript_seams"]
    for raw, expected in spec["must_hold"]:
        assert expected in dictate.tidy(raw), spec["why"]


def test_boilerplate_appended_to_a_real_sentence_never_survives_tidy() -> None:
    """The report: it adds a thank you at the end. The whole-line trap cannot see this one."""
    spec = CONTRACT["trailing_hallucination"]
    for raw, expected in spec["must_hold"]:
        assert dictate.tidy(raw) == expected, spec["why"]


# --- the core itself, not just the measurements -------------------------------------------------

#: The files another tool may vendor verbatim, each with a digest in voice-core.sha256.
SHARED = ("speech.py", "gesture.py")
_HERE = Path(__file__).resolve().parents[1]


def _core(name: str) -> Path:
    return _HERE / "murmurflow" / name


def _recorded(name: str) -> str:
    for line in (_HERE / "voice-core.sha256").read_text("utf-8").splitlines():
        if line.strip().endswith(name):
            return line.split()[0]
    raise AssertionError(f"{name} has no digest in voice-core.sha256")


@pytest.mark.parametrize("name", SHARED)
def test_the_vendorable_files_match_their_recorded_digest(name: str) -> None:
    """An edit to a file other tools may vendor is a deliberate act, and the digest says it moved.

    `voice-contract.json` pins the MEASUREMENTS and it did its job: the thresholds never drifted.
    What drifted, back when this layer lived in two hand-kept copies, was everything around them —
    a wav header offset, a hallucination table, a trailing-silence trim, a level scan, and a hold
    floor that waited its remainder in one copy and the whole floor again in the other.

    So the vendorable layer is TWO FILES (`speech.py`, the audio and the transcript; `gesture.py`,
    what a hand does with one key). A consumer compares its copy against `voice-core.sha256`; this
    test makes sure that file is never stale.
    """
    digest = hashlib.sha256(_core(name).read_bytes()).hexdigest()
    assert digest == _recorded(name), (
        f"murmurflow/{name} changed. Other tools may vendor it, so record the new digest: "
        "`(cd murmurflow && shasum -a 256 speech.py gesture.py) > voice-core.sha256`, and commit it."
    )


def test_nothing_in_the_shared_core_asks_a_question_about_this_install() -> None:
    """The floors are ARGUMENTS in there, never settings, or the file cannot be vendored unchanged.

    Every tool names its settings differently and reads them from a different place, so one line of
    config in `speech` is a line a consumer has to change — and a copy that differs is no longer a
    copy anybody can check against the digest.
    """
    for name in SHARED:
        src = _core(name).read_text("utf-8")
        for forbidden in ("import config", "config.flag", "_cfg(", "quiet_floor()", "os.environ"):
            assert forbidden not in src, (
                f"`{forbidden}` in murmurflow/{name}: a shared file reads no configuration. "
                "Take the value as an argument and let each tool answer for its own install."
            )


def test_the_polite_one_word_sentences_are_not_in_the_shared_table() -> None:
    """`deliberately_divergent.hallucination_list`, enforced where it can actually be enforced.

    A tool whose reader is a model that can decline to answer may block "thank you", "you", "so"
    and "bye" in its own list. Here the transcript is TYPED, so a real sentence swallowed reads as
    broken hardware — and the shared table is the one place those two bets could quietly be merged
    into one. This is the test that noticed when they were.
    """
    for word in ("thank you", "thank you.", "you", "so", "bye", "vielen dank", "."):
        assert word not in speech.HALLUCINATIONS, (
            f"{word!r} reached the SHARED hallucination table. It is a thing a person says, and "
            "MurmurFlow types what a person says."
        )
