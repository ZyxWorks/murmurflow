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

import hashlib
import inspect
import json
from pathlib import Path

from murmurflow import dictate, speech

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "voice-contract.json").read_text("utf-8")
)


def test_the_contract_is_the_version_this_test_was_written_against() -> None:
    """Bumping it is a deliberate act in BOTH repos — see the file's own ``its_honest_limit``."""
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


def test_a_segment_seam_inside_a_word_never_becomes_a_space() -> None:
    """The seam whisper leaves between two tokens of ONE word must close with nothing.

    Reported 2026-08-20 from a real dictation: "zyxworks.gith ub.io", "z yxworks.com". Both tools
    flatten whisper's per-segment newlines, so both tools own this half of it too.
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

_CORE = Path(__file__).resolve().parents[1] / "murmurflow" / "speech.py"
_DIGEST = Path(__file__).resolve().parents[1] / "voice-core.sha256"


def test_the_shared_speech_core_is_the_copy_both_tools_carry() -> None:
    """ONE COPY, TWO TOOLS — and a copy nothing checks is two copies again in three weeks.

    `voice-contract.json` pins the MEASUREMENTS and it did its job: the thresholds never drifted.
    What drifted was everything around them — a wav header offset, a hallucination table, a
    trailing-silence trim, a level scan — because "the same code in both repos" was a habit, and
    habits lose to three weeks and 24 commits.

    So the shared layer is ONE FILE (`murmurflow/speech.py`) and it is byte-identical in zyx. This
    test cannot see zyx, and does not try to: it checks that the file has not been edited since the
    two were last made equal. Editing it is fine and expected; editing it and leaving the other copy
    behind is what this names. The ritual is `make voice-sync`, run from the zyx checkout.
    """
    digest = hashlib.sha256(_CORE.read_bytes()).hexdigest()
    assert digest == _DIGEST.read_text("utf-8").split()[0], (
        "murmurflow/speech.py changed. It is the SHARED speech core: zyx carries the same file byte "
        "for byte. Run `make voice-sync` in the zyx checkout (it copies the file and rewrites the "
        "digest in both repos), then commit both."
    )


def test_nothing_in_the_shared_core_asks_a_question_about_this_install() -> None:
    """The floors are ARGUMENTS in there, never settings, or the file cannot be the same file.

    The two tools name their settings differently (`quietFloor` against `voiceQuietFloor`) and read
    them from different places, so one line of config in `speech` is a line that has to differ — and
    one line that differs is a file that is no longer shared.
    """
    src = _CORE.read_text("utf-8")
    for forbidden in ("import config", "config.flag", "_cfg(", "quiet_floor()", "os.environ"):
        assert forbidden not in src, (
            f"`{forbidden}` in murmurflow/speech.py: the shared core reads no configuration. "
            "Take the value as an argument and let each tool answer for its own install."
        )


def test_the_polite_one_word_sentences_are_not_in_the_shared_table() -> None:
    """`deliberately_divergent.hallucination_list`, enforced where it can actually be enforced.

    zyx's blocklist holds "thank you", "you", "so" and "bye", and that is right THERE: what reads
    the transcript is a model that can decline to answer. Here the transcript is TYPED, so a real
    sentence swallowed reads as broken hardware — and the shared table is the one place those two
    bets could quietly be merged into one. This is the test that noticed when they were.
    """
    for word in ("thank you", "thank you.", "you", "so", "bye", "vielen dank", "."):
        assert word not in speech.HALLUCINATIONS, (
            f"{word!r} reached the SHARED hallucination table. It is a thing a person says, and "
            "MurmurFlow types what a person says."
        )
