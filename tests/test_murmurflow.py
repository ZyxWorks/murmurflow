"""Hermetic tests: no microphone, no whisper, no network, no launchd.

Everything that can be tested without hardware is tested here, and that turns out to be most of the
logic that actually goes wrong — the hallucination traps, the punctuation repair, the filler strip,
the config coercion and the polish shell-out. The parts that genuinely need a Mac with a microphone
(``start``/``finish``/``inject``) are exercised by ``murmurflow doctor`` and ``keytest`` instead,
because a mock of CoreAudio would only ever test the mock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murmurflow import config, dictate, whisper


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.murmurflow. This is the whole isolation the suite needs."""
    monkeypatch.setenv(config.HOME_ENV, str(tmp_path))
    monkeypatch.setenv("MURMURFLOW_NO_AUDIO", "1")  # never make a sound in CI
    return tmp_path


# --- config ------------------------------------------------------------------------------------


def test_home_root_is_absolute_when_the_env_var_is_unset(monkeypatch, tmp_path):
    # `Path("")` is `Path(".")`, which is TRUTHY — so a naive `or` puts the entire home (config,
    # models, recorded audio) in whatever directory the process started in. Found by running
    # `murmurflow doctor` from a checkout and seeing a relative config path come back.
    monkeypatch.delenv(config.HOME_ENV, raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    root = config.home_root()
    assert root.is_absolute()
    assert root == tmp_path / ".murmurflow"


def test_missing_config_is_empty_not_an_error():
    assert config.load() == {}


def test_corrupt_config_degrades_rather_than_raising():
    config.config_path().write_text("{not json at all", "utf-8")
    assert config.load() == {}
    assert dictate.port() == dictate.DEFAULT_PORT  # the daemon keeps working


def test_set_value_roundtrips_and_unsets():
    config.set_value("trigger", "right_command")
    assert config.load()["trigger"] == "right_command"
    config.set_value("trigger", None)
    assert "trigger" not in config.load()


def test_coerce_reads_json_shapes_not_strings():
    assert config.coerce("true") is True
    assert config.coerce("off") is False
    assert config.coerce("8477") == 8477
    assert config.coerce('["a", "b"]') == ["a", "b"]
    assert config.coerce("right_option") == "right_option"


def test_flag_tolerates_hand_edited_strings():
    config.config_path().write_text(json.dumps({"doubleTap": "yes"}), "utf-8")
    assert dictate.double_tap_mode() is True


# --- the hallucination traps -------------------------------------------------------------------
# These are the highest-stakes logic in the tool: a false negative types words into your document
# that you never said.


@pytest.mark.parametrize(
    "text",
    [
        "Thank you.",
        "thanks for watching!",
        "[BLANK_AUDIO]",
        "Vielen Dank.",
        "you",
        "*sad*",
        "[MUSIC]",
        "(wind blowing)",
        "♪♪♪",
    ],
)
def test_whisper_silence_boilerplate_is_never_typed(text):
    assert dictate.is_hallucination(text)


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for sending the invoice over.",
        "So I think we should ship it on Friday.",
        # A fully parenthesised REAL sentence must survive: silently swallowing what someone said is
        # a worse bug than letting one hallucination through, because it looks like the mic failed.
        "(That said, we should ship it anyway.)",
    ],
)
def test_real_speech_survives_the_traps(text):
    assert not dictate.is_hallucination(text)


# --- cleanup -----------------------------------------------------------------------------------


def test_fillers_are_stripped_without_rewriting():
    assert dictate.strip_fillers("So, um, ship it on Friday") == "ship it on Friday"


def test_a_clean_transcript_passes_through_byte_identical():
    clean = "Ship the invoice on Friday."
    assert dictate.tidy(clean) == clean


def test_strip_never_empties_a_line_that_was_all_filler():
    # It removes what it can and keeps whatever survives ...
    assert dictate.strip_fillers("so, um, yeah") == "yeah"
    # ... but a line with nothing left after the strip falls back to the original rather than "".
    assert dictate.strip_fillers("um uh") == "um uh"


def test_punctuation_left_dangling_by_the_strip_is_repaired():
    # "...fixed, um, you know." -> the strip leaves "fixed," hanging at the end of the line.
    assert dictate.tidy("It is fixed, um, you know.") == "It is fixed."


def test_space_before_punctuation_is_closed_up():
    assert dictate.tidy("Ship it , then tell me .") == "Ship it, then tell me."


# --- polish ------------------------------------------------------------------------------------


def test_polish_is_a_no_op_when_unconfigured():
    assert dictate.polish("keep this exactly") == "keep this exactly"


def test_polish_runs_the_configured_command_over_stdin():
    config.set_value("polishCommand", "tr '[:lower:]' '[:upper:]'")
    out = dictate.polish("hello there")
    assert "HELLO THERE" in out  # the prompt is prepended, so match rather than compare


def test_polish_sends_the_bare_transcript_when_the_command_owns_the_prompt():
    config.set_value("polishCommand", "cat  # {prompt}")
    assert dictate.polish("just this") == "just this"


def test_a_broken_polish_command_never_costs_the_transcript():
    config.set_value("polishCommand", "exit 1")
    assert dictate.polish("my words") == "my words"
    config.set_value("polishCommand", "definitely-not-a-real-binary-xyz")
    assert dictate.polish("my words") == "my words"


def test_polish_that_prints_nothing_falls_back_rather_than_erasing():
    config.set_value("polishCommand", "true")
    assert dictate.polish("my words") == "my words"


# --- the transcriber's decisions ---------------------------------------------------------------


def test_model_prefers_the_best_one_present_not_the_first_found():
    models = config.home_root() / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "ggml-base.bin").write_bytes(b"x")
    assert Path(whisper.model()).name == "ggml-base.bin"
    (models / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    assert Path(whisper.model()).name == "ggml-large-v3-turbo.bin"


def test_an_explicit_model_path_wins_over_the_preference_order():
    models = config.home_root() / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    chosen = models / "ggml-small.bin"
    chosen.write_bytes(b"x")
    config.set_value("model", str(chosen))
    assert whisper.model() == str(chosen)


def test_a_configured_model_that_does_not_exist_falls_back_instead_of_breaking():
    config.set_value("model", "/nope/ggml-large-v3.bin")
    assert whisper.model() == "" or Path(whisper.model()).is_file()


def test_language_defaults_to_auto_rather_than_forcing_english():
    # Forcing `en` onto German speech makes whisper TRANSLATE instead of transcribe, which is why
    # this must never quietly default to a language.
    assert whisper.language() == "auto"
    config.set_value("language", "de")
    assert whisper.language() == "de"


def test_vocabulary_is_empty_until_words_are_configured():
    assert whisper.vocabulary() == ""
    config.set_value("vocabulary", ["Kubernetes", "Reinsch", "Kubernetes"])
    prompt = whisper.vocabulary()
    assert "Kubernetes" in prompt and "Reinsch" in prompt
    assert prompt.count("Kubernetes") == 1  # deduped


# --- availability ------------------------------------------------------------------------------


def test_available_names_the_fix_and_never_raises():
    ready, hint = dictate.available()
    assert ready or hint  # one or the other, always, on any machine


def test_the_short_clip_floor_is_not_treated_as_a_failure():
    # A brushed key is a gesture that was never a sentence; the daemon must not chime at it.
    assert dictate.TOO_SHORT in "too short — hold the key while you talk"
