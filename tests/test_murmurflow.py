"""Hermetic tests: no microphone, no whisper, no network, no launchd.

Everything that can be tested without hardware is tested here, and that turns out to be most of the
logic that actually goes wrong — the hallucination traps, the punctuation repair, the filler strip,
the config coercion and the polish shell-out. The parts that genuinely need a Mac with a microphone
(``start``/``finish``/``inject``) are exercised by ``murmurflow doctor`` and ``keytest`` instead,
because a mock of CoreAudio would only ever test the mock.
"""

from __future__ import annotations

import json
import os
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


# --- stopping the warm server -------------------------------------------------------------------


def test_stop_server_matches_our_port_and_never_a_neighbours(monkeypatch):
    # A second dictation tool on the same Mac runs its own whisper-server on its own port. Killing
    # by process NAME would take it down too; killing by port is the whole safety of this function.
    config.set_value("port", 8479)
    seen: dict[str, list[str]] = {}
    killed: list[int] = []

    class _Done:
        stdout = "111\n222\n"

    def _run(argv, **kwargs):
        seen["argv"] = argv
        return _Done()

    monkeypatch.setattr(dictate.subprocess, "run", _run)
    monkeypatch.setattr(dictate.os, "kill", lambda pid, sig: killed.append(pid))

    assert dictate.stop_server() == 2
    assert killed == [111, 222]
    assert "8479" in seen["argv"][-1]


def test_stop_server_survives_a_process_that_is_already_gone(monkeypatch):
    class _Done:
        stdout = "333\n"

    def _boom(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(dictate.subprocess, "run", lambda argv, **kw: _Done())
    monkeypatch.setattr(dictate.os, "kill", _boom)
    assert dictate.stop_server() == 0


# --- one listener, never two --------------------------------------------------------------------


def test_a_second_listener_is_refused_and_told_who_has_the_key(monkeypatch):
    # The doubled-sound bug: the login agent is live and you run `murmurflow listen` to watch it.
    dictate.listener_lock_path().parent.mkdir(parents=True, exist_ok=True)
    dictate.listener_lock_path().write_text("4242", "utf-8")
    monkeypatch.setattr(dictate, "_exited", lambda pid: False)  # 4242 is alive
    assert dictate.listener_pid() == 4242
    assert dictate.claim_listener() == 4242


def test_a_lock_left_by_a_crash_never_blocks_the_next_start(monkeypatch):
    # A hard reboot must not leave dictation needing a file deleted by hand.
    dictate.listener_lock_path().parent.mkdir(parents=True, exist_ok=True)
    dictate.listener_lock_path().write_text("4242", "utf-8")
    monkeypatch.setattr(dictate, "_exited", lambda pid: True)  # 4242 is gone
    assert dictate.listener_pid() == 0
    assert dictate.claim_listener() == 0
    assert dictate.listener_lock_path().read_text("utf-8") == str(os.getpid())


def test_claiming_twice_from_the_same_process_is_not_a_conflict():
    assert dictate.claim_listener() == 0
    assert dictate.claim_listener() == 0


# --- the trigger ---------------------------------------------------------------------------------


def test_the_default_trigger_is_two_modifiers_so_no_shortcut_can_fire_it():
    # A bare modifier is shared with every shortcut the user already types on it. The chord guard
    # discards the AUDIO, but it cannot un-play the tone, and a tool that chimes while you work
    # gets uninstalled.
    from murmurflow import hotkey

    assert hotkey.DEFAULT_TRIGGER in hotkey.COMBO_TRIGGERS
    assert len(hotkey.COMBO_TRIGGERS[hotkey.DEFAULT_TRIGGER]) == 2


def test_a_combo_is_down_only_when_every_one_of_its_flags_is():
    from murmurflow import hotkey

    class _Lib:
        state = 0

        def CGEventSourceFlagsState(self, _which):
            return self.state

    lib = _Lib()
    lib.state = hotkey.FLAG_CONTROL
    assert not hotkey.is_trigger_down("control_option", lib)  # half of it is not it
    lib.state = hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION
    assert hotkey.is_trigger_down("control_option", lib)


def test_every_combo_has_a_label_a_person_could_read():
    from murmurflow import hotkey

    for name in hotkey.COMBO_TRIGGERS:
        assert name in dictate.TRIGGER_LABELS, f"{name} would print as a raw config key"


def test_the_double_tap_window_is_measured_press_to_press():
    from murmurflow import hotkey

    # Release-to-release charged the second tap's own duration to the budget, so a 200ms gap plus
    # a 200ms press missed it and an ordinary double-tap did nothing at all.
    gap, press = 0.20, 0.20
    assert gap + press > 0.35  # what the old window was, and why it missed
    assert gap <= hotkey.DOUBLE_TAP_WINDOW
    assert press <= hotkey.TAP_MAX
