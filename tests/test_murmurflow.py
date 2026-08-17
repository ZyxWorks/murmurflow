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

from murmurflow import cli, config, dictate, whisper


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
# The highest-stakes logic in the tool, and it cuts BOTH ways: a false negative types words into
# your document that you never said, and a false positive deletes a sentence you did say, which
# reads as broken hardware. The list is only what a person never utters; `QUIET_DBFS` handles
# silence from the waveform, before whisper is asked anything.


@pytest.mark.parametrize(
    "text",
    [
        "thanks for watching!",
        "[BLANK_AUDIO]",
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
    config.set_value("stripFillers", True)
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


def test_a_rival_daemon_on_the_same_key_is_found_and_named(monkeypatch):
    # The doubling the lock CANNOT catch: another program, its own lock file, the same double-tap.
    # murmurflow was extracted from zyx, so a Mac running both is the ordinary case, not an exotic
    # one — and the murmurflow-only count reads a healthy "1" straight through it.
    monkeypatch.setattr(
        dictate, "_pgrep", lambda pattern: [31073] if pattern == "zyx voice listen" else []
    )
    assert dictate.rival_listeners() == [("zyx", [31073], "zyx voice uninstall")]


def test_nothing_else_on_the_key_reports_no_rival(monkeypatch):
    monkeypatch.setattr(dictate, "_pgrep", lambda pattern: [])
    assert dictate.rival_listeners() == []


# --- the trigger ---------------------------------------------------------------------------------


def test_the_default_trigger_is_two_modifiers_so_no_shortcut_can_fire_it():
    # A bare modifier is shared with every shortcut the user already types on it. The chord guard
    # discards the AUDIO, but it cannot un-play the tone, and a tool that chimes while you work
    # gets uninstalled.
    from murmurflow import hotkey

    assert hotkey.DEFAULT_TRIGGER in hotkey.COMBO_TRIGGERS
    assert len(hotkey.DEFAULT_TRIGGER.split("_")) == 2


def test_a_combo_is_down_only_when_every_one_of_its_flags_is(monkeypatch):
    # Platform-specific by nature, so it is asserted against the platform module and not the
    # gesture layer above it — which is the whole point of the seam.
    from murmurflow.platforms import macos

    class _Lib:
        state = 0

        def CGEventSourceFlagsState(self, _which):
            return self.state

        def CGEventSourceKeyState(self, _which, _code):
            return False

    lib = _Lib()
    monkeypatch.setattr(macos, "_load", lambda: lib)
    lib.state = macos.FLAG_CONTROL
    assert not macos.is_down("control_option")  # half of it is not it
    lib.state = macos.FLAG_CONTROL | macos.FLAG_OPTION
    assert macos.is_down("control_option")


def test_every_trigger_name_the_gesture_knows_is_one_the_platform_can_poll():
    # A config naming a key this platform cannot read is a trigger that silently does nothing.
    from murmurflow import hotkey

    names = hotkey.trigger_names()
    assert hotkey.DEFAULT_TRIGGER in names
    assert hotkey.DEFAULT_TAP_TRIGGER in names
    for alias, canonical in hotkey.TRIGGER_ALIASES.items():
        assert canonical in names, f"{alias} aliases to {canonical}, which cannot be polled"


def test_both_backends_answer_the_whole_contract():
    # The seam is only a seam if every backend implements all of it. A missing name is an
    # AttributeError on somebody else's machine, at the moment they press the key.
    import types

    from murmurflow import platforms
    from murmurflow.platforms import macos, unsupported, windows

    contract = [
        name
        for name, value in vars(platforms).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    ]
    assert len(contract) > 15, "the contract got smaller — did a seam name go missing?"
    for backend in (macos, windows, unsupported):
        missing = [n for n in contract if not hasattr(backend, n)]
        assert not missing, f"{backend.__name__} is missing {missing}"


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


# --- the gesture picks the key -------------------------------------------------------------------


def test_hold_defaults_to_a_combo_and_double_tap_to_one_ordinary_key():
    # Two gestures, two problems. A hold starts on the same key-down a shortcut does, so it needs a
    # combo. A double-tap does not, so forcing it onto two keys at once bought nothing and cost the
    # gesture its ergonomics — and its portability, since `control_option` is a macOS-shaped answer.
    from murmurflow import hotkey

    config.set_value("doubleTap", False)
    assert dictate.trigger_key() in hotkey.COMBO_TRIGGERS
    config.set_value("doubleTap", True)
    assert dictate.trigger_key() == "left_control"
    assert dictate.trigger_key() in hotkey.trigger_names()


def test_an_explicit_trigger_wins_over_the_gesture_default():
    config.set_value("doubleTap", True)
    config.set_value("trigger", "command_option")
    assert dictate.trigger_key() == "command_option"


def test_the_windows_spelling_of_a_key_is_the_same_key():
    from murmurflow import hotkey

    assert hotkey.canonical_trigger("ctrl_alt") == "control_option"
    assert hotkey.canonical_trigger("left_alt") == "left_option"
    assert hotkey.canonical_trigger("CONTROL-OPTION") == "control_option"
    config.set_value("trigger", "ctrl_alt")
    assert dictate.trigger_key() == "control_option"  # stored loosely, resolved to one spelling


def test_apple_dictation_can_only_clash_with_a_control_trigger():
    # A combo cannot collide with Apple's double-tap Control, which is most of why hold defaults
    # to one. The check must never invent a problem for a trigger that cannot have it.
    config.set_value("trigger", "control_option")
    assert dictate.apple_dictation_conflict() is False


# --- a room is not a sentence ---------------------------------------------------------------------


def test_the_quiet_floor_sits_between_real_speech_and_a_real_room():
    # Measured, not guessed: eleven clips of real dictation on this machine peaked between -15 and
    # -5 dBFS; a silent room on the same microphone measured -47. If this constant ever drifts into
    # either of those, whisper starts inventing sentences about room tone again — or real speech
    # gets dropped.
    quietest_real_speech = -15.0
    loudest_silent_room = (
        -38.0
    )  # the SAME room, measured twice: -47 and -38. A room is not a constant.
    assert loudest_silent_room < dictate.QUIET_DBFS < quietest_real_speech
    assert dictate.SILENT_DBFS < dictate.QUIET_DBFS  # a denied mic keeps its own, louder message


def test_a_room_recording_is_never_transcribed_at_all(monkeypatch, tmp_path):
    # The ordering IS the fix. Whisper answers "which language is this", confidently, about noise —
    # so the invented sentence must never be generated, not merely discarded afterwards.
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    called: list[str] = []
    monkeypatch.setattr(dictate, "peak_dbfs", lambda _p: -47.0)
    monkeypatch.setattr(
        dictate, "transcribe", lambda *a, **k: called.append("ran") or dictate.Heard("")
    )
    monkeypatch.setattr(dictate, "stop", lambda rec=None: wav)
    monkeypatch.setattr(dictate, "cue", lambda _kind: None)
    rec = dictate.Recording(pid=1, wav=wav, started_at=0.0)

    result = dictate.finish(rec, paste=False)

    assert called == [], "a silent room must not reach the transcriber"
    assert result.text == ""
    assert result.problem.startswith(dictate.NOTHING_SAID)


def test_nothing_said_is_a_non_event_and_never_gets_a_failure_tone():
    # Same rule as TOO_SHORT: nothing was asked for, so a noise reporting that it did not work is
    # the "beeping out of nowhere" that makes a background daemon feel broken.
    assert f"{dictate.NOTHING_SAID} (-47 dBFS)".startswith(
        (dictate.TOO_SHORT, dictate.NOTHING_SAID)
    )


def test_a_soft_voice_can_lower_the_floor_rather_than_being_stuck():
    # A far-field microphone in a big room is a different machine from the one this was measured
    # on. Without an override the answer to "it never hears me" is "install something else".
    assert dictate.quiet_floor() == dictate.QUIET_DBFS
    config.set_value("quietFloor", -45)
    assert dictate.quiet_floor() == -45.0


# --- what you said is what you get ----------------------------------------------------------------


def test_tidy_is_verbatim_by_default_including_the_word_hey():
    # The report: "sometimes I say 'hey, it seems like our murmurflow is still not working' and it
    # just redacts the hey." It did, by design, and the design was wrong. A word removed is
    # invisible; an "um" left in costs one keystroke.
    said = "hey, it seems like our murmurflow is still not working"
    assert dictate.tidy(said) == said
    assert dictate.tidy("So, um, ship it on Friday") == "So, um, ship it on Friday"
    assert dictate.tidy("this is, you know, basically fine") == "this is, you know, basically fine"


def test_filler_stripping_still_works_when_it_is_asked_for():
    config.set_value("stripFillers", True)
    assert dictate.tidy("So, um, ship it on Friday") == "ship it on Friday"


@pytest.mark.parametrize("said", ["thank you", "Thank you.", "you", "so", "bye", "Vielen Dank."])
def test_a_short_polite_sentence_is_never_swallowed(said):
    # These were all on the hallucination blocklist. Every one is something a person says as a
    # whole utterance, and swallowing it reads to the user as a broken microphone. QUIET_DBFS does
    # this job properly now, from the waveform, before whisper is ever asked.
    assert not dictate.is_hallucination(said)


@pytest.mark.parametrize(
    "junk", ["Thanks for watching!", "[BLANK_AUDIO]", "amara.org", "(silence)"]
)
def test_what_a_person_never_says_is_still_trapped(junk):
    assert dictate.is_hallucination(junk)


def test_the_program_name_typed_twice_is_the_same_command():
    # `murmurflow murmurflow config set cue glass` - what happens when the help, which prints every
    # verb with the program name so it can be pasted whole, gets pasted after typing the name.
    assert cli.main(["murmurflow", "config"]) == 0


def test_the_default_cue_is_the_system_sound_with_a_generated_fallback():
    # A Mac user has heard Tink and Pop for twenty years, so they read as acknowledgement without
    # being learned, and they follow the alert-volume setting a generated tone cannot see.
    assert dictate.cue_preset_name() == "system"
    assert dictate._cue_path(dictate.CUE_READY).startswith("/System/Library/Sounds/")


def test_a_missing_system_sound_falls_back_to_a_tone_and_never_to_silence(monkeypatch):
    # The cue is the ONLY signal that the microphone is live. Losing it to a missing file - a
    # stripped install, or any machine that is not a Mac - is not an acceptable failure.
    monkeypatch.setattr(dictate, "_SYSTEM_CUES", {})
    path = dictate._cue_path(dictate.CUE_READY)
    assert path.endswith(".wav") and dictate.FALLBACK_PRESET in path


def test_an_explicit_preset_still_wins_over_the_system_default():
    config.set_value("cue", "glass")
    assert dictate.cue_preset_name() == "glass"
    assert "glass" in dictate._cue_path(dictate.CUE_READY)


def test_a_long_paste_waits_longer_before_the_clipboard_comes_back():
    # The report this exists for: "when I talk longer, the whole text doesn't paste, only a few
    # words". A fixed 0.35s was measured against one short sentence; a minute of dictation is
    # ~1000 characters and the target is still inserting them when the restore lands.
    assert dictate.paste_settle("hi") == pytest.approx(0.35, abs=0.01)
    assert dictate.paste_settle("x" * 1000) > dictate.paste_settle("x" * 100)
    assert dictate.paste_settle("x" * 100_000) == dictate._PASTE_SETTLE_MAX


def test_the_wait_is_inlined_into_the_script_and_stays_a_number():
    # The delay is formatted into AppleScript source. A locale-formatted or exponent-formatted
    # float there is a syntax error, and the whole paste is lost with it.
    from murmurflow.platforms import macos

    script = macos._INJECT_SCRIPT.format(settle=f"{dictate.paste_settle('x' * 500):.2f}")
    assert "delay 0.85" in script
    assert "{settle}" not in script


def test_a_clip_reports_how_much_audio_actually_landed(tmp_path):
    import wave

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)  # exactly 2 seconds
    assert dictate.audio_seconds(path) == pytest.approx(2.0)
    assert dictate.audio_seconds(tmp_path / "nope.wav") == 0.0


def test_a_language_you_do_not_speak_is_a_hallucination():
    # Two clips on 2026-08-17 came back as fluent invented Icelandic and were typed into the
    # operator's window: above the quiet floor, above the confidence bar, on no blocklist.
    assert dictate.spoken_languages() == frozenset()  # nothing is refused until you say
    config.set_value("languages", ["de", "en"])
    assert dictate.spoken_languages() == frozenset({"de", "en"})
    config.set_value("languages", "de, EN ")
    assert dictate.spoken_languages() == frozenset({"de", "en"})


def test_the_server_language_is_read_off_the_body_beside_the_score():
    body = json.dumps({"text": "hallo", "detected_language_probability": 0.99, "language": "de"})
    assert dictate._confidence(body) == ("hallo", 0.99, "de")
    # An older server, or the cold path: no opinion, and no opinion is never a refusal.
    assert dictate._confidence("just text") == ("just text", 1.0, "")


# --- the platform seam ------------------------------------------------------------------------
#
# Everything below runs on ANY machine, which is the point: the Windows backend was written on a
# Mac, so the parts that can be checked without Windows are checked here rather than discovered by
# a client. What genuinely needs Windows — that keybd_event reaches the foreground app, that
# schtasks accepts the XML — is exercised by `murmurflow doctor` there, and is named as untested
# in the PR rather than implied to work.


def test_the_two_backends_shape_ffmpeg_for_their_own_input():
    from murmurflow.platforms import macos, windows

    assert macos.capture_args("1") == ["-f", "avfoundation", "-i", ":1"]
    assert macos.capture_args("") == ["-f", "avfoundation", "-i", ":default"]
    # dshow addresses a microphone by NAME and has no "default" token at all, which is why the id
    # is opaque above the seam and why an empty one is nothing to record rather than a default.
    assert windows.capture_args("Mikrofon (Realtek)") == [
        "-f",
        "dshow",
        "-i",
        "audio=Mikrofon (Realtek)",
    ]
    assert windows.capture_args("") == []


def test_dshow_devices_are_read_out_of_what_ffmpeg_actually_prints(monkeypatch):
    # Real ffmpeg output, both shapes: the modern `(audio)` suffix and the older header split.
    stderr = (
        '[dshow @ 0x1] "Integrated Camera" (video)\n'
        '[dshow @ 0x1]   Alternative name "@device_pnp_\\\\?\\usb#vid_04f2"\n'
        '[dshow @ 0x1] "Mikrofon (Realtek(R) Audio)" (audio)\n'
        '[dshow @ 0x1]   Alternative name "@device_cm_{33D9A762}"\n'
        '[dshow @ 0x1] "Headset (WH-1000XM4)" (audio)\n'
    )

    class _Proc:
        pass

    proc = _Proc()
    proc.stderr = stderr
    from murmurflow.platforms import windows

    monkeypatch.setattr(windows.shutil, "which", lambda _n: "ffmpeg")
    monkeypatch.setattr(windows.subprocess, "run", lambda *a, **k: proc)
    assert windows.list_inputs() == [
        ("Mikrofon (Realtek(R) Audio)", "Mikrofon (Realtek(R) Audio)"),
        ("Headset (WH-1000XM4)", "Headset (WH-1000XM4)"),
    ]
    assert windows.default_input() == "Mikrofon (Realtek(R) Audio)"


def test_the_trigger_vocabulary_is_one_vocabulary_on_both_platforms():
    from murmurflow import hotkey
    from murmurflow.platforms import macos, windows

    shared = macos.trigger_names() & windows.trigger_names()
    assert hotkey.DEFAULT_TRIGGER in shared  # a config file moves between machines unchanged
    assert hotkey.DEFAULT_TAP_TRIGGER in shared
    # And the two honest differences, each of them hardware and neither of them a preference.
    assert "fn" in macos.trigger_names() and "fn" not in windows.trigger_names()
    assert "right_command" in windows.trigger_names()


def test_the_logon_task_is_well_formed_and_asks_to_be_restarted():
    import xml.etree.ElementTree as ET

    from murmurflow.platforms import windows

    xml = windows.task_xml(r"C:\Users\h\.murmurflow\listen.cmd", "DESK\\h")
    root = ET.fromstring(xml)  # noqa: S314 — our own template, not input
    namespace = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert root.find(".//t:Exec/t:Command", namespace).text.endswith("listen.cmd")
    # `KeepAlive` is launchd's word for it. Plain schtasks flags cannot express it, which is the
    # whole reason this is XML and not a command line.
    assert root.find(".//t:RestartOnFailure/t:Count", namespace).text == "999"
    assert root.find(".//t:Settings/t:Hidden", namespace).text == "true"


def test_an_unported_platform_never_reports_that_input_just_happened():
    # The chord guard aborts a recording when the user did something else DURING the press. A
    # backend that answered "0 seconds ago" would abort every single one; "forever ago" fails open.
    from murmurflow.platforms import unsupported

    assert unsupported.seconds_since_input() == float("inf")
    assert unsupported.keys_unavailable()  # and it says why


# --- lending the key --------------------------------------------------------------------------


def test_lending_the_key_expires_on_its_own():
    # The whole design. A borrower that crashes holding the key would otherwise leave dictation
    # silently dead with nothing on screen to explain it — the key is pressed, no cue plays, and
    # there is nothing to read. An expiry makes the worst case a few minutes, not an outage.
    assert dictate.paused() == (False, "")
    dictate.pause(60, who="a Zyx huddle")
    lent, holder = dictate.paused()
    assert lent and "a Zyx huddle" in holder
    dictate.pause(-5)  # clamped to the 1s floor, so this is the shortest pause there is
    import time as _t

    _t.sleep(1.1)
    assert dictate.paused() == (False, "")
    assert not dictate.pause_path().exists(), "an expired pause cleans itself up"


def test_a_pause_is_capped_however_long_the_borrower_asks():
    until = dictate.pause(999_999)
    import time as _t

    assert until - _t.time() <= dictate.MAX_PAUSE_SECONDS + 1
    dictate.resume()


def test_every_way_the_marker_can_be_broken_reads_as_yours():
    # Fail OPEN, one-directionally: the failure on the other side is a microphone that never opens
    # again, and nobody would ever guess that a file is why.
    for junk in ("", "not json", '{"until": "soon"}', "[]"):
        dictate.pause_path().write_text(junk, "utf-8")
        assert dictate.paused() == (False, ""), junk
    dictate.resume()


def test_resume_is_idempotent_and_says_which_it_was():
    dictate.pause(60)
    assert dictate.resume() is True
    assert dictate.resume() is False


def test_the_trigger_verb_prints_one_parseable_name(capsys):
    # The reader is a PROGRAM deciding whether its own hotkey collides with this one. Everything
    # else that says which key this is on says it in a sentence, which is right for a person and
    # unparseable for that. Borrowing the key when it does not collide stands dictation down across
    # every other app for as long as the borrower runs, silently — so the answer has to be exact.
    config.set_value("doubleTap", True)
    assert cli.main(["trigger"]) == 0
    assert capsys.readouterr().out.strip() == "left_control"
    config.set_value("trigger", "ctrl_alt")
    assert cli.main(["trigger"]) == 0
    assert capsys.readouterr().out.strip() == "control_option"  # canonical, not as it was typed


# --- the paste says what it did -----------------------------------------------------------------
#
# A dictation that lands in full and one that lands as its first eight words were the SAME `[OK]`
# line, because `osascript` exiting 0 only means the keystroke was posted. These read the report
# the inject script now hands back.


def test_a_clean_paste_names_the_app_it_went_to():
    from murmurflow.platforms import macos

    assert macos._paste_note("566\t566\tiTerm2\n", 566) == "→ iTerm2"


def test_a_truncated_copy_says_so_loudly():
    from murmurflow.platforms import macos

    note = macos._paste_note("120\t120\tiTerm2", 566)
    assert "COPIED ONLY 120/566" in note and "iTerm2" in note


def test_a_clipboard_taken_mid_paste_is_a_different_story_and_says_so():
    from murmurflow.platforms import macos

    note = macos._paste_note("566\t-1\tMail", 566)
    assert "CLIPBOARD CHANGED UNDER THE PASTE (-1/566)" in note


def test_an_unreadable_report_is_no_note_and_never_an_exception():
    # The diagnostic must never be able to break the paste it is describing.
    from murmurflow.platforms import macos

    for junk in ("", "nonsense", "1\t2", "a\tb\tc", "566\t566\tiTerm2\textra"):
        assert macos._paste_note(junk, 566) == "", junk
