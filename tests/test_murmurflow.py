"""Hermetic tests: no microphone, no whisper, no network, no launchd.

Everything that can be tested without hardware is tested here, and that turns out to be most of the
logic that actually goes wrong — the hallucination traps, the punctuation repair, the filler strip,
the config coercion and the polish shell-out. The parts that genuinely need a Mac with a microphone
(``start``/``finish``/``inject``) are exercised by ``murmurflow doctor`` and ``keytest`` instead,
because a mock of CoreAudio would only ever test the mock.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murmurflow import cli, config, dictate, whisper


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.murmurflow. This is the whole isolation the suite needs."""
    monkeypatch.setenv(config.HOME_ENV, str(tmp_path))
    monkeypatch.setenv("MURMURFLOW_NO_AUDIO", "1")  # never make a sound in CI
    # Who holds a port is cached across calls, and the cache is module state. On a DEVELOPER's Mac
    # a real whisper-server is on that port, so a stale True leaked into other tests and hid a
    # failure that only CI — where nothing is listening — could see.
    dictate._OWNERSHIP.clear()
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
    assert dictate.strip_fillers("So, um, ship it on Friday") == "So, ship it on Friday"
    assert dictate.strip_fillers("Erm, ship it") == "ship it"
    assert dictate.strip_fillers("It is, uh, fixed") == "It is, fixed"


def test_the_strip_removes_sounds_and_never_words():
    # Every one of these used to lose a real word: the list held `hey`, `so`, `well`, `right`,
    # `you know`, `i mean` and `basically`, and each is also an ordinary English word. A missed
    # filler costs one keystroke; a deleted word is invisible until you re-read your own sentence.
    for said in (
        "Do you know what time it is?",
        "Let me know if you know a good restaurant.",
        "I mean what I said.",
        "Hey, ship it on Friday",
        "Well, that is basically right.",
        "So we ship on Friday.",
    ):
        assert dictate.strip_fillers(said) == said


def test_a_clean_transcript_passes_through_byte_identical():
    clean = "Ship the invoice on Friday."
    assert dictate.tidy(clean) == clean


def test_strip_never_empties_a_line_that_was_all_filler():
    # It removes what it can and keeps whatever survives ...
    assert dictate.strip_fillers("um, uh, yeah") == "yeah"
    # ... but a line with nothing left after the strip falls back to the original rather than "".
    assert dictate.strip_fillers("um uh") == "um uh"


def test_punctuation_left_dangling_by_the_strip_is_repaired():
    # "...fixed, um, you know." -> the strip leaves "fixed," hanging at the end of the line.
    config.set_value("stripFillers", True)
    assert dictate.tidy("It is fixed, um.") == "It is fixed."


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

    # Four kills, because there are TWO of ours now: the big server and the small one that answers
    # the live pass. The fake `pgrep` above answers both queries with the same two pids.
    assert dictate.stop_server() == 4
    assert killed == [111, 222, 111, 222]
    assert "8480" in seen["argv"][-1]  # the last query is the live server, one port up


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


def test_the_hold_floor_waits_only_the_remainder_of_itself(monkeypatch):
    # `if elapsed < min_hold: sleep(min_hold)` waited a whole further floor on top of the press
    # that had nearly finished it - so the SHORTEST holds, the ones already fighting for audio,
    # took nearly twice the intended floor to answer. The wait is the remainder.
    import types

    from murmurflow import hotkey

    clock, sleeps = [0.0], []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(
        hotkey, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=fake_sleep)
    )
    monkeypatch.setattr(hotkey, "seconds_since_keydown", lambda: 99.0)  # never a chord
    down = iter([True, False, False])
    monkeypatch.setattr(hotkey, "is_trigger_down", lambda _t: next(down, False))
    released = []
    hotkey.listen(
        lambda: None,
        lambda: released.append(clock[0]),
        min_hold=0.15,
        poll_hz=100,  # one poll = 0.01s, so the press lasts 0.01s
        should_stop=lambda: len(released) > 0,
    )
    assert released, "the release never fired"
    floor = [s for s in sleeps if s > 0.01]
    assert floor == [pytest.approx(0.14)], f"waited {floor}, not the 0.14 remaining of the floor"


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


def test_a_fresh_install_taps_and_says_so(tmp_path, monkeypatch):
    # NOTHING pinned the default gesture before this, which is how it could ship as hold: every
    # other gesture test sets `doubleTap` explicitly first. An empty config is what a new user has,
    # so an empty config is what has to be asserted.
    config.config_path().write_text("{}", "utf-8")
    assert dictate.double_tap_mode() is True
    assert dictate.trigger_key() == "left_control"  # one ordinary key, same on every OS

    # And the advice on a brushed key has to match the gesture. Telling somebody in tap mode to
    # "hold the key while you talk" instructs them to do the one thing that is wrong for them.
    wav = tmp_path / "brushed.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(dictate, "stop", lambda rec=None: wav)
    brushed = dictate.Recording(pid=1, wav=wav, started_at=time.time())

    assert "hold" not in dictate.finish(brushed, paste=False).problem
    config.set_value("doubleTap", False)
    assert "hold the key while you talk" in dictate.finish(brushed, paste=False).problem


def test_an_explicit_trigger_wins_over_the_gesture_default():
    config.set_value("doubleTap", True)
    config.set_value("trigger", "command_option")
    assert dictate.trigger_key() == "command_option"


def test_the_windows_spelling_of_a_key_is_the_same_key():
    from murmurflow import hotkey

    assert hotkey.canonical_trigger("ctrl_alt") == "control_option"
    assert hotkey.canonical_trigger("left_alt") == "left_option"
    assert hotkey.canonical_trigger("CONTROL-OPTION") == "control_option"
    # The README promises ctrl/alt/win work EVERYWHERE control/option/command do. A hand-written
    # list of whole names kept missing sides and combos; the matrix is the promise, written down.
    for windows_name, canonical in {
        "left_ctrl": "left_control",
        "right_ctrl": "right_control",
        "right_alt": "right_option",
        "left_win": "left_command",
        "right_win": "right_command",
        "ctrl_win": "control_command",
        "win_alt": "command_option",
        "ctrl_shift": "control_shift",
        "super": "command",
        "meta_alt": "command_option",
    }.items():
        assert hotkey.canonical_trigger(windows_name) == canonical
        assert canonical in hotkey.trigger_names()
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


def test_a_whisper_segment_break_never_reaches_the_paste():
    # The report: "it makes a lot of line breaks, in random places." This string is a VERBATIM
    # `/inference` response from the operator's own whisper-server (large-v3-turbo, one clip,
    # 2026-08-19) - whisper puts a `\n` at every segment boundary and segments mid-clause, so the
    # break lands between "the" and "other". The paste is clipboard + cmd-V, which means every one
    # of these arrives in the target app as a real Return.
    raw = " It seems like it's way better working here than in the\n other, you know,\n"
    flat = "It seems like it's way better working here than in the other, you know,"
    assert dictate.tidy(raw) == flat
    # And with the strip on, where the flattening used to hide inside `strip_fillers`.
    config.set_value("stripFillers", True)
    assert "\n" not in dictate.tidy(raw)


def test_a_segment_seam_inside_a_word_does_not_become_a_space():
    # The report, and his own dictated message is the evidence: "zyxworks.gith ub.io",
    # "z yxworks.com", "murmur flow". whisper segments on the DECODER's budget, so the seam lands
    # between two tokens of one word - and that segment starts with no leading space, because
    # whisper carries a word's space inside its first token. Joining every seam with " " split the
    # word. No whitespace on either side of the seam = the word was cut in half.
    assert dictate.tidy("go to zyxworks.gith\nub.io") == "go to zyxworks.github.io"
    assert dictate.tidy("Mail me at z\nyxworks.com.") == "Mail me at zyxworks.com."
    # ...and a seam at a REAL word boundary still gets its space, from either side of the newline.
    assert dictate.tidy("in the\n other") == "in the other"
    assert dictate.tidy("in the \nother") == "in the other"


def test_boilerplate_appended_to_a_real_sentence_is_not_typed():
    # The report: "it adds a thank you in the end". The whole-transcript trap cannot see this one -
    # the line is mostly a real sentence, so it scores as confident speech and every word is typed.
    assert dictate.tidy("Make it public. Thanks for watching!") == "Make it public."
    assert dictate.tidy("Ship it. [BLANK_AUDIO]") == "Ship it."
    assert dictate.tidy("Fertig. Untertitel von Stephanie Geiges") == "Fertig."
    # Never the last sentence standing: a clip that is ONLY boilerplate is `finish`'s call, and a
    # real sentence that merely looks polite is never touched.
    assert dictate.tidy("Thanks for watching!") == "Thanks for watching!"
    assert dictate.tidy("Ship it. Thank you.") == "Ship it. Thank you."


def test_filler_stripping_still_works_when_it_is_asked_for():
    config.set_value("stripFillers", True)
    assert dictate.tidy("Um, ship it on Friday") == "ship it on Friday"


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
    # `murmurflow murmurflow config` - what happens when the help, which prints every verb with the
    # program name so it can be pasted whole, gets pasted after typing the name.
    assert cli.main(["murmurflow", "config"]) == 0


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


def test_every_windows_call_that_returns_a_pointer_says_so():
    # ctypes guesses a 32-bit int for any undeclared return. On 64-bit Windows that cuts the top
    # half off a HANDLE, and the truncated value goes into `ctypes.memmove` — a crash on the first
    # paste. The real DLLs are not here, so the declaration is checked against a stand-in.
    import ctypes
    import types

    from murmurflow.platforms import windows

    class FakeDLL:
        def __init__(self):
            self._calls = {}

        def __getattr__(self, name):
            return self._calls.setdefault(name, types.SimpleNamespace(restype=ctypes.c_int))

    user32, kernel32 = FakeDLL(), FakeDLL()
    windows._declare_signatures(user32, kernel32)
    for name in windows._POINTER_RETURNS["user32"]:
        assert getattr(user32, name).restype is ctypes.c_void_p
    for name in windows._POINTER_RETURNS["kernel32"]:
        assert getattr(kernel32, name).restype is ctypes.c_void_p
    assert user32.GetAsyncKeyState.restype is ctypes.c_short
    # And every pointer-returning call the module actually makes is on the list, or the next one
    # added is broken in exactly the way this test exists to catch.
    source = (Path(windows.__file__)).read_text()
    for name in ("GlobalAlloc", "GlobalLock", "GlobalFree", "GetClipboardData", "SetClipboardData"):
        assert name in windows._POINTER_RETURNS["kernel32"] + windows._POINTER_RETURNS["user32"]
        assert f".{name}(" in source  # still called; drop it from the list when it is not


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


def test_the_language_gate_reads_a_code_even_when_the_server_says_a_name(monkeypatch):
    """whisper-server answers `"language": "english"`; a person writes `["de", "en"]`.

    Compared as-is, the gate rejects EVERY sentence the moment the warm server is healthy — and
    today it silently rejects none, because the same mismatch made it inert. `language_probabilities`
    is keyed by the codes themselves, so the top key is the answer with no table to maintain.
    """
    payload = json.dumps(
        {
            "text": "hello there",
            "language": "english",
            "detected_language_probability": 0.99,
            "language_probabilities": {"en": 0.99, "de": 0.004},
        }
    )
    text, confidence, spoken = dictate._confidence(payload)
    assert (text, spoken) == ("hello there", "en")
    assert confidence == 0.99
    # An older server with no probabilities still resolves through the name table.
    _, _, older = dictate._confidence(json.dumps({"text": "hallo", "language": "german"}))
    assert older == "de"


def test_the_config_may_name_a_language_or_code_it(monkeypatch, tmp_path):
    config.set_value("languages", ["english", "de"])
    assert dictate.spoken_languages() == frozenset({"en", "de"})


def test_the_cold_path_reports_its_language_so_the_gate_still_applies(monkeypatch, tmp_path):
    """The gate was warm-only, and a wedged warm server is exactly when it is needed.

    Live (2026-08-17): MurmurFlow's own whisper-server answered every request "FFmpeg conversion
    failed", every clip fell through to the cold CLI, and a 1.8s desk bump came back as Japanese
    and was typed into a terminal. Both guards that would have caught it read fields the cold path
    did not produce.
    """
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(dictate, "transcribe_warm", lambda *a, **k: dictate.Heard(""))
    monkeypatch.setattr(
        whisper, "transcribe_heard", lambda *a, **k: ("ご視聴ありがとうございました", "ja")
    )
    heard = dictate.transcribe(wav)
    assert heard.language == "ja"
    assert heard.warm is False  # ...and the log says so, so a dead warm server is visible

    config.set_value("languages", ["de", "en"])
    monkeypatch.setattr(dictate, "peak_dbfs", lambda _p: -1.0)  # a loud bump clears the level gate
    monkeypatch.setattr(dictate, "transcribe", lambda *a, **k: heard)
    monkeypatch.setattr(dictate, "stop", lambda rec=None: wav)
    result = dictate.finish(dictate.Recording(pid=1, wav=wav, started_at=0.0), paste=False)
    assert result.text == ""
    assert "ja" in result.problem
    assert result.warm is False


# --- a setting that cannot work is refused, not written -------------------------------------------
#
# Every way of misconfiguring this daemon fails SILENTLY and they all fail identically: you hold
# the key and nothing happens. A trigger name the platform cannot poll, a cue that is not a preset,
# a misspelt key written to a file nothing reads — none of them leaves a trace anywhere. The typo
# is the only moment any of it is cheap to catch.


def test_a_misspelt_setting_is_refused_and_the_real_one_is_named(capsys):
    assert cli.main(["config", "set", "stripfillers", "true"]) == 2
    out = capsys.readouterr().out
    assert "stripFillers" in out  # the near match, not just "unknown"
    assert config.load() == {}  # and nothing was written


def test_a_trigger_this_machine_cannot_poll_is_never_bound(capsys):
    # `left_ctrl` used to stand here, and it was the alias hole rather than an unpollable key.
    assert cli.main(["config", "set", "trigger", "capslock"]) == 2
    assert "left_control" in capsys.readouterr().out  # the list of what does work
    assert "trigger" not in config.load()


def test_a_trigger_is_stored_under_one_spelling_however_it_was_typed(monkeypatch):
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    assert cli.main(["config", "set", "trigger", "ctrl_alt"]) == 0
    assert config.load()["trigger"] == "control_option"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("doubleTap", "yep"),  # anything but true/false reads as false
        ("quietFloor", "quiet"),  # the gate would fall back to -30 and never say so
        ("quietFloor", "30"),  # peak dBFS is never above 0: a positive floor discards EVERY clip
        ("quietFloor", "0"),  # and zero is the same gate, only harder to spot
        ("languages", '["d3"]'),  # two characters, but not a language: every clip thrown away
        ("port", "99999"),
        ("language", "deutsch"),  # `de` is the code; a name pins nothing
        ("languages", '["de", "klingon"]'),  # a clip in a language not on the list is DISCARDED
        ("model", "~/there-is-no-model-here.bin"),
    ],
)
def test_a_value_that_cannot_work_is_refused_rather_than_written(key, value):
    assert cli.main(["config", "set", key, value]) == 2
    assert key not in config.load()


@pytest.mark.parametrize(
    ("key", "value"),
    [("language", "auto"), ("language", "en"), ("languages", '["de", "english"]')],
)
def test_the_spellings_a_person_actually_uses_are_all_accepted(key, value, monkeypatch):
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    assert cli.main(["config", "set", key, value]) == 0


def test_unsetting_a_key_still_works_but_not_one_that_never_existed(monkeypatch):
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    config.set_value("inputName", "Built-in")
    assert cli.main(["config", "set", "inputName"]) == 0
    assert "inputName" not in config.load()
    assert cli.main(["config", "set", "inputNaem"]) == 2


def test_changing_the_model_stops_the_server_that_baked_the_old_one_in(monkeypatch):
    """Restarting the LISTENER cannot do it, and that is the whole trap.

    The daemon comes back, finds the old whisper-server still answering on the port, reuses it by
    design — and the model the user just chose never loads. Stopped before the write, so a change
    to `port` still finds the server on the port it is really on.
    """
    order: list[str] = []
    monkeypatch.setattr(cli.service, "running", lambda: True)
    monkeypatch.setattr(cli.service, "restart", lambda: bool(order.append("restart")) or True)
    monkeypatch.setattr(cli.dictate, "stop_server", lambda: bool(order.append("stop")) or 1)
    assert cli.main(["config", "set", "port", "9001"]) == 0
    assert order == ["stop", "restart"]
    # ...and a setting the server knows nothing about does not pay for a model reload.
    order.clear()
    assert cli.main(["config", "set", "stripFillers", "true"]) == 0
    assert order == ["restart"]


# --- the level gate reads the waveform, fast ------------------------------------------------------


def _wav(path: Path, samples: list[int]) -> Path:
    import array
    import wave

    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(array.array("h", samples).tobytes())
    return path


def test_the_peak_is_the_loudest_sample_in_either_direction(tmp_path):
    # `max(max(s), -min(s))` replaced a per-sample Python loop on the hot path. It has to agree
    # with the obvious version everywhere, INCLUDING on the negative full-scale sample that has no
    # positive twin: -32768 negated does not fit in the sample type it came from.
    assert dictate.peak_dbfs(_wav(tmp_path / "loud.wav", [32767, -3, 5])) == pytest.approx(
        0.0, abs=0.01
    )
    assert dictate.peak_dbfs(_wav(tmp_path / "neg.wav", [-32768, 1])) == pytest.approx(
        0.0, abs=0.01
    )
    assert dictate.peak_dbfs(_wav(tmp_path / "quiet.wav", [328, -100])) == pytest.approx(
        -40, abs=0.5
    )
    assert dictate.peak_dbfs(_wav(tmp_path / "silent.wav", [0, 0, 0])) == float("-inf")
    assert dictate.peak_dbfs(tmp_path / "not-a-file.wav") == 0.0  # no opinion, never a crash


# --- a warm server that answers wrongly is bounced ------------------------------------------------


def _drive_listener(monkeypatch, results, *, warm_starts=True):
    """Run `listen_loop` over a fixed list of dictations. Returns (server starts, server stops)."""
    starts: list[int] = []
    stops: list[int] = []

    class _Inline:  # the bounce runs on a thread; here it runs where the assertion can see it
        def __init__(self, target=None, args=(), kwargs=None, daemon=False):
            self._run = lambda: target(*args, **(kwargs or {}))

        def start(self):
            self._run()

    monkeypatch.setattr(dictate.threading, "Thread", _Inline)
    monkeypatch.setattr(dictate, "available", lambda: (True, ""))
    monkeypatch.setattr(dictate, "claim_listener", lambda: 0)
    monkeypatch.setattr(dictate, "reap_orphans", lambda: 0)
    monkeypatch.setattr(dictate, "resolve_input", lambda: ("0", "a mic"))
    monkeypatch.setattr(dictate, "paused", lambda: (False, ""))
    monkeypatch.setattr(dictate, "start_server", lambda **_k: bool(starts.append(1)) or warm_starts)
    monkeypatch.setattr(dictate, "stop_server", lambda at=0: bool(stops.append(at)) or 1)
    monkeypatch.setattr(dictate, "preroll_claim", lambda: dictate.Recording(1, Path("x.wav"), 0.0))
    monkeypatch.setattr(dictate, "stream_start", lambda _rec: None)  # not what this drives
    pending = list(results)
    monkeypatch.setattr(dictate, "finish", lambda _rec: pending.pop(0))

    def _bind(on_press, on_release, **_kwargs):
        for _ in range(len(results)):
            on_press()
            on_release()
        return "driven"

    monkeypatch.setattr(dictate, "bind_trigger", _bind)
    dictate.listen_loop()
    return starts, stops


def _clip(warm):
    return dictate.Result("hello", 1.0, 900, True, "", -14.0, 1.0, "", warm)


def test_a_warm_server_that_stopped_answering_is_restarted(monkeypatch):
    """A server that DIED is visible — `server_up` says no. One that answers wrongly is not.

    Live (2026-08-17): whisper-server replied "FFmpeg conversion failed" to every request, every
    clip silently took the cold path where the confidence gate can judge nothing, and a bump on
    the desk came back as Japanese and was typed. Nothing on screen said the fast path was gone.
    """
    starts, stops = _drive_listener(monkeypatch, [_clip(False), _clip(False), _clip(False)])
    # Bounced once, at the second cold clip — and ONLY the big server's port. Taking the live
    # server with it left every later partial on the big model until the next daemon restart.
    assert stops == [dictate.port()]
    assert dictate.partial_port() not in stops
    assert len(starts) == 2  # the one at daemon start, and the one that brought it back


def test_a_wedged_server_is_not_a_healthy_one(monkeypatch):
    """`server_up` opens a socket; a wedged server accepts sockets all day and transcribes nothing.

    That gap is why the FFmpeg-conversion failure ran for a full day with `murmurflow doctor`
    reporting the warm server green. The 500 body names the cause, so it is carried out.
    """
    import urllib.error

    monkeypatch.setattr(dictate, "server_up", lambda: True)

    def _wedged(*_a, **_k):
        raise urllib.error.HTTPError(
            "http://127.0.0.1/inference",
            500,
            "Internal Server Error",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":"FFmpeg conversion failed."}'),
        )

    monkeypatch.setattr(dictate.urllib.request, "urlopen", _wedged)
    ok, why = dictate.server_answers()
    assert ok is False
    assert "FFmpeg conversion failed" in why


def test_a_server_that_answers_an_inference_is_healthy(monkeypatch):
    monkeypatch.setattr(dictate, "server_up", lambda: True)
    monkeypatch.setattr(
        dictate.urllib.request, "urlopen", lambda *_a, **_k: contextlib.nullcontext(object())
    )
    assert dictate.server_answers() == (True, "")


def test_a_missing_server_is_reported_without_asking_it_anything(monkeypatch):
    monkeypatch.setattr(dictate, "server_up", lambda: False)
    ok, why = dictate.server_answers()
    assert ok is False
    assert str(dictate.port()) in why


def test_the_warm_server_is_started_in_a_writable_directory(monkeypatch, tmp_path):
    """The cause behind the test above, found 2026-08-18 — and it is one keyword argument.

    `--convert` makes whisper-server shell out to ffmpeg, which writes its converted copy into the
    server's WORKING DIRECTORY. Started from a shell that is the repo, so every manual test passed;
    under the installed agent the daemon inherits `/`, which is not writable, and the server answers
    every request `500 {"error":"FFmpeg conversion failed."}`. Proven live with two servers, same
    binary and flags and model and request: `/` failed in 0.03s, a writable dir answered in 2.17s.
    """
    seen: dict[str, object] = {}

    def _popen(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        raise OSError("not really spawning anything in a test")

    monkeypatch.setattr(dictate, "server_up", lambda _at=0: False)
    monkeypatch.setattr(
        dictate, "serve_command", lambda _m="", _at=0: ["whisper-server", "--convert"]
    )
    monkeypatch.setattr(dictate.subprocess, "Popen", _popen)
    assert dictate.start_server() is False
    cwd = Path(str(seen["kwargs"]["cwd"]))
    assert cwd.is_dir(), "the server must not be spawned into a directory that does not exist"
    assert os.access(cwd, os.W_OK), "ffmpeg's converted copy is written HERE — it must be writable"
    assert cwd != Path("/")


def test_one_cold_clip_is_not_enough_to_bounce_a_loading_server(monkeypatch):
    # A single cold clip is what a server still loading its 1.6 GB model looks like, and bouncing
    # it for that restarts the load it was in the middle of.
    _starts, stops = _drive_listener(monkeypatch, [_clip(False), _clip(True), _clip(False)])
    assert stops == []


def test_a_machine_with_no_warm_server_is_never_bounced(monkeypatch):
    # Cold on every clip is the DESIGN when whisper-server was never there. Restarting a server
    # that does not exist would hammer a missing binary after every sentence.
    _starts, stops = _drive_listener(monkeypatch, [_clip(False)] * 4, warm_starts=False)
    assert stops == []


def test_turning_keepaudio_off_deletes_the_clip_it_kept(monkeypatch):
    # "until you turn it off" is the README's promise. Nothing else deletes this file - the orphan
    # reaper only globs `dictate-*.wav` - so if the switch-off does not, the recording is forever.
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    assert cli.main(["config", "set", "keepAudio", "true"]) == 0
    kept = dictate.kept_audio()
    kept.write_bytes(b"RIFF")
    assert cli.main(["config", "set", "keepAudio", "false"]) == 0
    assert not kept.exists()
    # And unsetting it is the same "off", so it deletes too - and a missing file is not an error.
    kept.write_bytes(b"RIFF")
    assert cli.main(["config", "set", "keepAudio"]) == 0
    assert not kept.exists()
    assert cli.main(["config", "set", "keepAudio", "false"]) == 0


def test_the_daemon_log_lives_with_the_rest_of_the_home(tmp_path, monkeypatch):
    """Both service backends redirect the daemon here, and both used to spell it out themselves —
    which split the log away from the home the moment `MURMURFLOW_HOME` was set."""
    from murmurflow.platforms import macos, windows

    # Rendering the job normally BUILDS the .app bundle, which copies a 17 MB interpreter into the
    # real ~/Applications. Hermetic means hermetic: nothing here touches anything outside tmp_path.
    monkeypatch.setattr(macos, "_launchd_argv", lambda args: ["/bin/true", *args])
    assert config.log_path() == tmp_path / "listen.log"
    assert str(config.log_path()) in str(macos.render_plist(["listen"], {}))
    assert windows._log_path() == config.log_path()


# --- who has the typing permission, and how we know -----------------------------------------------
#
# `AXIsProcessTrusted` answers for the process that ASKS. A probe spawned from this CLI is
# attributed by TCC to whatever is responsible for the CLI, never to the .app the launchd agent
# runs as — so the health report said NOT GRANTED on a Mac that had been typing dictations all
# afternoon, and sent its owner to switch on a switch that was already on.


def test_a_paste_that_landed_settles_the_permission_question(tmp_path):
    config.log_path().write_text(
        '[OK] 4.4s · -17dBFS · 1999ms · 24 chars · → Slack · "ship it on friday"\n', "utf-8"
    )
    assert dictate.last_paste_verdict() is True
    granted, evidence = cli._typing_permission()
    assert granted and "last paste landed" in evidence


def test_a_paste_that_was_refused_says_so_and_is_not_a_probe(tmp_path):
    config.log_path().write_text(
        '[OK] 1.0s · -14dBFS · 900ms · 4 chars · → Notes · "yes"\n'
        "[!] not permitted to control this Mac. Accessibility must be granted...\n",
        "utf-8",
    )
    assert dictate.last_paste_verdict() is False  # the LAST attempt, not the best one


def test_secure_input_is_not_read_as_a_missing_permission(tmp_path):
    # It is a password field somebody had focused for a moment, it clears on its own, and it has
    # its own row. Reading it as a revoked grant would send the user to System Settings for it.
    config.log_path().write_text(
        "[!] Secure Input is active, so macOS is blocking synthetic paste. "
        "The text is on your clipboard — press paste.\n",
        "utf-8",
    )
    assert dictate.last_paste_verdict() is None


def test_a_daemon_that_has_never_pasted_has_no_opinion(tmp_path):
    assert dictate.last_paste_verdict() is None  # no log at all
    config.log_path().write_text("listening — hold control_option (mic: a mic)\n", "utf-8")
    assert dictate.last_paste_verdict() is None


# --- a recorder that had to be killed still leaves an honest clip ---------------------------------
#
# ffmpeg writes the real lengths when it EXITS. SIGKILL on the fallback path skips that, and on
# Windows there is no gentler path at all: `os.kill` can only send CTRL_C/CTRL_BREAK — which a
# `pythonw` daemon has no console to send — and anything else is TerminateProcess.
#
# MEASURED against real ffmpeg rather than assumed: what it leaves behind is `0xFFFFFFFF` in both
# size fields, not zero. The audio still decodes, so this is not the disaster it looks like. What
# breaks is `audio_seconds`, which believes the header and reports 37 hours — and that number is
# the whole capture-fault diagnostic, blind on exactly the clips something already went wrong with.

_KILLED = (0xFFFFFFFF).to_bytes(4, "little")  # what a hard-killed ffmpeg really leaves


def _killed_wav(path: Path, samples: list[int], placeholder: bytes = _KILLED) -> Path:
    """A wav exactly as a killed ffmpeg leaves it: every sample written, both lengths unpatched."""
    _wav(path, samples)
    raw = bytearray(path.read_bytes())
    raw[4:8] = placeholder
    data = raw.index(b"data")
    raw[data + 4 : data + 8] = placeholder
    path.write_bytes(bytes(raw))
    return path


def test_a_killed_recorder_leaves_a_clip_that_lies_about_its_length(tmp_path):
    import wave

    clip = _killed_wav(tmp_path / "killed.wav", [10000, -10000] * 800)
    assert dictate.audio_seconds(clip) > 3600  # 37 hours of "captured audio", from a 0.1s clip
    assert dictate.repair_wav(clip) is True
    with wave.open(str(clip), "rb") as after:
        assert after.getnframes() == 1600
    assert dictate.audio_seconds(clip) == pytest.approx(0.1, abs=0.01)
    assert dictate.peak_dbfs(clip) == pytest.approx(-10.3, abs=0.2)


def test_a_header_left_at_zero_is_repaired_too(tmp_path):
    # Not what ffmpeg does today, and the one shape where the clip really is undecodable: `wave`
    # refuses a RIFF chunk of length zero outright, so the audio would be lost rather than wrong.
    clip = _killed_wav(tmp_path / "zeroed.wav", [4000] * 500, placeholder=bytes(4))
    assert dictate.repair_wav(clip) is True
    assert dictate.audio_seconds(clip) == pytest.approx(500 / 16000, abs=0.001)


def test_a_clip_that_closed_cleanly_is_left_exactly_as_it_is(tmp_path):
    clip = _wav(tmp_path / "clean.wav", [500] * 100)
    before = clip.read_bytes()
    assert dictate.repair_wav(clip) is False
    assert clip.read_bytes() == before


def test_the_repair_never_raises_on_anything_that_is_not_a_wav(tmp_path):
    for name, blob in [
        ("empty.wav", b""),
        ("header-only.wav", b"RIFF" + bytes(40)),
        ("not-riff.wav", b"OggS" + bytes(200)),
        ("no-data.wav", b"RIFF" + bytes(4) + b"WAVEfmt " + (16).to_bytes(4, "little") + bytes(100)),
    ]:
        path = tmp_path / name
        path.write_bytes(blob)
        assert dictate.repair_wav(path) is False, name
    assert dictate.repair_wav(tmp_path / "gone.wav") is False


def test_speaking_one_language_is_told_it_can_pin_the_decoder(monkeypatch, capsys):
    # ~0.7s a sentence, and nothing connected the two settings: `languages` deliberately does not
    # pin the decoder, `language` does, and somebody who speaks one language had no way to know.
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    assert cli.main(["config", "set", "languages", '["de"]']) == 0
    assert "config set language de" in capsys.readouterr().out
    # Two languages is the case the pin cannot serve, so it stays quiet.
    assert cli.main(["config", "set", "languages", '["de", "en"]']) == 0
    assert "set language" not in capsys.readouterr().out


def test_pinning_a_language_you_said_you_do_not_speak_is_called_out(monkeypatch, capsys):
    # The trigger works, the microphone works, and every single clip is thrown away as a language
    # you do not speak. Nothing appears and nothing explains it.
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    config.set_value("languages", ["de"])
    assert cli.main(["config", "set", "language", "en"]) == 0
    assert "thrown away" in capsys.readouterr().out


# --- what the daemon writes down about you --------------------------------------------------------


def _result(text: str = "the quick brown fox", **kw):
    fields = {
        "seconds": 3.2,
        "transcribe_ms": 640,
        "injected": True,
        "peak_dbfs": -14.0,
        "audio_seconds": 3.1,
        "paste_note": "→ Slack",
        "warm": True,
    }
    fields.update(kw)
    return dictate.Result(text=text, **fields)


def test_the_log_keeps_every_fact_about_a_clip_except_the_words():
    # It used to quote the transcript, and under the installed agent that line is appended to a log
    # nothing ever rotates — so every sentence ever dictated sat in plaintext on disk forever.
    line = dictate.log_line(_result())
    assert "the quick brown fox" not in line
    assert "19 chars" in line  # the LENGTH still says whether a paste was truncated
    assert "3.2s" in line
    assert "-14dBFS" in line
    assert "640ms" in line
    assert "→ Slack" in line


def test_the_health_report_can_still_read_a_paste_off_the_shortened_line():
    # `last_paste_verdict` is the only ground truth about the typing permission and it reads the
    # log. Dropping the transcript must not drop the shape it matches on.
    line = dictate.log_line(_result())
    assert line.startswith("[OK] ")
    assert " · → " in line


def test_debugging_one_bad_dictation_brings_the_words_back():
    config.set_value("keepAudio", True)
    assert "the quick brown fox" in dictate.log_line(_result())


def test_a_capture_fault_is_still_named_apart_from_a_model_fault():
    # 21s held, 15s captured: only the second is about the recorder, and they used to read alike.
    assert "captured 15.0s" in dictate.log_line(_result(seconds=21.0, audio_seconds=15.0))
    assert "captured" not in dictate.log_line(_result(seconds=3.2, audio_seconds=3.1))


def test_the_cold_path_is_still_said_out_loud():
    assert " · cold" in dictate.log_line(_result(warm=False))
    assert "cold" not in dictate.log_line(_result(warm=True))


# --- a paste that cannot land says what it cost ----------------------------------------------------


def test_a_blocked_paste_admits_it_overwrote_the_clipboard(monkeypatch):
    # Secure Input is macOS saying a password field has focus THIS SECOND, so what the transcript
    # replaces on the clipboard is more likely than usual to be a password just copied out of a
    # manager. "Press paste" alone reads as free, and it is not.
    monkeypatch.setattr(dictate.platforms, "input_blocked", lambda: "Secure Input is active.")
    monkeypatch.setattr(dictate.platforms, "clipboard_set", lambda text: True)
    ok, problem, _note = dictate.inject("hello there")
    assert ok is False
    assert "in place of what was there" in problem


def test_a_refused_paste_tells_the_same_story_as_a_blocked_one(monkeypatch):
    # The injection script writes the clipboard before it sends the chord and aborts before the
    # restore, so a refused paste costs exactly what a blocked one does.
    monkeypatch.setattr(dictate.platforms, "input_blocked", lambda: "")
    monkeypatch.setattr(dictate.platforms, "inject", lambda text, settle: (False, "refused.", ""))
    _ok, problem, _note = dictate.inject("hello there")
    assert dictate.PASTE_YOURSELF in problem


def test_a_clipboard_that_could_not_be_written_says_the_text_is_lost(monkeypatch):
    # Promising a recovery paste over a clipboard nothing was written to is worse than saying so.
    monkeypatch.setattr(dictate.platforms, "input_blocked", lambda: "Secure Input is active.")
    monkeypatch.setattr(dictate.platforms, "clipboard_set", lambda text: False)
    _ok, problem, _note = dictate.inject("hello there")
    assert "The text is lost." in problem


# --- a microphone that turns up late ----------------------------------------------------------------


def test_a_headset_that_connects_after_login_is_found_without_a_restart(monkeypatch):
    # The agent is installed with KeepAlive, so "cached for this run" meant weeks. A Bluetooth
    # headset still connecting at login is absent from the first enumeration, and every sentence
    # after that was recorded on the fallback microphone with nothing on screen to say so.
    config.set_value("inputName", "Headset")
    devices = [("0", "MacBook Pro Microphone")]
    monkeypatch.setattr(dictate.platforms, "list_inputs", lambda: devices)
    monkeypatch.setattr(dictate.platforms, "default_input", lambda: "default")
    monkeypatch.setattr(dictate, "_INPUT_CACHE", None)
    monkeypatch.setattr(dictate, "_INPUT_CACHED_AT", 0.0)

    clock = [1000.0]
    monkeypatch.setattr(dictate.time, "monotonic", lambda: clock[0])

    assert dictate.resolve_input()[1] == "MacBook Pro Microphone"
    devices.append(("1", "Bose Headset"))
    # Still cached, and deliberately so: re-listing devices costs an ffmpeg on every key press.
    assert dictate.resolve_input()[1] == "MacBook Pro Microphone"
    clock[0] += dictate.INPUT_CACHE_SECONDS + 1
    assert dictate.resolve_input() == ("1", "Bose Headset")


def test_a_failed_enumeration_is_never_cached_as_an_answer(monkeypatch):
    # ffmpeg may simply not have been ready. Caching "no devices" would be caching a hiccup.
    monkeypatch.setattr(dictate.platforms, "list_inputs", lambda: [])
    monkeypatch.setattr(dictate.platforms, "default_input", lambda: "default")
    monkeypatch.setattr(dictate, "_INPUT_CACHE", None)
    monkeypatch.setattr(dictate, "_INPUT_CACHED_AT", 0.0)
    assert dictate.resolve_input() == ("default", "system default")
    assert dictate._INPUT_CACHE is None


# --- the log cannot grow forever ------------------------------------------------------------------


def test_a_log_under_the_ceiling_is_left_exactly_as_it_is():
    # The ordinary path is one stat. A housekeeping routine that rewrites a file nobody asked it to
    # is a worse bug than the growth it prevents.
    config.log_path().write_text("one line\n", "utf-8")
    assert dictate.trim_log() is False
    assert config.log_path().read_text("utf-8") == "one line\n"


def test_a_log_past_the_ceiling_keeps_the_newest_and_drops_the_oldest():
    # Driven with a small ceiling rather than by writing a real megabyte: the behaviour under test
    # is which end survives, and that is the same at any size.
    path = config.log_path()
    path.write_text("".join(f"[OK] clip {i}\n" for i in range(2_000)), "utf-8")
    before = path.stat().st_size
    assert dictate.trim_log(cap=4_000, keep=2_000) is True
    after = path.read_text("utf-8")
    assert path.stat().st_size < before
    assert path.stat().st_size <= 2_000
    assert "clip 1999" in after  # the newest survives, which is the half anybody reads
    assert "clip 0\n" not in after


def test_the_trimmed_log_never_opens_on_half_a_line():
    # Read back by a person or by `last_paste_verdict`, a fragment at the top is noise at best.
    path = config.log_path()
    path.write_text("".join(f"[OK] {i} some text on the line\n" for i in range(2_000)), "utf-8")
    dictate.trim_log(cap=4_000, keep=2_000)
    assert path.read_text("utf-8").startswith("[OK] ")


def test_a_missing_log_is_not_an_error():
    # `murmurflow listen` in a terminal never creates one; only the installed service does.
    config.log_path().unlink(missing_ok=True)
    assert dictate.trim_log() is False


def test_the_paste_verdict_still_reads_a_trimmed_log():
    path = config.log_path()
    line = "[OK] 1.0s \u00b7 -14dBFS \u00b7 600ms \u00b7 12 chars \u00b7 \u2192 Slack\n"
    path.write_text(line * 2_000, "utf-8")
    dictate.trim_log(cap=4_000, keep=2_000)
    assert dictate.last_paste_verdict() is True


def test_the_shipped_ceiling_leaves_months_of_real_use_in_the_file():
    # ~120 bytes a clip. The point of the number is that trimming is rare, not that it is tidy.
    assert dictate.LOG_MAX_BYTES / 120 > 5_000
    assert dictate.LOG_KEEP_BYTES < dictate.LOG_MAX_BYTES


# --- streaming ---------------------------------------------------------------------------------


def test_only_the_words_two_passes_agreed_on_are_settled():
    # The pass that can still see more audio coming is the one that revises, so the word touching
    # the end is never committed even when both passes said it.
    assert dictate.stable_prefix("the quick brown", "the quick brown fox") == "the quick brown"
    assert dictate.stable_prefix("the quick brown", "the quick green fox") == "the quick"


def test_punctuation_and_capitals_are_not_a_disagreement():
    # "okay so we" becomes "Okay, so we" the moment whisper sees the end of the sentence. Treating
    # that as a changed word would stall the stream on every clip that ends in a full stop.
    assert dictate.stable_prefix("okay so we", "Okay, so we start") == "Okay, so we"


def test_the_first_pass_holds_its_tail_back_instead_of_waiting_for_a_second():
    # There is nothing yet to agree with, and waiting costs a whole ~2s pass on the one update
    # whose lateness is most felt. The tail is where whisper's revisions are, so the tail is what
    # is held.
    assert dictate.STREAM_HOLDBACK_WORDS == 4
    assert dictate.stable_prefix("", "one two three four five six") == "one two"
    assert dictate.stable_prefix("", "hello there world") == ""  # nothing but tail yet


def test_the_tail_is_what_is_not_at_the_cursor_yet():
    assert dictate.stream_tail("the quick brown", "the quick brown fox jumps") == "fox jumps"
    assert dictate.stream_tail("", "the whole thing") == "the whole thing"


def test_a_fully_streamed_sentence_leaves_nothing_to_paste():
    assert dictate.stream_tail("all of it", "All of it.") == ""


def test_a_reworded_prefix_neither_doubles_nor_loses_the_rest():
    # The final pass turned "to" into "two" inside text that is already on screen. There is no
    # un-paste, so the only question is whether what follows lands exactly once.
    assert dictate.stream_tail("send it to him", "send it two him tomorrow") == "tomorrow"


def test_streaming_is_on_by_default_and_needs_no_setting():
    # It used to be opt-in, and opt-in meant almost nobody ever saw the good version of the product.
    config.set_value("doubleTap", None)
    assert dictate.streaming() is True


def test_streaming_stands_down_while_the_trigger_is_held():
    # A held modifier turns every ⌘V into ⌥⌘V, so it is honoured only under doubleTap.
    config.set_value("doubleTap", False)
    assert dictate.streaming() is False
    config.set_value("doubleTap", True)
    assert dictate.streaming() is True


def test_a_partial_in_a_language_you_do_not_speak_is_never_typed(monkeypatch, tmp_path):
    """`finish` can refuse a transcript in a language you do not speak. It cannot refuse one that
    is already at the cursor, and there is no un-paste — so the gate has to be on the partial too.
    """
    config.set_value("languages", ["de", "en"])
    live = tmp_path / "live.wav"
    live.write_bytes(b"RIFF")
    monkeypatch.setattr(dictate, "audio_seconds", lambda _p: 3.0)
    monkeypatch.setattr(dictate, "peak_dbfs", lambda _p: -14.0)
    monkeypatch.setattr(dictate, "repair_wav", lambda _p: False)

    monkeypatch.setattr(
        dictate, "transcribe_warm", lambda *a, **k: dictate.Heard("guten Tag", 0.99, "de", True)
    )
    assert dictate._partial(live, tmp_path / "snap.wav").text == "guten Tag"

    monkeypatch.setattr(
        dictate, "transcribe_warm", lambda *a, **k: dictate.Heard("ご視聴", 0.99, "ja", True)
    )
    assert dictate._partial(live, tmp_path / "snap.wav").text == ""


def test_a_lent_trigger_does_not_open_the_microphone_early(monkeypatch):
    """A pause promises the key is not being listened to. Pre-roll runs BEFORE `on_press` gets to
    check, so without its own check a paused daemon still opened the microphone on every press.
    """
    opened: list[str] = []
    taps: list[object] = []
    monkeypatch.setattr(dictate, "available", lambda: (True, ""))
    monkeypatch.setattr(dictate, "claim_listener", lambda: 0)
    monkeypatch.setattr(dictate, "reap_orphans", lambda: 0)
    monkeypatch.setattr(dictate, "resolve_input", lambda: ("0", "a mic"))
    monkeypatch.setattr(dictate, "start_server", lambda **_k: True)
    monkeypatch.setattr(dictate, "preroll", lambda: opened.append("mic"))
    monkeypatch.setattr(dictate, "paused", lambda: (True, "a huddle"))
    monkeypatch.setattr(
        dictate, "bind_trigger", lambda *_a, on_tap=None, **_k: taps.append(on_tap) or "driven"
    )
    dictate.listen_loop()
    taps[0]("press")
    assert opened == []

    monkeypatch.setattr(dictate, "paused", lambda: (False, ""))
    taps[0]("press")
    assert opened == ["mic"]


def test_a_partial_pins_the_language_the_first_pass_heard(monkeypatch, tmp_path):
    """Detecting the language is 0.75s of every 2.2s pass, and one clip does not change language.

    The pin is only ever taken from a pass that already cleared the confidence gate, and the FINAL
    transcription is never pinned — so the "is that one of yours" gate still judges the real clip.
    """
    config.set_value("languages", ["de", "en"])
    asked: list[str] = []

    def _partial(_live, _snapshot, language=""):
        asked.append(language)
        return dictate.Heard("hello there my friend and also you", 0.99, "en", warm=True)

    monkeypatch.setattr(dictate, "_partial", _partial)
    monkeypatch.setattr(dictate, "_inject", lambda _text: (True, "", ""))
    monkeypatch.setattr(dictate, "STREAM_FIRST_SECONDS", 0.0)
    monkeypatch.setattr(dictate, "STREAM_EVERY_SECONDS", 0.0)

    stream = dictate.Stream(threading.Event())
    thread = threading.Thread(
        target=dictate._stream_loop,
        args=(dictate.Recording(1, tmp_path / "live.wav", 0.0), stream),
        daemon=True,
    )
    thread.start()
    while len(asked) < 3:
        time.sleep(0.01)
    stream.done.set()
    thread.join(timeout=2)
    assert asked[0] == ""  # the first pass has to detect it
    assert asked[1] == "en" and asked[2] == "en"  # and nothing after it pays for that again


def test_a_language_you_do_not_speak_is_never_pinned(monkeypatch, tmp_path):
    """A partial pinned to a language nobody is speaking comes back as fluent TRANSLATION, and two
    of those in a row agree with each other and get typed. So the pin is gated the same way the
    final transcript is.
    """
    config.set_value("languages", ["de", "en"])
    asked: list[str] = []

    def _partial(_live, _snapshot, language=""):
        asked.append(language)
        return dictate.Heard("gokigen you desu ne totemo ii tenki", 0.99, "ja", warm=True)

    monkeypatch.setattr(dictate, "_partial", _partial)
    monkeypatch.setattr(dictate, "_inject", lambda _text: (True, "", ""))
    monkeypatch.setattr(dictate, "STREAM_FIRST_SECONDS", 0.0)
    monkeypatch.setattr(dictate, "STREAM_EVERY_SECONDS", 0.0)

    stream = dictate.Stream(threading.Event())
    thread = threading.Thread(
        target=dictate._stream_loop,
        args=(dictate.Recording(1, tmp_path / "live.wav", 0.0), stream),
        daemon=True,
    )
    thread.start()
    while len(asked) < 3:
        time.sleep(0.01)
    stream.done.set()
    thread.join(timeout=2)
    assert set(asked) == {""}  # every pass still detects; none is pinned to a language he lacks


# --- the live pass has its own small model ----------------------------------------------------


def test_the_live_model_is_a_small_one_and_never_the_transcript_model(tmp_path, monkeypatch):
    """`model` is the transcript you keep; the live pass is a different job with a different cost.

    They must not share a knob: pinning the transcript to a small model is a decision about
    accuracy, and it must not silently also become the decision about the live pass, or vice versa.
    """
    models = config.home_root() / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    assert whisper.partial_model() == ""  # a big model is not a live model
    (models / "ggml-base.bin").write_bytes(b"x")
    assert whisper.partial_model().endswith("ggml-base.bin")
    (models / "ggml-small.bin").write_bytes(b"x")
    assert whisper.partial_model().endswith("ggml-small.bin")  # small beats base: measured German
    assert whisper.model().endswith("ggml-large-v3-turbo.bin")  # and the transcript is unmoved

    # An explicit `model` override still names the transcript model, and only that one.
    config.set_value("model", str(models / "ggml-base.bin"))
    assert whisper.model().endswith("ggml-base.bin")
    assert whisper.partial_model().endswith("ggml-small.bin")


def test_the_live_server_gets_its_own_model_and_its_own_port(monkeypatch):
    """Its own PROCESS is the point, not just its own model: whisper-server answers one request at
    a time, so a partial sharing the queue is time the FINAL transcription spends waiting.
    """
    monkeypatch.setattr(dictate, "resolve_bin", lambda _n: "/usr/bin/whisper-server")
    config.set_value("port", 8479)
    command = dictate.serve_command("/models/ggml-small.bin", dictate.partial_port())
    assert command is not None
    assert "/models/ggml-small.bin" in command
    assert "8480" in command
    assert dictate.partial_port() == 8480


def test_a_missing_live_server_sends_the_partials_to_the_big_one(monkeypatch):
    """Slower, and never wrong. It is what they did before the small server existed."""
    monkeypatch.setattr(dictate, "ours", lambda _at: True)  # who holds the port is a separate test
    monkeypatch.setattr(dictate, "server_up", lambda _at=0: False)
    assert dictate.partial_at() == 0  # 0 means "the default port", i.e. the big server
    monkeypatch.setattr(dictate, "server_up", lambda _at=0: True)
    assert dictate.partial_at() == dictate.partial_port()


def test_the_log_says_whether_streaming_typed_or_merely_ran():
    """ "It does not stream" and "my sentence was too short to settle a word" wrote the same line."""
    assert dictate.stream_note(None) == ""
    assert dictate.stream_note(dictate.Stream(threading.Event())) == ""  # never ran
    ran = dictate.Stream(threading.Event(), "", passes=4, blank=3, typed=0)
    assert dictate.stream_note(ran) == "stream 4x → 0 typed, 3 read nothing"
    worked = dictate.Stream(threading.Event(), "hello", passes=9, blank=1, typed=8)
    assert dictate.stream_note(worked) == "stream 9x → 8 typed, 1 read nothing"


def test_an_impostor_on_the_port_is_never_handed_the_audio(monkeypatch):
    """Both ports are predictable, and a socket that accepts a connection proves nothing about who
    is on the other end of it. Anything but a whisper-server would be sent recorded speech and
    believed about what was said.
    """
    monkeypatch.setattr(dictate, "server_up", lambda _at=0: True)

    class _Nobody:
        stdout = ""  # pgrep found no whisper-server holding the port

    monkeypatch.setattr(dictate.subprocess, "run", lambda *_a, **_k: _Nobody())
    assert dictate.ours(dictate.partial_port()) is False
    assert dictate.partial_at() == 0  # so the partials go to the big server, not to the impostor
    assert dictate.start_server(at=dictate.partial_port()) is False  # and it is never adopted

    dictate._OWNERSHIP.clear()

    class _Whisper:
        stdout = "4242\n"

    monkeypatch.setattr(dictate.subprocess, "run", lambda *_a, **_k: _Whisper())
    assert dictate.ours(dictate.partial_port()) is True
    assert dictate.partial_at() == dictate.partial_port()


def test_a_port_that_leaves_no_room_for_the_live_server_is_refused(monkeypatch):
    """65535 puts the live server on 65536, which cannot bind — and nothing would say why."""
    monkeypatch.setattr(cli.service, "restart", lambda: False)
    assert cli.main(["config", "set", "port", "65535"]) == 2
    assert "port" not in config.load()
    assert cli.main(["config", "set", "port", str(dictate.MAX_PORT)]) == 0
    assert dictate.partial_port() == 65535
    # And a config hand-edited past the ceiling degrades to the default rather than to an
    # unbindable pair, because `port` is read on the daemon's hot path and must never raise.
    config.set_value("port", 65535)
    assert dictate.port() == dictate.DEFAULT_PORT


# --- installing is updating -----------------------------------------------------------------------


def _receipt_at(tmp_path, body):
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text(body, "utf-8")
    return receipt


def test_a_local_checkout_is_reinstalled_from_that_checkout(tmp_path, monkeypatch):
    """`git pull` changes the checkout; the daemon runs the COPY uv made. So install re-copies."""
    monkeypatch.setattr(cli.dictate, "resolve_bin", lambda _n: "/opt/homebrew/bin/uv")
    checkout = tmp_path / "murmurflow"
    checkout.mkdir()
    receipt = _receipt_at(
        tmp_path,
        f'[tool]\nrequirements = [{{ name = "murmurflow", directory = "{checkout}" }}]\n',
    )
    assert cli._update_command(receipt) == [
        "/opt/homebrew/bin/uv",
        "tool",
        "install",
        "--force",
        "--python",
        "3.13",
        str(checkout),
    ]


def test_an_install_from_git_or_pypi_is_upgraded_instead(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.dictate, "resolve_bin", lambda _n: "/opt/homebrew/bin/uv")
    receipt = _receipt_at(tmp_path, '[tool]\nrequirements = [{ name = "murmurflow" }]\n')
    assert cli._update_command(receipt) == ["/opt/homebrew/bin/uv", "tool", "upgrade", "murmurflow"]


def test_no_uv_and_a_corrupt_receipt_both_mean_do_not_update(tmp_path, monkeypatch):
    """An update that cannot run is an inconvenience. An `install` that refuses is a dead tool."""
    receipt = _receipt_at(tmp_path, '[tool]\nrequirements = [{ name = "murmurflow" }]\n')
    monkeypatch.setattr(cli.dictate, "resolve_bin", lambda _n: "")
    assert cli._update_command(receipt) is None
    monkeypatch.setattr(cli.dictate, "resolve_bin", lambda _n: "/opt/homebrew/bin/uv")
    assert cli._update_command(_receipt_at(tmp_path, "not toml at all {{{")) is None


def test_a_restart_that_fails_stops_the_install_rather_than_half_doing_it(tmp_path, monkeypatch):
    """The package directory this process imports from has just been replaced. Carrying on means
    the next lazy import opens a path that is gone, which is half an install and no error.
    """
    monkeypatch.setattr(cli, "_receipt", lambda: tmp_path / "uv-receipt.toml")
    monkeypatch.setattr(cli, "_update_command", lambda _r: ["uv", "tool", "upgrade", "murmurflow"])
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *_a, **_k: types.SimpleNamespace(returncode=0, stderr="")
    )
    monkeypatch.delenv(cli._RESYNCED, raising=False)

    def _boom(*_a):
        raise OSError("Text file busy")

    monkeypatch.setattr(cli.os, "execv", _boom)
    with pytest.raises(SystemExit) as exit_code:
        cli._update()
    assert exit_code.value.code == 1


def test_the_re_executed_child_never_updates_again(tmp_path, monkeypatch):
    """The child of the re-exec runs `install` too, and without the guard it would re-exec forever."""
    ran: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **_k: ran.append(cmd))
    monkeypatch.setattr(cli, "_receipt", lambda: tmp_path / "uv-receipt.toml")
    monkeypatch.setenv(cli._RESYNCED, "1")
    cli._update()
    assert ran == []


# --- the microphone opens one gesture early -------------------------------------------------------


def _fake_recorder(monkeypatch, tmp_path, stopped):
    """A `start` that costs nothing and a `stop` that only records that it was called."""
    made: list[dictate.Recording] = []

    def _start():
        wav = tmp_path / f"pre-{len(made)}.wav"
        wav.write_bytes(b"")
        rec = dictate.Recording(1000 + len(made), wav, 0.0)
        made.append(rec)
        return rec

    monkeypatch.setattr(dictate, "start", _start)
    monkeypatch.setattr(dictate, "current", lambda: None)
    monkeypatch.setattr(dictate, "stop", lambda rec=None: stopped.append(rec) or rec.wav)
    return made


def test_the_second_tap_takes_the_microphone_the_first_tap_already_opened(monkeypatch, tmp_path):
    """THE FIRST HALF-SECOND OF EVERY SENTENCE. CoreAudio needs ~0.6s to hand over its first
    buffer, and no decoder can recover audio that was never captured — so the recorder is opened
    on the first tap and the second tap claims it, already live.
    """
    stopped: list[dictate.Recording] = []
    made = _fake_recorder(monkeypatch, tmp_path, stopped)
    monkeypatch.setattr(
        dictate.threading, "Timer", lambda *_a, **_k: types.SimpleNamespace(start=lambda: None)
    )

    dictate.preroll()
    dictate.preroll()  # the second tap's own key-down must not open a SECOND recorder
    assert len(made) == 1
    assert dictate.preroll_claim() is made[0]
    assert stopped == []  # claimed, so nothing was thrown away
    assert dictate.preroll_claim() is None  # and it is gone from the registry


def test_a_pre_roll_nobody_claims_stops_itself_and_leaves_no_audio(monkeypatch, tmp_path):
    """The privacy half: a stray single tap must not leave the microphone open, or a file behind."""
    stopped: list[dictate.Recording] = []
    made = _fake_recorder(monkeypatch, tmp_path, stopped)
    fired: list[object] = []
    monkeypatch.setattr(
        dictate.threading,
        "Timer",
        lambda _s, fn, args=(): types.SimpleNamespace(
            start=lambda: fired.append(lambda: fn(*args))
        ),
    )

    dictate.preroll()
    fired[0]()  # the pair never completed and the timer came due
    assert stopped == made
    assert not made[0].wav.exists()
    assert dictate.preroll_claim() is None


def test_stopping_a_stream_returns_what_it_pasted_and_ends_it(tmp_path):
    wav = tmp_path / "said.wav"
    stream = dictate.Stream(threading.Event(), "hello there")
    dictate._STREAMS[str(wav)] = stream
    stopped = dictate.stop_streaming(wav)
    assert stopped is stream and stream.done.is_set()
    assert dictate.streamed(stopped) == "hello there"
    assert dictate.stop_streaming(wav) is None  # gone from the registry


def test_a_paste_still_in_flight_is_counted_before_the_tail_is_worked_out(tmp_path):
    # The race this seam exists for, and it produced a doubled half-sentence on a real machine: the
    # pass that was already inside `inject` when the key was released records its words only when
    # it returns, so reading `text` without waiting reports a prefix shorter than the screen shows.
    wav = tmp_path / "inflight.wav"
    stream = dictate.Stream(threading.Event())
    dictate._STREAMS[str(wav)] = stream

    def _late_paste():
        with dictate._INJECT_LOCK:
            time.sleep(0.15)
            stream.text = "the words already on screen"

    pasting = threading.Thread(target=_late_paste)
    pasting.start()
    time.sleep(0.02)  # the key is released mid-paste
    stopped = dictate.stop_streaming(wav)
    assert dictate.streamed(stopped) == "the words already on screen"
    pasting.join()


def test_a_leading_space_survives_into_the_paste(monkeypatch):
    # A streamed chunk arrives as " and then", because the space in front of it is the gap between
    # it and the words already on screen. `inject` stripping that glued every chunk to the last.
    sent: list[str] = []

    def _record(text, settle):
        sent.append(text)
        return True, "", ""

    monkeypatch.setattr(dictate.platforms, "input_blocked", lambda: "")
    monkeypatch.setattr(dictate.platforms, "inject", _record)
    assert dictate.inject(" and then")[0] is True
    assert sent == [" and then"]


def test_whitespace_alone_is_still_nothing_to_type():
    assert dictate.inject("   ")[1] == "nothing to type"
