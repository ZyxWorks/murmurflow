"""``murmurflow`` — the command line. Ten verbs, and most people only ever type two.

``setup`` then ``install`` is the whole happy path. Everything else here exists because dictation
fails in exactly four ways — the key is not seen, the microphone is not heard, the model is not
found, the text is not typed — and each of those has its own verb that answers it in one run.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, dictate, hotkey, platforms, service, whisper


def _out(line: str = "") -> None:
    print(line, flush=True)


# --- setup ------------------------------------------------------------------------------------


def _setup(name: str = "") -> int:
    """Download a ggml speech model into ``~/.murmurflow/models/``.

    The model is the single biggest lever on accuracy — bigger models are the difference between
    proper nouns transcribed and proper nouns guessed at — so the default is the big one, and the
    download is the only part of installation that takes real time.
    """
    model = name or whisper.DEFAULT_MODEL
    if not model.startswith("ggml-") or not model.endswith(".bin"):
        model = f"ggml-{model}.bin"
    if model not in whisper.MODEL_PREFERENCE:
        _out(f"unknown model: {model}")
        _out("choose one of: " + ", ".join(m[5:-4] for m in whisper.MODEL_PREFERENCE))
        return 2
    target = config.home_root() / "models" / model
    if target.is_file():
        _out(f"[OK] already have {model} ({target.stat().st_size / 1e6:.0f} MB)")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{whisper.MODEL_BASE_URL}/{model}"
    _out(f"downloading {model}")
    _out(f"  from {url}")
    tmp = target.with_suffix(".bin.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(
                            f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB  {pct:5.1f}%",
                            end="",
                            flush=True,
                        )
        print()
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        _out(f"[!] download failed: {exc}")
        return 1
    tmp.replace(target)  # atomic: a half-downloaded model must never look like a usable one
    _out(f"[OK] {target}")
    # The warm whisper-server loaded whichever model was best WHEN IT STARTED and answers with
    # that one until something stops it — so downloading a better model would otherwise change
    # nothing at all until the next reboot, which reads as "I got the big model and it is still
    # guessing at names".
    if service.running() and dictate.stop_server():
        service.restart()
        _out("[OK] restarted the listener so it transcribes with this model now")
    return 0


def _install() -> int:
    """Warm the microphone, then install the launchd agent so dictation is live after every login."""
    ready, hint = dictate.available()
    if not ready:
        _out(hint)
        return 2
    if not service.supported():
        _out(f"murmurflow has no always-on listener for {sys.platform} yet — see the README")
        return 2

    # The FIRST ever CoreAudio access on a Mac takes ~10 seconds. Paying it here, explicitly, means
    # the user's first real sentence is fast instead of looking broken. Windows has no such tax,
    # and the warm-up is harmless there.
    _out("warming the microphone (the first access can take ~10s, once)...")
    rec = dictate.start()
    if rec is not None:
        dictate.ready(rec, timeout=15.0)
        wav = dictate.stop(rec)
        if wav is not None:
            wav.unlink(missing_ok=True)
        _out("[OK] microphone ready")
    else:
        _out("[!] could not open the microphone — grant Microphone access and re-run")

    ok, detail = service.install()
    _out(f"[OK] installed {service.LABEL}" if ok else f"[!] could not install it: {detail}")
    _out("")
    # Installing is the exact moment a second daemon joins the key, so it is the moment to say so.
    # Silence here costs the user a session of "it worked yesterday" before anyone runs the health
    # report that already knew.
    for name, pids, fix in dictate.rival_listeners():
        _out(f"[!] {name} is listening on the same key ({', '.join(str(p) for p in pids)}).")
        _out(f"    Both will chime and both will type. Switch one off: {fix}")
        _out("")
    _out(f"{dictate.trigger_hint()} anywhere and talk. Release: the text types itself.")
    _out("")
    _out("macOS will ask for two permissions the first time. Neither can be granted from a script,")
    _out("and both are required:")
    _out("  1. Microphone      — to hear you")
    _out("  2. Accessibility   — to type into the app you are using")
    _out("")
    _out("The apps you dictate INTO need nothing. Every permission goes to this one program,")
    _out("which System Settings lists as:")
    _out(f"  {_tcc_entry()}")
    # Opening the exact pane, rather than naming a path through System Settings, is the difference
    # between "one switch is highlighted for you" and a person who has never opened Privacy &
    # Security hunting for it. Best-effort: a Mac that refuses the URL still has the sentence above.
    _out("")
    # Only where there is something to grant. On Windows there is no such pane, and sending
    # somebody to look for one is worse than saying nothing.
    if service.is_macos():
        _out("Opening the Accessibility pane now — switch that entry ON.")
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [
                    "open",
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                ],
                timeout=10,
                check=False,
            )
    return 0 if ok else 1


def _uninstall() -> int:
    ok, detail = service.uninstall()
    # The warm whisper-server is detached on purpose and would otherwise sit on ~1.8 GB until the
    # next reboot, long after the thing that talked to it was removed.
    freed = dictate.stop_server()
    # And the .app, or `uninstall` leaves an application in ~/Applications forever. Recreating it
    # later at the same path from the same interpreter reproduces the cdhash, so the Privacy grant
    # is not spent by removing it.
    removed = service.remove_identity()
    _out("[OK] dictation stopped and removed from login." if ok else f"[!] {detail}")
    if freed:
        _out("[OK] stopped the warm whisper-server")
    if removed:
        _out("[OK] removed the MurmurFlow.app bundle")
    return 0 if ok else 1


# --- diagnosis --------------------------------------------------------------------------------


def _tcc_entry() -> str:
    """What System Settings CALLS this program in its permission lists, and where that file is.

    macOS names a permission row after the EXECUTABLE that asked, so the listener is installed
    inside its own bundle and the row reads MurmurFlow — but an install that could not write the
    bundle still has to tell the truth, and the truth there is the interpreter's own name,
    ``python3.13``, which nobody scrolling for ``murmurflow`` would ever find.
    """
    named = service.identity()
    if named:
        return named
    real = Path(sys.executable).resolve()
    return f"{real.name}  ({real})"


def _typing_permission() -> tuple[bool, str]:
    """Does the thing that does the typing have permission to — and how do we know.

    Asked in order of how much the answer is worth. What the DAEMON's last paste actually did beats
    any probe: see :func:`dictate.last_paste_verdict` for why asking the OS from here answers about
    the wrong process and reported NOT GRANTED on a Mac that was typing dictations all afternoon.
    Only when the daemon has never tried does this fall back to asking the bundle, and then to
    asking this CLI about itself, which is the weakest answer of the three and still better than a
    blank row.
    """
    landed = dictate.last_paste_verdict()
    if landed is not None:
        return landed, (
            "granted — the listener's last paste landed"
            if landed
            else "NOT granted — the listener's last paste was refused"
        )
    trusted = service.permission_trusted()
    if trusted is None:
        trusted = hotkey.accessibility_trusted()
    return bool(trusted), "granted" if trusted else "NOT granted — nothing can be typed"


def _doctor(*, verbs: bool = False) -> int:
    """Everything that has to be true, each with the one command that makes it true.

    Ordered the way it actually fails: no recorder, no transcriber, no model, no permission.
    """
    rows: list[tuple[bool, str, str]] = []
    ffmpeg = dictate.resolve_bin("ffmpeg")
    rows.append((bool(ffmpeg), f"recorder: {ffmpeg or 'ffmpeg NOT FOUND'}", "brew install ffmpeg"))
    server = dictate.resolve_bin("whisper-server")
    binary = whisper.found_binary()
    rows.append(
        (
            bool(server or binary),
            f"transcriber: {server or binary or 'NOT FOUND'}"
            + ("" if server else "  (no whisper-server: every clip pays ~1s model load)"),
            "brew install whisper-cpp",
        )
    )
    model = whisper.model()
    rows.append((bool(model), f"model: {model or 'NOT FOUND'}", "murmurflow setup"))
    # A warm server is the difference between the first sentence after a boot taking ~1s and
    # taking ~13s, and it is invisible either way — the tool just feels slow. Worth its own row
    # for the second reason too: a server that is up but WEDGED answers every request with an
    # error, so every clip silently takes the cold path where the language and confidence gates
    # are weaker. The daemon bounces one now (see `dictate.COLD_CLIPS_BEFORE_RESTART`); this is
    # where you see whether it is currently there at all.
    if server:
        warm = dictate.server_up()
        rows.append(
            (
                warm,
                "warm server: "
                + (
                    f"answering on :{dictate.port()}"
                    if warm
                    else f"nothing on :{dictate.port()} — every clip pays the model load again"
                ),
                "murmurflow install   (the listener starts and keeps it warm)",
            )
        )
    rows.append(
        (
            hotkey.available(),
            "key polling: " + ("available" if hotkey.available() else "BLOCKED"),
            hotkey.unavailable_reason() or "run: murmurflow keytest",
        )
    )
    # THE ROW PRINTS ON EVERY PLATFORM, INCLUDING THE ONE WITH NOTHING TO GRANT. Windows has no
    # TCC: no Accessibility entry, no identity to defend, no grant to lose to a reinstall. The
    # tempting thing is to drop the row there, and it is wrong — a check that is present on one
    # machine and absent on another reads as a check somebody forgot to write, and the person
    # reading it is a client wondering what else is missing. It says "not required" instead.
    #
    # On macOS the daemon is the .app bundle, so the BUNDLE is who this has to be asked about:
    # this CLI having the grant says nothing about whether the thing that types your words does.
    needed = bool(dictate.permission_hint())
    if not needed:
        rows.append((True, f"typing permission: not required on {platforms.NAME}", ""))
    else:
        trusted, evidence = _typing_permission()
        rows.append(
            (
                trusted,
                f"accessibility: {evidence}",
                f"switch on '{_tcc_entry()}' in System Settings > Privacy & Security > "
                "Accessibility",
            )
        )
    clash = dictate.apple_dictation_conflict()
    rows.append(
        (
            not clash,
            "system dictation: "
            + (
                "not competing"
                if not clash
                else "ON — it fires on the same key and will fight this"
            ),
            "System Settings > Keyboard > Dictation > Shortcut > Off" if clash else "",
        )
    )
    _, device = dictate.resolve_input()
    rows.append((True, f"microphone: {device}", ""))
    live = dictate.listener_pids()
    rows.append(
        (
            len(live) <= 1,
            "listeners: "
            + (
                f"{len(live)} ({', '.join(str(pid) for pid in live) or 'none — dictation is off'})"
                if len(live) <= 1
                else f"{len(live)} AT ONCE ({', '.join(str(pid) for pid in live)}) "
                "— every sound and every sentence doubles"
            ),
            "murmurflow uninstall, then murmurflow install",
        )
    )
    # A murmurflow-only count reads "listeners: 1 [OK]" on a Mac where a SECOND program holds the
    # same key — which is the setup the user actually hears: two chimes in two different presets,
    # one sentence pasted twice. Reinstalling murmurflow cannot fix that, so it is its own row
    # with its own fix.
    rivals = dictate.rival_listeners()
    rows.append(
        (
            not rivals,
            "other dictation: "
            + (
                "nothing else on this key"
                if not rivals
                else ", ".join(
                    f"{name} IS LISTENING TOO ({', '.join(str(pid) for pid in pids)})"
                    for name, pids, _ in rivals
                )
                + " — every sound and every sentence doubles"
            ),
            "; ".join(fix for _, _, fix in rivals),
        )
    )
    rows.append(
        (
            not dictate.secure_input_active(),
            "secure input: "
            + ("off" if not dictate.secure_input_active() else "ON — pasting is blocked"),
            "quit whatever has a password field focused",
        )
    )
    # A LENT KEY MUST NEVER BE AN INVISIBLE SILENCE. This is the one state where everything above
    # is green and the trigger still does nothing, so it is the one state a health check exists for.
    # It reads as OK rather than as a fault: somebody asked for it, and it expires by itself.
    lent, holder = dictate.paused()
    rows.append(
        (
            True,
            "trigger: " + (f"LENT to {holder}" if lent else "yours"),
            "murmurflow resume" if lent else "",
        )
    )
    installed = service.installed()
    rows.append(
        (
            installed,
            f"login agent: {'installed' if installed else 'not installed'}"
            + (f", {'running' if service.running() else 'not loaded'}" if installed else ""),
            "murmurflow install",
        )
    )
    for ok, line, fix in rows:
        _out(f"  {'[OK]' if ok else '[!] '} {line}")
        if not ok and fix:
            _out(f"       fix: {fix}")
    _out("")
    _out(f"  config: {config.config_path()}")
    # "It did not work" has one answer and it is this file: every clip's hold, level, transcribe
    # time, transcript and the app the paste went to is already written there. Nothing said so.
    _out(f"  log: {config.log_path()}")
    _out(f"  trigger: {dictate.trigger_hint()}")
    _out(f"  cue: {dictate.cue_preset_name()}")
    polish = str(config.load().get("polishCommand", "") or "")
    _out(f"  polish: {polish or 'off (deterministic cleanup only)'}")
    if verbs:
        _verbs()
    return 0 if all(ok for ok, _, _ in rows) else 1


#: The bare `murmurflow` prints these under the health report. Health answers "is it working";
#: this answers "and what do I type next", which is the question that actually followed — a user
#: who wanted a different tone had no way to discover `config set cue` from a list of green ticks.
_VERBS = (
    ("config set cue pebble", "change the sound: pebble · glass · marimba · soft · system · off"),
    ("cues", "play every tone so you can pick one"),
    (
        "config set trigger control_option",
        "change the key: control_option · command_option · left_control · f13",
    ),
    ("config set doubleTap true", "tap twice to start and once to stop, instead of holding"),
    ("config set language en", "pin the language — worth ~0.7s a sentence"),
    ("config set stripFillers true", "delete 'um' and a leading 'hey' — off, so you get verbatim"),
    ("config", "every setting, with what it does"),
    ("keytest", "does this Mac see your key, and does it read your gesture the way you think"),
    ("devices", "list microphones (then: config set inputName <part of a name>)"),
    ("pause / resume", "lend the trigger key to another program, and take it back"),
    ("install / uninstall", "turn dictation on or off for every login"),
    ("--help", "everything else"),
)


def _verbs() -> None:
    width = max(len(verb) for verb, _ in _VERBS)
    _out("")
    _out("  what you can change:")
    for verb, what in _VERBS:
        _out(f"    murmurflow {verb.ljust(width)}  {what}")


def _devices() -> int:
    devices = dictate.list_inputs()
    if not devices:
        _out("no audio inputs found (is ffmpeg installed?)")
        return 1
    _, chosen = dictate.resolve_input()
    for index, name in devices:
        _out(f"  {'->' if name == chosen else '  '} [{index}] {name}")
    _out("")
    _out('pick one with: murmurflow config set inputName "<part of the name>"')
    return 0


def _keytest(*, trigger: str = "", seconds: float = 20.0) -> int:
    """Does this machine SEE the key, and does it read the gesture the way you think it does?

    With no ``--trigger`` it watches EVERY bindable key at once. That is the only question worth
    asking when the listener reacts to nothing, because never-read, taps-too-slow and
    chord-guarded are indistinguishable from the outside and have completely different fixes. One
    run of this separates them.

    Watching the side-agnostic modifier flags AND the side-specific keycodes side by side is what
    reveals the hardware truth: on some Macs pressing the RIGHT Command lights up ``left_command``,
    because that machine never reports the right-side keycode at all.
    """
    if not hotkey.available():
        _out("cannot read key state on this machine (is this macOS?)")
        return 1
    watching = [trigger] if trigger else sorted(hotkey.trigger_names())
    gesture = "double-tap" if dictate.double_tap_mode() else "hold"
    _out(f"press any of these now — watching {len(watching)} key(s) for {seconds:.0f}s.")
    _out(f"your trigger is {dictate.trigger_key()}, your gesture is {gesture}. ctrl-c to stop.")
    _out("  (a 'command'/'option' hit means EITHER side; 'left_*'/'right_*' means that side)")
    held: dict[str, float] = {}
    presses: dict[str, list[float]] = {name: [] for name in watching}
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            for name in watching:
                down = hotkey.is_trigger_down(name)
                if down and name not in held:
                    held[name] = now
                elif not down and name in held:
                    length = now - held.pop(name)
                    presses[name].append(length)
                    kind = "tap" if length <= hotkey.TAP_MAX else "hold"
                    _out(f"[OK] {name}  {kind}  ({length * 1000:.0f}ms)")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    seen = {name: taps for name, taps in presses.items() if taps}
    if not seen:
        _out("[!] no key was seen going down at all.")
        _out("    if you WERE pressing one, grant Input Monitoring to the process running this")
        _out("    (this terminal now; the murmurflow agent once installed) in")
        _out("    System Settings > Privacy & Security > Input Monitoring.")
        _out("    The apps you dictate INTO never need any permission.")
        return 1
    for name, taps in seen.items():
        quick = [t for t in taps if t <= hotkey.TAP_MAX]
        _out(
            f"[OK] {name}: {len(taps)} press(es), {len(quick)} short enough to count as a tap "
            f"(a pair within {hotkey.DOUBLE_TAP_WINDOW:.2f}s starts a turn)"
        )
        if taps and not quick:
            _out(
                f"     every press was longer than {hotkey.TAP_MAX:.2f}s — in double-tap mode that "
                "is a HOLD and can never open a pair. Tap faster, or turn doubleTap off."
            )
    _out("")
    _out("the keys above are seen by this machine; anything not listed was never pressed.")
    return 0


def _cues(name: str = "") -> int:
    """Play the three tones, so "which sound am I hearing" is answerable without reading config."""
    if name:
        config.set_value("cue", name)
        _out(f"cue set to {name}")
    _out(f"preset: {dictate.cue_preset_name()}")
    _out(f"available: {', '.join(dictate.cue_presets())}, system, off")
    for kind in (dictate.CUE_READY, dictate.CUE_DONE, dictate.CUE_FAIL):
        _out(f"  {kind}")
        dictate.cue(kind)
        time.sleep(0.7)
    _out("")
    _out(f"your own sounds: drop ready/done/fail files into {dictate.custom_cue_dir()}")
    return 0


# --- settings ---------------------------------------------------------------------------------


def _pause(seconds: float, who: str) -> int:
    """Lend the trigger key. Prints the deadline, because a pause with no visible end is a bug.

    This exists for the program that wants the SAME double-tap for something else — a voice
    assistant taking a turn, a screen recorder that must not have the microphone pulled out from
    under it. The listener stays up and the whisper server stays warm; only the trigger stands down.
    """
    until = dictate.pause(seconds, who=who)
    stamp = time.strftime("%H:%M:%S", time.localtime(until))
    _out(
        f"[OK] the trigger is lent out until {stamp}. It comes back on its own; `resume` is sooner."
    )
    return 0


def _resume() -> int:
    if dictate.resume():
        _out("[OK] the trigger is yours again.")
        return 0
    _out("the trigger was not lent out.")
    return 0


#: Settings the running whisper-server BAKED IN at the moment it started, so changing one of them
#: has to stop the server as well. Restarting the listener alone cannot do it: the daemon comes
#: back, finds the old server still answering on the port, reuses it by design, and the setting
#: the user just changed never takes effect anywhere.
_SERVER_SETTINGS = frozenset({"model", "port"})


def _reject(key: str, value: object) -> str:
    """Why writing ``value`` to ``key`` would not do what was just asked for, or ``""``.

    **Every way of misconfiguring this daemon fails silently, and they all fail the same way.** A
    trigger name this machine cannot poll binds a key that never fires. A cue name that is not a
    preset quietly falls back to a different sound. A misspelt setting is written to a file nothing
    ever reads. From the outside all three are "I hold the key and nothing happens", none of them
    leaves a trace anywhere, and each costs an evening to find. The typo is the only moment any of
    it is cheap to catch, so it is caught here and the write is refused.
    """
    if key not in config.KEYS:
        near = difflib.get_close_matches(key, list(config.KEYS), n=1)
        return (
            f"unknown setting `{key}`."
            + (f" Did you mean `{near[0]}`?" if near else "")
            + " `murmurflow config` lists every one."
        )
    text = str(value).strip()
    if key == "trigger":
        if hotkey.canonical_trigger(text) not in hotkey.trigger_names():
            return (
                f"`{text}` is not a key {platforms.NAME} can poll, so the listener would bind a "
                f"trigger that never fires. Pick one of: "
                f"{', '.join(sorted(hotkey.trigger_names()))}"
            )
    elif key == "cue":
        accepted = {*dictate.cue_presets(), "system", *dictate.CUE_OFF}
        if text.lower() not in accepted and not Path(text).expanduser().is_dir():
            # OFFERED is not the same set as ACCEPTED: `none`/`mute`/`silent`/`false`/`0` all mean
            # off and all keep working, but listing five spellings of one answer makes the four
            # real choices harder to see.
            offered = (*dictate.cue_presets(), "system", "off")
            return (
                f"`{text}` is not a cue, and an unknown one plays a different sound instead of "
                f"saying so. Pick one of: {', '.join(offered)} — or a folder holding your own "
                "ready/done/fail sound files."
            )
    elif key in {"doubleTap", "stripFillers", "keepAudio"}:
        if not isinstance(value, bool):
            return f"`{key}` is true or false, not `{text}`."
    elif key == "quietFloor":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"`{key}` is a peak level in dBFS below zero, e.g. -30 or -40, not `{text}`."
    elif key == "port":
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            return f"`{key}` is a TCP port between 1 and 65535, not `{text}`."
    elif key == "language":
        code = dictate.language_code(text)
        if code != "auto" and not (len(code) == 2 and code.isalpha()):
            return (
                f"`{text}` is not a language. Use `auto`, or a two-letter code like `en` or `de` "
                "— pinning the WRONG one makes whisper translate instead of transcribe."
            )
    elif key == "languages":
        entries = value if isinstance(value, list) else text.split(",")
        bad = [str(v).strip() for v in entries if len(dictate.language_code(str(v))) != 2]
        if bad:
            return (
                f"`{', '.join(bad)}` is not a language code, and a clip in a language that is not "
                "on this list is thrown away. Write it as a JSON list of two-letter codes: "
                '\'["de", "en"]\''
            )
    elif key == "model":
        # A bare NAME is legal here — an openai-whisper CLI downloads its own weights by name (see
        # `whisper.openai_model_name`). Only something shaped like a PATH is checked, because that
        # one belongs to whisper.cpp and a typo in it silently keeps the old model.
        looks_like_path = "/" in text or text.endswith(".bin")
        if looks_like_path and not Path(text).expanduser().is_file():
            return (
                f"no model file at {Path(text).expanduser()} — the best model in "
                f"{config.home_root() / 'models'} would go on being used and nothing would change."
            )
    return ""


def _warn(key: str, value: object) -> str:
    """A setting that is legal but probably not what was meant. Said out loud, never refused.

    The line between this and :func:`_reject` is whether we can be sure. An unplugged headset is a
    perfectly reasonable thing to name in ``inputName`` before plugging it back in, so refusing it
    would be wrong; saying nothing while dictation quietly records the wrong microphone is worse.
    """
    text = str(value).strip()
    if key == "inputName" and text:
        known = any(text.lower() in name.lower() for _id, name in dictate.list_inputs())
        if not known:
            return (
                f"no microphone here has `{text}` in its name (`murmurflow devices` lists them), "
                "so recording falls back to the system default until one does."
            )
    if key == "languages":
        # Somebody who speaks ONE language into this is leaving ~0.7s a sentence on the table and
        # has no way to know it: `languages` deliberately does not pin the decoder, `language`
        # does, and nothing connects the two. Said once, here, rather than made automatic — pinning
        # behind somebody's back is how German speech comes back as an English paraphrase.
        spoken = dictate.spoken_languages()
        if len(spoken) == 1 and whisper.language() == "auto":
            only = next(iter(spoken))
            return (
                f"you speak only {only}, so `murmurflow config set language {only}` pins the "
                "decoder and saves about 0.7s on every sentence."
            )
    if key == "language" and text.lower() != "auto":
        # Pin `en`, then have `["de"]` in `languages`, and EVERY clip is thrown away as a language
        # you do not speak — the trigger works, the microphone works, and nothing ever appears.
        spoken = dictate.spoken_languages()
        if spoken and dictate.language_code(text) not in spoken:
            return (
                f"`languages` says you speak {', '.join(sorted(spoken))}, so every clip decoded as "
                f"{dictate.language_code(text)} would be thrown away. Add it there, or unset it."
            )
    if key == "polishCommand" and text:
        program = ""
        with contextlib.suppress(ValueError, IndexError):
            program = shlex.split(text)[0]
        if program and not dictate.resolve_bin(program):
            return (
                f"`{program}` is not on PATH, so polish would fail and degrade to the plain "
                "transcript on every sentence."
            )
    return ""


def _config(action: str = "", key: str = "", value: str = "") -> int:
    if action == "set":
        if not key:
            _out("usage: murmurflow config set <key> <value>")
            return 2
        parsed = config.coerce(value) if value else None
        if key == "trigger" and isinstance(parsed, str):
            # ONE spelling in the file. `ctrl_alt` and `control_option` are the same key and both
            # are accepted, but a config that stores whichever one was typed cannot be compared
            # against anything — least of all by the person reading it back.
            parsed = value = hotkey.canonical_trigger(parsed)
        if parsed is None and key not in config.KEYS:
            _out(f"[!] unknown setting `{key}` — there is nothing to unset.")
            return 2
        if parsed is not None:
            problem = _reject(key, parsed)
            if problem:
                _out(f"[!] {problem}")
                return 2
        # Stopped BEFORE the write, so that changing `port` still finds the server on the port it
        # is actually running on. Only when a listener is up to start it again — killing the warm
        # server with nothing left to restart it would trade a stale model for a slow one.
        bounce = key in _SERVER_SETTINGS and service.running() and dictate.stop_server()
        config.set_value(key, parsed)
        _out(f"{key} = {value or '(unset)'}")
        note = _warn(key, parsed) if parsed is not None else ""
        if note:
            _out(f"[!] {note}")
        if bounce:
            _out("[OK] stopped the warm whisper-server so it reloads with this setting")
        # The running listener read `trigger` and `doubleTap` when it bound and will not look
        # again, so without this the setting changed and the behaviour did not.
        if service.restart():
            _out("[OK] restarted the listener so it takes effect now")
        return 0
    current = config.load()
    _out(f"{config.config_path()}")
    _out("")
    width = max(len(k) for k in config.KEYS)
    for name, what in config.KEYS.items():
        set_to = current.get(name)
        mark = "*" if name in current else " "
        _out(f" {mark}{name.ljust(width)}  {set_to if name in current else ''}")
        _out(f"  {' ' * width}  {what}")
    return 0


# --- the daemon -------------------------------------------------------------------------------


def _listen(*, trigger: str = "") -> int:
    dictate.listen_loop(trigger=trigger, on_event=_out)
    return 0


def _toggle() -> int:
    """Press-once-to-start, press-again-to-stop — for binding to a macOS Shortcut or Automator.

    Exists because a key binding that only fires on key-DOWN structurally cannot do hold-to-talk.
    """
    result = dictate.toggle()
    if result is None:
        _out("recording... run this again to stop")
        return 0
    if result.problem:
        _out(f"[!] {result.problem}")
        return 1
    _out(result.text)
    return 0


def _transcribe(path: str) -> int:
    """Transcribe an existing audio file and print it. No microphone, no clipboard, no paste."""
    source = Path(path).expanduser()
    if not source.is_file():
        _out(f"no such file: {source}")
        return 2
    text = whisper.transcribe(source)
    if not text:
        _out("[!] nothing transcribed (run `murmurflow doctor`)")
        return 1
    _out(dictate.tidy(text))
    return 0


def main(argv: list[str] | None = None) -> int:
    # `murmurflow doctor | head` must not end in a BrokenPipeError traceback. Restoring the default
    # SIGPIPE makes this exit quietly when the reader walks away, the way every other Unix tool does.
    with contextlib.suppress(OSError, ValueError):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = argparse.ArgumentParser(
        prog="murmurflow",
        description="Hold a key, talk, let go. The text appears at your cursor. Nothing leaves your Mac.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("listen", help="run the press-to-talk daemon in this terminal (blocks)")
    sub.add_parser("install", help="install the login agent so dictation is always live")
    sub.add_parser("uninstall", help="stop dictation and remove it from login")
    sub.add_parser("doctor", help="what is missing, and the one command that fixes each thing")
    sub.add_parser("devices", help="list microphones")
    sub.add_parser("toggle", help="start/stop one recording (for a Shortcuts binding)")

    p_setup = sub.add_parser("setup", help="download a speech model")
    p_setup.add_argument("model", nargs="?", default="", help="e.g. large-v3-turbo, base")

    p_key = sub.add_parser("keytest", help="does this Mac see your trigger key?")
    p_key.add_argument("--trigger", default="", help="watch only this key")
    p_key.add_argument("--seconds", type=float, default=20.0)

    p_cues = sub.add_parser("cues", help="play the three tones, or switch preset")
    p_cues.add_argument("preset", nargs="?", default="")

    p_pause = sub.add_parser("pause", help="lend the trigger key to another program for a while")
    p_pause.add_argument("--seconds", type=float, default=dictate.DEFAULT_PAUSE_SECONDS)
    p_pause.add_argument("--who", default="", help="what is borrowing it, in words")
    sub.add_parser("resume", help="take the trigger key back")
    sub.add_parser("trigger", help="print the trigger key this install is on, and nothing else")

    p_cfg = sub.add_parser("config", help="show or change settings")
    p_cfg.add_argument("action", nargs="?", default="", choices=["", "set"])
    p_cfg.add_argument("key", nargs="?", default="")
    p_cfg.add_argument("value", nargs="?", default="")

    p_tr = sub.add_parser("transcribe", help="transcribe an audio file and print the text")
    p_tr.add_argument("path")

    p_listen = sub.add_parser("run", help=argparse.SUPPRESS)  # alias for `listen`
    p_listen.add_argument("--trigger", default="")

    # `murmurflow murmurflow config set cue glass` is what happens when somebody copies a line out
    # of the help, which prints every verb with the program name in front so it can be pasted
    # whole. Both readings are the same intent and argparse answered the honest one with a wall of
    # `invalid choice`. Only a LEADING repeat is dropped, so `transcribe murmurflow.wav` is safe.
    argv = list(sys.argv[1:] if argv is None else argv)
    while argv and argv[0] == "murmurflow":
        argv.pop(0)

    args = parser.parse_args(argv)
    command = args.command or "doctor"
    try:
        if command in {"listen", "run"}:
            return _listen(trigger=getattr(args, "trigger", ""))
        if command == "install":
            return _install()
        if command == "uninstall":
            return _uninstall()
        if command == "setup":
            return _setup(args.model)
        if command == "doctor":
            # The BARE `murmurflow` also lists what you can change. `murmurflow doctor`, typed on
            # purpose, does not — that one is a health check somebody is reading for one answer.
            return _doctor(verbs=not args.command)
        if command == "devices":
            return _devices()
        if command == "keytest":
            return _keytest(trigger=args.trigger, seconds=args.seconds)
        if command == "cues":
            return _cues(args.preset)
        if command == "pause":
            return _pause(args.seconds, args.who)
        if command == "resume":
            return _resume()
        if command == "trigger":
            # ONE canonical name on stdout and nothing else, because the reader is a program.
            # Everything else that says which key this is on says it in a sentence ("double-tap
            # left Control to start, tap once to stop"), which is right for a person and unparseable
            # for the tool deciding whether its own hotkey COLLIDES with this one. Borrowing the key
            # when it does not collide is not a small mistake: it stands dictation down across every
            # other app for as long as the borrower runs, silently.
            _out(dictate.trigger_key())
            return 0
        if command == "config":
            return _config(args.action, args.key, args.value)
        if command == "toggle":
            return _toggle()
        if command == "transcribe":
            return _transcribe(args.path)
    except KeyboardInterrupt:
        return 130
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
