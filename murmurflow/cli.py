"""``murmurflow`` — the command line. Ten verbs, and most people only ever type two.

``setup`` then ``install`` is the whole happy path. Everything else here exists because dictation
fails in exactly four ways — the key is not seen, the microphone is not heard, the model is not
found, the text is not typed — and each of those has its own verb that answers it in one run.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, dictate, hotkey, service, whisper


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
    return 0


def _install() -> int:
    """Warm the microphone, then install the launchd agent so dictation is live after every login."""
    ready, hint = dictate.available()
    if not ready:
        _out(hint)
        return 2
    if not service.is_macos():
        _out("murmurflow is macOS-only")
        return 2

    # The FIRST ever CoreAudio access on a Mac takes ~10 seconds. Paying it here, explicitly, means
    # the user's first real sentence is fast instead of looking broken.
    _out("warming the microphone (the first CoreAudio access takes ~10s, once)...")
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
    _out(f"[OK] installed {service.LABEL}" if ok else f"[!] launchctl: {detail}")
    _out("")
    _out(f"{dictate.trigger_hint()} anywhere and talk. Release: the text types itself.")
    _out("")
    _out("macOS will ask for two permissions the first time. Neither can be granted from a script,")
    _out("and both are required:")
    _out("  1. Microphone      — to hear you")
    _out("  2. Accessibility   — to type into the app you are using")
    _out("")
    _out("The apps you dictate INTO need nothing. Every permission goes to this one process,")
    _out("which System Settings lists under the name of its interpreter, NOT 'murmurflow':")
    _out(f"  {_tcc_entry()}")
    # Opening the exact pane, rather than naming a path through System Settings, is the difference
    # between "one switch is highlighted for you" and a person who has never opened Privacy &
    # Security hunting for it. Best-effort: a Mac that refuses the URL still has the sentence above.
    _out("")
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
    _out("[OK] dictation stopped and removed from login." if ok else f"[!] {detail}")
    if freed:
        _out("[OK] stopped the warm whisper-server")
    return 0 if ok else 1


# --- diagnosis --------------------------------------------------------------------------------


def _tcc_entry() -> str:
    """What System Settings CALLS this program in its permission lists, and where that file is.

    macOS names a permission row after the EXECUTABLE that asked, and for a Python tool that is the
    interpreter rather than the console script — so the row reads ``python3.13`` and somebody
    scrolling for ``murmurflow`` never finds it, or worse, switches on a neighbouring row belonging
    to some other tool sharing an interpreter. Naming it exactly is the difference between a switch
    found and a switch hunted for. A signed .app bundle is what would make the row say MurmurFlow,
    and that is a bigger change than this one.
    """
    real = Path(sys.executable).resolve()
    return f"{real.name}  ({real})"


def _doctor() -> int:
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
    rows.append(
        (
            hotkey.available(),
            "key polling: " + ("available" if hotkey.available() else "BLOCKED"),
            "grant Input Monitoring, then run: murmurflow keytest",
        )
    )
    rows.append(
        (
            hotkey.accessibility_trusted(),
            "accessibility: "
            + (
                "granted"
                if hotkey.accessibility_trusted()
                else "NOT granted — nothing can be typed"
            ),
            f"switch on '{_tcc_entry()}' in System Settings > Privacy & Security > Accessibility",
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
    rows.append(
        (
            not dictate.secure_input_active(),
            "secure input: "
            + ("off" if not dictate.secure_input_active() else "ON — pasting is blocked"),
            "quit whatever has a password field focused",
        )
    )
    installed = service.plist_path().is_file()
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
    _out(f"  trigger: {dictate.trigger_hint()}")
    _out(f"  cue: {dictate.cue_preset_name()}")
    polish = str(config.load().get("polishCommand", "") or "")
    _out(f"  polish: {polish or 'off (deterministic cleanup only)'}")
    return 0 if all(ok for ok, _, _ in rows) else 1


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
    watching = [trigger] if trigger else [*hotkey.FLAG_TRIGGERS, *hotkey.TRIGGERS]
    gesture = "double-tap" if dictate.double_tap_mode() else "hold"
    _out(f"press any of these now — watching {len(watching)} key(s) for {seconds:.0f}s.")
    _out(f"your trigger is {dictate.trigger_key()}, your gesture is {gesture}. ctrl-c to stop.")
    _out("  (a 'command'/'option' hit means EITHER side; 'left_*'/'right_*' means that side)")
    lib = hotkey._load()
    held: dict[str, float] = {}
    presses: dict[str, list[float]] = {name: [] for name in watching}
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            for name in watching:
                down = hotkey.is_trigger_down(name, lib)
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


def _config(action: str = "", key: str = "", value: str = "") -> int:
    if action == "set":
        if not key:
            _out("usage: murmurflow config set <key> <value>")
            return 2
        config.set_value(key, config.coerce(value) if value else None)
        _out(f"{key} = {value or '(unset)'}")
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

    p_cfg = sub.add_parser("config", help="show or change settings")
    p_cfg.add_argument("action", nargs="?", default="", choices=["", "set"])
    p_cfg.add_argument("key", nargs="?", default="")
    p_cfg.add_argument("value", nargs="?", default="")

    p_tr = sub.add_parser("transcribe", help="transcribe an audio file and print the text")
    p_tr.add_argument("path")

    p_listen = sub.add_parser("run", help=argparse.SUPPRESS)  # alias for `listen`
    p_listen.add_argument("--trigger", default="")

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
            return _doctor()
        if command == "devices":
            return _devices()
        if command == "keytest":
            return _keytest(trigger=args.trigger, seconds=args.seconds)
        if command == "cues":
            return _cues(args.preset)
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
