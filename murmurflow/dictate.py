"""The engine: capture, transcribe warm, clean up, inject at the cursor.

Hold a key, speak, release: the text lands wherever your cursor is — Slack, a terminal, a browser,
your notes app. Nothing you say leaves the machine.

**Latency is the whole product**, so it is worth being precise about where it goes. Measured on an
M4 Pro, macOS 26, large-v3-turbo, an 11s clip, ``-t 8``:

============================================================  =======
step                                                          seconds
============================================================  =======
first transcription after boot (1.6 GB model off disk)          13.3
cold ``whisper-cli``, model file in the page cache                2.2
warm ``whisper-server``, model resident                           2.3
ffmpeg avfoundation capture start, first ever                     9.9
ffmpeg avfoundation capture start, thereafter                     0.3
============================================================  =======

Read that table honestly: **once the model file is in the OS page cache, warm and cold are the same
within noise.** The warm ``whisper-server`` earns its place on the FIRST transcription — 13.3s
against 2.3s — which is precisely the one a person forms their opinion on, and precisely the one a
freshly-booted laptop serves. :func:`listen_loop` therefore starts the server when the daemon starts
and pays that cost while nobody is waiting, and the cold CLI
(:func:`murmurflow.whisper.transcribe`) remains the fallback when no server could be started.

These numbers move a lot with the machine, the model and what else is resident: two whisper-servers
on one Mac roughly doubled them here. Measure your own before believing any of them, including ours.

**What this deliberately does NOT do.** No compiled helper — hold-to-talk is detected by polling
``CGEventSourceFlagsState`` through stdlib :mod:`ctypes` (see :mod:`murmurflow.hotkey`), which needs
no Xcode, no code signing and no notarization. No streaming or partial decode: it would cost a
permanently resident microphone to save a fraction of a batch pass. No LLM rewrite on the hot path
unless you ask for one — :func:`polish` is opt-in precisely because spawning a model costs seconds
against a transcription measured in low single digits, which would multiply the felt latency.

**Privacy.** Audio is captured to ``~/.murmurflow/audio/`` and deleted the instant it has been
transcribed — before the optional polish call and before the paste, so a crash downstream cannot
leave your voice on disk. The recorder also bounds its own life (:data:`MAX_CLIP_SECONDS`), so a
daemon killed mid-clip cannot leave the microphone hot. The transcript itself is never inspected,
filtered or sent anywhere: it goes to your clipboard and then to your cursor. The one exception is
``polishCommand``, which is a command *you* configure and which therefore sends the text exactly
where you told it to — see :func:`polish`.
"""

from __future__ import annotations

import array
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import config, platforms, whisper

# Homebrew's bin dirs. launchd hands an agent a minimal PATH that excludes them, so a bare
# shutil.which() finds nothing when the listener runs from a plist while working fine in a shell
# (the TUNNEL-PATH-1 lesson, generalized here rather than re-learned).
_FALLBACK_BIN_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")

# The mic to actually speak into. A default CoreAudio device is often an aggregate
# ("Push3 + Z1 + Volt2" — a music interface), so ":default" would record the wrong input entirely.
# Resolved by NAME at runtime because avfoundation INDICES shift when a USB device is plugged in.
DEFAULT_INPUT_NAME = ""

# whisper-server's loopback port. High and odd, to miss the usual 8080/5000 collisions — and NOT
# 8477, which is whisper-server's own commonly-copied example port. `start_server` treats any server
# already answering here as ours and reuses it, so landing on a port someone else's whisper-server
# occupies means silently transcribing against THEIR model and ignoring `model` entirely.
# Found exactly that way: a second tool on 8477 answered every request and the configured model
# never loaded. Change it with `murmurflow config set port <n>` if something else wants this one.
DEFAULT_PORT = 8479

_STATE_NAME = "dictate.json"

# The ffmpeg recorder THIS process started, kept so :func:`stop` can reap it instead of polling a
# zombie. ``None`` whenever the recorder belongs to another process (the `toggle` shape, where one
# CLI invocation starts the capture and a second one stops it).
_PROC: subprocess.Popen[bytes] | None = None


def resolve_bin(name: str) -> str:
    """Absolute path to ``name``, searching PATH then Homebrew's dirs; ``''`` if absent.

    Never raises. The fallback dirs matter only under launchd (see :data:`_FALLBACK_BIN_DIRS`).
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in _FALLBACK_BIN_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def available() -> tuple[bool, str]:
    """``(ready, hint)`` — whether dictation can run, and the ONE command that fixes it if not.

    A missing binary is reported in plain English naming the exact command that installs it, never
    as an exception. Three things are genuinely required and there is no fourth: something to record
    with, something to transcribe with, and a model to transcribe against.
    """
    ok, why = platforms.capture_available()
    if not ok:
        return False, f"Dictation needs a recorder. {why}"
    if not (resolve_bin("whisper-server") or whisper.available()):
        # The one-command fix is not the same command on both platforms, and a hint naming the
        # wrong package manager is worse than no hint — it sends somebody off to install Homebrew
        # on a machine that has never had it.
        fix = (
            "`brew install whisper-cpp`"
            if sys.platform == "darwin"
            else "re-run the installer — it fetches whisper.cpp's own Windows build"
        )
        return False, f"Dictation needs a local transcriber. Install it: {fix}"
    if not whisper.model():
        return False, ("No whisper model found. Download one: `murmurflow setup`")
    return True, ""


# --- config -----------------------------------------------------------------------------------


def _cfg() -> dict[str, object]:
    try:
        return config.load()
    except Exception:  # noqa: BLE001 — a broken config must never wedge the hotkey daemon
        return {}


def port() -> int:
    """The loopback port for the warm whisper-server (``port``)."""
    try:
        return int(str(_cfg().get("port", "") or DEFAULT_PORT))
    except (TypeError, ValueError):
        return DEFAULT_PORT


def input_name() -> str:
    """The avfoundation input device NAME to record from (``inputName``); ``""`` = system default."""
    return str(_cfg().get("inputName", "") or DEFAULT_INPUT_NAME).strip()


def state_path() -> Path:
    """``~/.murmurflow/dictate.json`` — the in-flight recording marker (pid + wav), if any."""
    return config.home_root() / _STATE_NAME


def _scratch_dir() -> Path:
    """``~/.murmurflow/audio/`` — where in-flight dictation audio is staged, created on demand.

    Each clip is deleted the instant it has been transcribed, so this directory is empty between
    sentences. It is the only place your recorded voice ever exists.
    """
    root = config.home_root() / "audio"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --- microphone -------------------------------------------------------------------------------


def list_inputs() -> list[tuple[str, str]]:
    """Every audio input as ``(id, name)``, from the platform. ``[]`` if none can be enumerated.

    The id is opaque above this line: an avfoundation index on macOS, a dshow device name on
    Windows. Only :func:`murmurflow.platforms.capture_args` ever reads it, which is what lets the
    recorder below be written once.
    """
    return platforms.list_inputs()


# The resolved mic, cached because enumerating avfoundation devices costs a whole ffmpeg subprocess
# (~160ms measured) and `start()` did it on EVERY key press — a third of the dead time between the
# key going down and the first audio sample, spent re-learning something that only changes when
# hardware is plugged in.
_INPUT_CACHE: tuple[str, str] | None = None
_INPUT_CACHED_AT = 0.0

# How long that answer is trusted before the devices are listed again.
#
# CACHING IT FOR THE DAEMON'S LIFETIME WAS THE BUG, and the lifetime is the reason: the agent is
# installed with `KeepAlive`, so "for this run" means until the next logout, which is weeks. A
# Bluetooth headset that is still connecting when the agent starts at login is simply absent from
# the first enumeration, the fallback picks the built-in microphone, and every sentence for the
# next fortnight is recorded on the wrong device — while `murmurflow doctor`, a fresh process with
# an empty cache of its own, cheerfully reports the right one. Nothing on screen can be told from a
# working setup.
#
# Ten minutes bounds that to one coffee break, and the cost is the ~160ms enumeration paid at most
# once in every ten, which is far below the ~300ms the microphone itself takes to open. Monotonic
# and not wall clock: a laptop that sleeps, wakes and syncs its clock backwards would otherwise
# hold a stale device for however far back the correction went.
INPUT_CACHE_SECONDS = 600.0


def resolve_input() -> tuple[str, str]:
    """``(device_id, resolved_name)`` for the configured mic.

    Matched by NAME (case-insensitive substring) against the live device list, because ids shift
    the moment a USB interface is plugged in — pinning one would silently start recording the wrong
    device. Falls back to whatever the platform calls "the system default" when the named device is
    absent (unplugged headset, a client machine with different hardware), so dictation degrades to
    "wrong-ish mic" rather than "broken". On Windows there is no default TOKEN — dshow has no
    spelling for it — so the fallback there is the first audio input there is.
    """
    global _INPUT_CACHE, _INPUT_CACHED_AT
    if _INPUT_CACHE is not None and time.monotonic() - _INPUT_CACHED_AT < INPUT_CACHE_SECONDS:
        return _INPUT_CACHE
    want = input_name().lower()
    devices = list_inputs()
    resolved = (platforms.default_input(), "system default")
    for index, name in devices:
        if want and want in name.lower():
            resolved = (str(index), name)
            break
    else:
        for index, name in devices:  # name miss: prefer a real mic over an aggregate interface
            if "microphone" in name.lower() or "mikrofon" in name.lower():
                resolved = (str(index), name)
                break
    if devices:  # never cache a failed enumeration — ffmpeg may simply not have been ready
        _INPUT_CACHE, _INPUT_CACHED_AT = resolved, time.monotonic()
    return resolved


# --- capture ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """An in-flight capture: the ffmpeg pid and the wav it is writing."""

    pid: int
    wav: Path
    started_at: float

    @property
    def seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)


def current() -> Recording | None:
    """The in-flight recording, or ``None``. Stale markers (dead pid) are cleaned up and ignored."""
    path = state_path()
    try:
        raw = json.loads(path.read_text("utf-8"))
        rec = Recording(int(raw["pid"]), Path(raw["wav"]), float(raw["started_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        os.kill(rec.pid, 0)  # signal 0 = liveness probe, kills nothing
    except (OSError, ProcessLookupError):
        path.unlink(missing_ok=True)
        return None
    return rec


def start() -> Recording | None:
    """Begin capturing the mic to a fresh 16kHz mono wav; ``None`` if already recording or unable.

    Returns as soon as ffmpeg is spawned — CoreAudio needs ~300ms more before the first sample
    actually lands (measured; it is a device-start floor, not an ffmpeg tax, so a compiled helper
    would not beat it). Callers that cue the user should cue on :func:`ready`, not on this
    return, or the first word is clipped.
    """
    if current() is not None:
        return None
    ffmpeg = resolve_bin("ffmpeg")
    if not ffmpeg:
        return None
    index, _ = resolve_input()
    wav = _scratch_dir() / f"dictate-{int(time.time())}-{uuid.uuid4().hex[:8]}.wav"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        # A HARD CEILING on one clip, and it is a privacy control, not a convenience. ffmpeg is
        # spawned with `start_new_session=True` so it survives its parent: kill the daemon (launchd
        # restart, a crash, `kickstart -k`) while a clip is running and nothing ever stops it. Found
        # live in development — THREE orphaned recorders, 4.5 hours each, 1.4 GB of recorded
        # voice on disk and the microphone hot the whole time, which is the exact incident this
        # product exists not to cause. `-t` makes the recorder bound its own life, with no
        # supervisor needed and nothing to remember.
        "-t",
        str(MAX_CLIP_SECONDS),
        *platforms.capture_args(index),
        # ffmpeg's avfoundation input keeps exactly ONE pending audio buffer and releases the
        # previous one whenever a new buffer arrives before its reader has taken it, so a little
        # scheduling jitter silently costs samples. Measured here: ~11% of every capture, on the
        # built-in mic, on an aggregate interface AND on a pure-software loopback with no hardware
        # clock at all — so it is the input device implementation, not the microphone. It is also
        # unreachable from the CLI: identical loss whether ffmpeg resamples and converts or copies
        # raw bytes, and `-thread_queue_size` changes nothing.
        #
        # The damage is not the missing samples themselves but WHERE the hole goes. ffmpeg takes
        # the timestamps from the buffers it did get, so the gap is spliced out and the whole
        # sentence is handed to whisper ~11% too fast. `async=1` fills the gaps instead of closing
        # them, which keeps the clip on real time (measured 87% -> 98% of the hold) and stops
        # speech being sped up. It cannot bring the lost samples back; recovering those means
        # leaving avfoundation for a ctypes CoreAudio recorder, which is not worth it while
        # transcripts are this good.
        "-af",
        "aresample=async=1",
        "-ar",
        "16000",
        "-ac",
        "1",
        # Write every packet straight through instead of buffering. Without this ffmpeg holds ~2s
        # of audio in memory before the file grows, so `ready()` cannot tell that the mic went live
        # until long after it did — so the "start talking" cue arrives two seconds late, by which
        # point half the sentence has already been said into a microphone that was not listening.
        "-flush_packets",
        "1",
        "-y",
        str(wav),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # survive the caller; we stop it explicitly by pid
        )
    except (OSError, subprocess.SubprocessError):
        return None
    global _PROC
    _PROC = proc  # so stop() can REAP it — see the zombie note there
    rec = Recording(proc.pid, wav, time.time())
    with contextlib.suppress(OSError):
        state_path().write_text(
            json.dumps({"pid": rec.pid, "wav": str(rec.wav), "started_at": rec.started_at}),
            "utf-8",
        )
    return rec


def ready(rec: Recording, *, timeout: float = 2.0) -> bool:
    """Block until the wav has actual audio frames in it (or ``timeout``). ``True`` if it does.

    The ~300ms CoreAudio start-up is invisible only if the cue fires when the mic is
    genuinely live. A 44-byte wav is a header with no samples yet.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if rec.wav.stat().st_size > 1024:
                return True
        except OSError:
            pass
        time.sleep(0.02)
    return False


def _exited(pid: int) -> bool:
    """True once the recorder with ``pid`` is really gone — zombies included.

    ``os.kill(pid, 0)`` is NOT enough: a child that has exited but not been reaped is a zombie, and
    signalling a zombie SUCCEEDS. Polling it therefore never observes the exit, which cost a flat
    two seconds on every single dictation (measured: ffmpeg itself is gone in ~34ms) before the
    loop gave up and SIGKILLed a process that had been dead the whole time. When the recorder is
    our own child we reap it with ``waitpid``; when it is not (``toggle`` starts it in one CLI
    invocation and stops it in another) no zombie can exist for us, so signal-0 is accurate.
    """
    proc = _PROC
    if proc is not None and proc.pid == pid:
        return proc.poll() is not None
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child: signal-0 below is authoritative
    except OSError:
        return True
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return True
    return False


def repair_wav(wav: Path) -> bool:
    """Patch a RIFF header whose lengths were never written. ``True`` if it had to. Never raises.

    **ffmpeg writes the real lengths when it EXITS, and it does not always get to.** SIGKILL on the
    fallback path in :func:`stop` is one way; Windows is the other, and there it is not a fallback
    but the only path — ``os.kill`` there cannot deliver anything gentler than ``TerminateProcess``
    for a signal that is not CTRL_C/CTRL_BREAK, and a daemon started by ``pythonw`` has no console
    to send those through.

    Measured rather than assumed, because the guess was wrong and the wrong guess would have been
    a scary comment about a bug that does not exist. A killed ffmpeg leaves ``0xFFFFFFFF`` in both
    size fields, not zero, so the audio still decodes — ``-flush_packets 1`` already wrote every
    sample (see :func:`start`) and the readers stop at the end of the file. What breaks is
    :func:`audio_seconds`, which believes the header and reports **37 hours** of captured audio:
    that number is the entire capture-fault diagnostic (see the daemon loop, where a clip shorter
    than the hold is how a throttled agent recording a quarter of every sentence was found), and it
    goes silently blind on exactly the clips something already went wrong with.

    Cheaper than what it protects: a stat and twelve bytes on the ordinary path, where the header
    is already right and this returns immediately.
    """
    try:
        size = wav.stat().st_size
        if size <= 44:  # a header and nothing else — there is nothing to rescue
            return False
        with wav.open("r+b") as handle:
            if handle.read(4) != b"RIFF":
                return False
            handle.seek(8)
            if handle.read(4) != b"WAVE":
                return False
            offset = 12
            while offset + 8 <= size:
                handle.seek(offset)
                name = handle.read(4)
                declared = int.from_bytes(handle.read(4), "little")
                if name == b"data":
                    actual = size - (offset + 8)
                    if declared and declared <= actual:
                        return False  # the trailer was written; leave it exactly as it is
                    handle.seek(offset + 4)
                    handle.write(actual.to_bytes(4, "little"))
                    handle.seek(4)
                    handle.write((size - 8).to_bytes(4, "little"))
                    return True
                if declared <= 0:
                    return False  # a chunk with no length: the walk cannot go on honestly
                offset += 8 + declared + (declared % 2)  # chunks are padded to an even length
    except (OSError, ValueError):
        return False
    return False


def stop(rec: Recording | None = None) -> Path | None:
    """Stop the in-flight capture and return the finished wav (``None`` if nothing was recording).

    SIGINT (not SIGKILL) so ffmpeg writes the RIFF trailer — a killed ffmpeg leaves a wav whose
    header claims zero length and whisper decodes it as silence. This sits directly on the felt
    latency (it runs the instant the key is released), so exit is detected by REAPING
    the child rather than polling it — see :func:`_exited`.
    """
    global _PROC
    rec = rec or current()
    if rec is None:
        return None
    with contextlib.suppress(OSError, ProcessLookupError):
        os.kill(rec.pid, signal.SIGINT)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if _exited(rec.pid):
            break
        time.sleep(0.01)
    else:  # never exited — force it, the trailer is lost but a truncated wav still decodes
        with contextlib.suppress(OSError, ProcessLookupError):
            os.kill(rec.pid, signal.SIGKILL)
    if _PROC is not None and _PROC.pid == rec.pid:
        _PROC = None
    _clear_state_for(rec.pid)
    if not rec.wav.is_file():
        return None
    repair_wav(rec.wav)  # a recorder that had to be killed still leaves a decodable clip
    return rec.wav


def reap_orphans() -> int:
    """Stop every recorder left behind by a previous run, and delete its audio. Returns how many.

    A recorder is spawned with ``start_new_session=True`` so it outlives the call that started it —
    which also means it outlives the DAEMON. A launchd restart, a crash or a ``kickstart -k`` in the
    middle of a clip leaves ffmpeg running forever with the microphone open: found live on the
    machine as three recorders, 4.5 hours each, 1.4 GB of recorded voice on disk. `-t`
    (:data:`MAX_CLIP_SECONDS`) bounds the damage; this ends it, because a daemon that is only now
    starting cannot own a clip from before it existed.

    Matched on the exact scratch-path pattern this module writes, so no other ffmpeg on the machine
    is ever a candidate. Best-effort and silent on any failure — a reaper must never keep the daemon
    from starting.
    """
    scratch = _scratch_dir()
    reaped = 0
    try:
        pattern = f"-y {scratch}/dictate-"
        found = subprocess.run(
            # `--` IS LOad-BEARING, and its absence is why this reaper never reaped anything. The
            # pattern begins with `-y`, so without the guard pgrep parses it as its own option and
            # exits with "illegal option -- y" before matching a single process. It failed silently
            # in exactly the shape this function is written to tolerate — empty stdout, no raise —
            # so every daemon start reported nothing to reap while an orphan held the microphone
            # open. Found live: one recorder open since 4:03pm, writing
            # to a wav that had already been deleted, with the mic indicator lit the whole time.
            ["pgrep", "-f", "--", pattern],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pids = [int(line) for line in found.stdout.split() if line.strip().isdigit()]
        for pid in pids:
            if pid == os.getpid():
                continue
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, signal.SIGINT)  # SIGINT, so the wav still gets its RIFF trailer
                reaped += 1
    except (OSError, subprocess.SubprocessError, ValueError):
        return reaped
    # The audio goes too: a clip nobody is waiting for has no transcription to outlive, and the
    # contract is that your voice does not sit on disk (`keepAudio` keeps ONE file,
    # deliberately, and it is not named like these).
    with contextlib.suppress(OSError):
        for wav in scratch.glob("dictate-*.wav"):
            wav.unlink(missing_ok=True)
    state_path().unlink(missing_ok=True)
    return reaped


def _clear_state_for(pid: int) -> None:
    """Drop the in-flight marker, but ONLY if it still describes ``pid``.

    A second surface may answer on a worker thread while a new clip starts, which means the NEXT
    recording can already be running by the time the previous one is stopped. Unlinking
    unconditionally orphaned it — ffmpeg still capturing, ``current()`` reporting nothing — so the
    clip already in flight could never be finished. An unreadable marker is cleared,
    since a marker nobody can parse is worse than none.
    """
    try:
        raw = json.loads(state_path().read_text("utf-8"))
        if int(raw["pid"]) != pid:
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    state_path().unlink(missing_ok=True)


# --- warm transcription -----------------------------------------------------------------------


def server_url() -> str:
    return f"http://127.0.0.1:{port()}"


def server_up() -> bool:
    """True if a warm whisper-server answers on the loopback port."""
    try:
        with urllib.request.urlopen(f"{server_url()}/", timeout=0.5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def server_answers() -> tuple[bool, str]:
    """Does the warm server actually TRANSCRIBE — ``(ok, what went wrong)``. Never raises.

    :func:`server_up` opens a socket to ``/``, and that is the check the health report trusted while
    every single clip took the cold path for a day. A wedged server accepts the connection happily
    and answers ``/inference`` with a 500; from the outside the two are identical, and the only
    place the difference showed was a ``· cold ·`` in the daemon log that nobody reads until
    something is already wrong.

    So this asks the question that matters, with the smallest possible clip: a fifth of a second of
    silence, through the same multipart path a real dictation uses. Not free — a wedged server
    answers in milliseconds but a healthy one decodes, so budget a moment — which is why it lives in
    ``doctor`` and never on the hot path.
    """
    if not server_up():
        return False, f"nothing on :{port()}"
    try:
        with tempfile.TemporaryDirectory(prefix="murmurflow-probe-") as tmp:
            probe = Path(tmp) / "probe.wav"
            with wave.open(str(probe), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00" * 6400)  # 0.2s of digital silence
            body, content_type = _multipart(probe, {"response_format": "json", "temperature": "0"})
            request = urllib.request.Request(
                f"{server_url()}/inference", data=body, headers={"Content-Type": content_type}
            )
            with urllib.request.urlopen(request, timeout=30):
                return True, ""
    except urllib.error.HTTPError as error:
        # The 500 body names the cause in one sentence ("FFmpeg conversion failed."), and that
        # sentence is the entire difference between a fixable setup and a mystery.
        detail = ""
        with contextlib.suppress(Exception):
            detail = error.read().decode("utf-8", errors="ignore").strip()[:120]
        return False, detail or f"HTTP {error.code}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, str(error)[:120]


def serve_command() -> list[str] | None:
    """The argv that starts the warm whisper-server, or ``None`` if it cannot be built."""
    binary = resolve_bin("whisper-server")
    model = whisper.model()
    if not binary or not model:
        return None
    return [
        binary,
        "-m",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port()),
        "-t",
        whisper.threads(),
        "--convert",  # let the server transcode anything ffmpeg reads, not just wav
        # Stop whisper emitting "*sad*"/"[MUSIC]"-style sound annotations at the SOURCE rather than
        # filtering them afterwards. `is_hallucination` stays as the backstop: -sns reduces these
        # but does not eliminate them.
        "-sns",
    ]


def start_server(*, wait: float = 60.0) -> bool:
    """Spawn the warm whisper-server if it is not already up; block until it answers.

    Loading large-v3-turbo takes a few seconds, which is exactly the cost we are paying ONCE here
    so that every subsequent dictation does not. Returns True if a server is answering.

    **The ``cwd`` is the whole warm path**, and leaving it out is how MurmurFlow quietly lost it.
    ``--convert`` (see :func:`serve_command`) makes whisper-server shell out to ffmpeg, and ffmpeg
    writes its converted copy into the server's WORKING DIRECTORY. Started from a shell that
    directory is the repo and everything works, which is why this survived every manual test. Under
    the installed agent the daemon inherits ``/`` — not writable — so the conversion fails, the
    server answers **every single request** with ``500 {"error":"FFmpeg conversion failed."}``, and
    every clip silently falls through to the cold CLI. Measured live 2026-08-18 with one server on
    ``/`` and one on a writable dir, same binary, same flags, same model, same request: 500 in 0.03s
    against a clean transcript in 2.17s.

    Nothing on screen says so. The transcripts still arrive, just slower and — because the cold path
    carries no server-side prompt and reports no confidence — measurably worse, with the two silence
    gates weakened to boot. `watch_warm` in :func:`listen_loop` faithfully bounced the server after
    every second cold clip, all day, into the same broken working directory.

    So the server is started in :func:`_scratch_dir`, which is ours, writable, and already the one
    place recorded voice lives — whisper-server deletes its converted copy when it is done, so the
    "empty between sentences" promise there still holds.
    """
    if server_up():
        return True
    cmd = serve_command()
    if cmd is None:
        return False
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(_scratch_dir()),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    deadline = time.time() + wait
    while time.time() < deadline:
        if server_up():
            return True
        time.sleep(0.1)
    return False


def stop_server() -> int:
    """Stop the warm whisper-server this install started. Returns how many were stopped.

    ``start_server`` detaches it with ``start_new_session=True`` so it outlives the listener, which
    is the whole point while dictation is installed — and a leak the moment it is not: 1.8 GB
    resident with nothing left to ask it anything, until the next reboot. Matched on OUR port,
    because a whisper-server on any other port belongs to somebody else.
    """
    try:
        found = subprocess.run(
            ["pgrep", "-f", f"whisper-server.*--port {port()}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    stopped = 0
    for token in found.stdout.split():
        with contextlib.suppress(ValueError, ProcessLookupError, PermissionError):
            os.kill(int(token), signal.SIGTERM)
            stopped += 1
    return stopped


def _multipart(wav: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    """Build a multipart/form-data body for whisper-server's ``/inference`` (stdlib only)."""
    boundary = f"----murmurflow{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{wav.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    parts.append(wav.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# An older whisper-server answers with the language NAME and no `language_probabilities` to read a
# code off. Only the names anybody actually lists in `languages` need to be here; anything else
# falls through as itself, and an unrecognised language is compared as whisper spelled it.
_LANGUAGE_CODES: dict[str, str] = {
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "dutch": "nl",
    "portuguese": "pt",
    "polish": "pl",
    "russian": "ru",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
}


def language_code(name: str) -> str:
    """``"german"``, ``"German"`` and ``"de"`` are one answer; anything else passes through as itself.

    One seam, so the gate that reads what whisper detected and the gate that reads what the user
    said they speak can never disagree about what a language is called.
    """
    key = str(name).strip().lower()
    return _LANGUAGE_CODES.get(key, key)


class Heard(NamedTuple):
    """A transcript plus how sure whisper was that it was listening to speech at all.

    ``confidence`` is whisper's own ``detected_language_probability``: how strongly the audio looks
    like ANY one human language. Speech scores 0.97-0.999 (measured: a 1.2s English slice 0.969, a
    single "Okay." 0.981, a German sentence 0.999); room tone, pink noise and digital silence score
    0.32-0.45, because there is no language in them to be sure about. It is ``1.0`` — deliberately
    "certain" — whenever the signal is unavailable (cold path, forced language, older server), so a
    missing score can never silently swallow something that was said. See :data:`SPEECH_CONFIDENCE`.
    """

    text: str
    confidence: float = 1.0
    language: str = ""
    #: Did the WARM server answer this one? A wedged server is invisible otherwise — it answers
    #: every request with an error, every clip quietly takes the cold path, and the two gates that
    #: read a confidence and a language are weaker there. That degradation has to be legible in the
    #: log, or the next person to hit it is also debugging blind. ``None`` = nothing was decoded.
    warm: bool | None = None


def _confidence(payload: str) -> tuple[str, float, str]:
    """Pull ``(text, detected_language_probability, language)`` out of verbose_json. Never raises.

    ``strict=False`` is load-bearing, not defensive dressing: whisper-server puts the transcript's
    trailing newline into the JSON string RAW, which is invalid JSON that ``json.loads`` rejects
    outright. A strict parse fails on exactly the short utterances this gate exists to judge.
    """
    try:
        data = json.loads(payload, strict=False)
    except (ValueError, TypeError):
        # Not JSON at all — an older server answering a verbose_json request with plain text. Take
        # it as the transcript and claim no opinion rather than discarding a real sentence.
        return payload.strip(), 1.0, ""
    if not isinstance(data, dict):
        return payload.strip(), 1.0, ""
    text = str(data.get("text", "")).strip()
    raw = data.get("detected_language_probability")
    # A CODE, NEVER THE NAME. whisper-server reports `"language": "english"` while a person writes
    # `["de", "en"]` in their config, so the gate below compared "english" against {"de","en"} and
    # would have rejected every sentence he ever spoke the moment the warm server came back up —
    # a gate that is inert today and catastrophic tomorrow. `language_probabilities` is keyed by
    # the codes themselves, so the top key IS the answer with no table to keep in sync.
    probabilities = data.get("language_probabilities")
    spoken = ""
    if isinstance(probabilities, dict) and probabilities:
        numeric = {k: v for k, v in probabilities.items() if isinstance(v, (int, float))}
        if numeric:
            spoken = str(max(numeric, key=lambda k: numeric[k])).strip().lower()
    if not spoken:
        spoken = language_code(str(data.get("language", "") or ""))
    return text, float(raw) if isinstance(raw, (int, float)) else 1.0, spoken


def transcribe_warm(wav: Path, *, timeout: float = 60.0) -> Heard:
    """Transcribe via the warm server; empty text if it is not up or errors. Never raises.

    Reuses :mod:`whisper`'s language and vocabulary decisions, so every surface that transcribes
    hears your own proper nouns the same way.

    Asks for ``verbose_json`` rather than ``text`` purely to get the language score back; the
    transcript is identical either way.
    """
    if not wav.is_file():
        return Heard("")
    fields = {
        "response_format": "verbose_json",
        "language": whisper.language(),
        "prompt": whisper.vocabulary(),
        "temperature": "0",
    }
    try:
        body, content_type = _multipart(wav, fields)
    except OSError:
        return Heard("")
    request = urllib.request.Request(
        f"{server_url()}/inference", data=body, headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError, ValueError):
        return Heard("")
    return Heard(*_confidence(payload), warm=True)


def transcribe(wav: Path, *, timeout: float = 60.0) -> Heard:
    """Warm server first, cold ``whisper-cli`` second. Empty text when both fail. Never raises.

    The cold path reports no confidence (its CLI prints only text), so it returns the default 1.0
    and the silence gate simply does not apply there — fail-open, per :class:`Heard`.
    """
    warm = transcribe_warm(wav, timeout=timeout)
    if warm.text:
        return warm
    try:
        # THE COLD PATH REPORTS ITS LANGUAGE TOO. It used to report only text, which made the
        # "is that one of yours" gate inert on the exact path a wedged warm server drops you onto
        # — and that is how a desk bump came back as Japanese and got typed. Confidence stays the
        # fail-open 1.0: whisper.cpp's CLI has no probability to give and a missing number must
        # never swallow a real sentence.
        text, spoken = whisper.transcribe_heard(wav, timeout=timeout)
        return Heard(text, 1.0, spoken, warm=False)
    except Exception:  # noqa: BLE001 — any failure at all: dictation degrades, it never crashes
        return Heard("")


# --- cleanup ----------------------------------------------------------------------------------

# Leading throat-clearing ("so, um, okay so ...") and mid-sentence fillers. Whisper transcribes these
# faithfully, because they were genuinely said; nobody wants them typed. Deliberately a word list
# and not a rewrite: the ceiling of a deterministic strip is exactly this, and reaching past it means
# reaching for a language model (`polishCommand`), which is a choice the user makes, not a default.
_FILLER_LEAD = re.compile(
    r"^(?:(?:yo|hey|okay so|ok so|so|well|alright|right|um+|uh+)[,.]?\s+)+", re.IGNORECASE
)
_FILLER_WORD = re.compile(
    r"\b(?:um+|uh+|erm+|hmm+|basically|you know|i mean)\b[,.]?\s*", re.IGNORECASE
)


def strip_fillers(transcript: str) -> str:
    """Remove spoken filler from a transcript. Removes only — it never rewrites or re-cases.

    Falls back to the original when the strip would eat the whole line: "so, um, yeah" must not
    become "". A filler-free transcript passes through byte-identical.
    """
    original = transcript.strip()
    text = _FILLER_LEAD.sub("", original)
    text = _FILLER_WORD.sub("", text)
    text = " ".join(text.split())
    return text or original


# A clip this short cannot contain a word — it is a brushed key or an aborted chord. Transcribing
# it wastes a second and, worse, invites the hallucination below.
MIN_CLIP_SECONDS = 0.4

# The one problem that is NOT worth a sound: see the daemon's release handler.
TOO_SHORT = "too short"

# The clip was long enough and loud enough to be a sentence, but there was no speech in it. ONE
# string for every way we reach that conclusion (whisper's language score, the boilerplate word
# list, an empty transcript) because callers act on it rather than print it: the huddle counts
# consecutive occurrences to decide when to stop trusting the microphone. Kept in plain words —
# nobody cares which of the three traps fired.
NO_SPEECH = "I didn't hear anything"

# The longest single clip the recorder will ever produce (see the `-t` flag in `start`). Ten minutes
# is far beyond any real hold — the longest measured real dictation is ~60s — and short
# enough that an orphaned recorder costs ~20 MB and ten minutes of open microphone instead of hours.
MAX_CLIP_SECONDS = 600

# What whisper emits when handed near-silence: it does not return "", it confidently returns one of
# its training-set boilerplate lines. Untrapped, these get TYPED INTO YOUR DOCUMENT, which
# is the worst failure this tool has — silence should produce nothing, never words you did not
# say. Matched on the whole (stripped, lowercased) transcript only, so a sentence that genuinely
# contains "thank you" is untouched.
#
# NARROWED, after it ate real utterances. It used to hold "thank you", "you", "so" and "bye" — all
# four are things a person says as a complete sentence, and swallowing them looked to the user like
# the microphone had failed. That list was doing the job `QUIET_DBFS` now does properly, from the
# waveform and before whisper is ever asked: silence no longer reaches the transcriber at all, so
# the blocklist no longer has to guess whether a short polite sentence was real.
#
# What stays is only what a person does NOT say: subtitle-rip credits and literal audio markers.
# The bias is deliberate and one-directional — a hallucination that slips through is visible and
# deletable, while a real sentence that is swallowed is invisible and looks like broken hardware.
_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "thanks for watching!",
        "thanks for watching.",
        "[blank_audio]",
        "(silence)",
        "untertitel von stephanie geiges",
        "untertitel der amara.org-community",
        "untertitelung aufgrund der amara.org-community",
        "amara.org",
        # THE SAME BOILERPLATE, IN THE LANGUAGES IT INVENTS. Whisper does not answer silence with
        # nothing; it answers with the credit line of whatever it was trained on, and which
        # language that lands in is a coin toss. Live: a 1.8s desk bump came back as
        # "ご視聴ありがとうございました" (thank you for watching) and was typed into a terminal.
        # The structural guards above it are the real fix; this is the list for the exact strings
        # already seen, and it costs nothing to carry.
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "おやすみなさい",
        "字幕by索兰娅",
        "字幕由amara.org社区提供",
        "字幕志愿者 李宗盛",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
        "多谢您的观看",
        "감사합니다",
        "구독과 좋아요 부탁드립니다",
        "sous-titres réalisés par la communauté d'amara.org",
        "subtítulos realizados por la comunidad de amara.org",
        "sottotitoli e revisione a cura di qtss",
        "legendas pela comunidade amara.org",
    }
)


# Whisper also annotates NON-SPEECH sound rather than returning nothing: "*sad*", "[MUSIC]",
# "(wind blowing)", "♪♪♪". Observed live in a quiet room (it produced "*sad*"). These
# are an open CLASS, not a word list — a blocklist would need a new entry forever — so the whole
# class is matched structurally: a transcript that is ENTIRELY one bracketed/asterisked annotation
# was a description of a sound, not something you said, and must never be typed.
_ANNOTATION_ONLY = re.compile(r"^[\s♪]*[\*\[\(]([^\]\)\*]*)[\*\]\)][\s♪.]*$")

# An annotation is a LABEL ("sad", "wind blowing", "MUSIC"), never a sentence. Without this cap the
# trap also eats a real dictated line that happens to be fully parenthesised — "(That said, ship it
# anyway.)" — which is the worse bug of the two: a hallucination that slips through is visible and
# deletable, whereas silently swallowing what you actually said looks like the mic failed.
# Bias deliberately toward letting text through.
_MAX_ANNOTATION_WORDS = 3


def is_hallucination(text: str) -> bool:
    """True if ``text`` is whisper's output for silence rather than something that was said.

    Three shapes: the fixed boilerplate lines it emits for pure silence, a transcript of nothing
    but music notes, and the open class of non-speech ANNOTATIONS it emits for room noise. All
    three must be trapped — any of them typed into your document is a word you did not say.
    """
    stripped = text.strip().lower().strip("♪ ")
    # Bare "♪♪♪" with no brackets around it, which the annotation pattern below cannot match
    # because that one requires a bracket or an asterisk. Once the notes and the whitespace are
    # removed there is nothing left, so there was no speech in the clip.
    if not stripped:
        return bool(text.strip())
    if stripped in _HALLUCINATIONS:
        return True
    match = _ANNOTATION_ONLY.match(text.strip())
    if match is None:
        return False
    return len(match.group(1).split()) <= _MAX_ANNOTATION_WORDS


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_DOUBLED_PUNCT = re.compile(r"([,;:])\s*([.!?])")
# A filler strip can leave the sentence dangling on the comma that preceded it ("...fixed, you
# know." -> "...fixed,"). Invisible on a Slack echo; sloppy when it is typed into a document.
_DANGLING_TAIL = re.compile(r"[,;:]+\s*$")


def tidy(transcript: str) -> str:
    """Deterministic cleanup of a raw transcript. VERBATIM unless ``stripFillers`` is turned on.

    Costs ~0 ms, which is why it — and not an LLM — runs on the hot path. The judgement calls an
    LLM would make (resolving "no wait, scratch that", reflowing into bullets) live in
    :func:`polish`, off by default.

    **Filler stripping is OFF by default, and that is a correction, not a preference.** It shipped
    on, and it deleted a leading "hey" — exactly as designed, and still wrong. Somebody dictating
    "hey, it seems like our murmurflow is still not working" watched the first word of their own
    sentence vanish, with no way to know why or that a setting existed. The contract of a dictation
    tool is that what you said is what appears. A tool that silently edits you is not a faster
    keyboard, it is an unpredictable one. An "um" left in is visible and deletable in one keystroke;
    a word removed is invisible, and you only find it by re-reading your own sentence.
    """
    stripping = config.flag("stripFillers", False, cfg=_cfg())
    text = strip_fillers(transcript) if stripping else transcript.strip()
    # WHISPER SEGMENTS ARE NOT LINE BREAKS, and this line is the whole fix for "it puts line breaks
    # in random places". whisper-server returns one `\n` per SEGMENT, and it segments on decoder
    # budget and pauses — never on sentences — so a single dictated sentence comes back as
    # "...than in the\n other, you know,\n" and the paste lands with a hard break mid-clause.
    # Nobody can speak a newline, so collapsing them is not an edit to what was said: it is undoing
    # a transport artefact, and it belongs OUTSIDE the `stripping` branch. It used to be inside
    # `strip_fillers`, which is exactly how it went missing — defaulting filler-stripping off (the
    # right call) silently took the flattening with it, and the two have nothing to do with each
    # other.
    text = " ".join(text.split())
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DOUBLED_PUNCT.sub(r"\2", text)
    if stripping:
        # Seam repair, and ONLY meaningful after a strip: it exists to close the ", ," a removed
        # filler leaves behind. Run unconditionally it would quietly eat a trailing comma somebody
        # dictated on purpose, which is the same class of bug as the strip itself.
        ended = transcript.rstrip().endswith((".", "!", "?"))
        text = _DANGLING_TAIL.sub("." if ended else "", text)
    return text.strip()


# The instruction prepended to the transcript when `polishCommand` is a bare model runner that
# takes a prompt rather than a formatter that already knows its job. It is a constant rather than a
# setting because getting it wrong is how polish starts ANSWERING dictated questions instead of
# cleaning them — the "do not answer anything" clause is the load-bearing line, not boilerplate.
POLISH_PROMPT = (
    "Clean up this dictated text for direct insertion into a document. Rules:\n"
    '- Resolve spoken self-corrections ("go left, no wait, right" -> "go right").\n'
    "- Fix punctuation, capitalisation and obvious homophones.\n"
    "- Keep the speaker's own words, language and tone. Do NOT summarise, translate,"
    " add or answer anything.\n"
    "- Output ONLY the cleaned text, with no preamble, quotes or commentary.\n\n"
)


def polish(text: str, *, timeout: float = 20.0) -> str:
    """OPTIONAL cleanup through a command of your choosing. Off unless ``polishCommand`` is set.

    Resolving spoken self-corrections ("go left, no wait, right") is the one thing a language model
    does that a regex cannot, and it is most of what people mean when they say a paid dictation app
    "reads their mind". It is off by default because it is not free: spawning a model costs seconds
    against a ~0.6s transcription, so turning it on trades chat-speed dictation for document-quality
    dictation. That is a real trade, and it is yours to make rather than ours.

    ``polishCommand`` is a shell command that receives the transcript on **stdin** and prints the
    cleaned text on stdout. Anything that shape works::

        "polishCommand": "claude -p --model haiku"
        "polishCommand": "ollama run llama3.2"
        "polishCommand": "llm -m gpt-4o-mini"

    :data:`POLISH_PROMPT` is prepended unless the command contains ``{prompt}``, in which case you
    are telling us the instruction is already handled on your side and we send the bare transcript.

    **This is the one place your words can leave the machine, and only because you sent them
    there.** A local runner keeps everything local; a command pointing at a hosted API does not. The
    default of "unset" means the answer to "does this tool phone home" stays no until you change it.

    Returns ``text`` unchanged on ANY failure — a missing binary, a non-zero exit, a timeout, empty
    output. A broken polish command degrades to the deterministic transcript; it never costs you
    the sentence you just spoke.
    """
    command = str(_cfg().get("polishCommand", "") or "").strip()
    if not text.strip() or not command:
        return text
    payload = text if "{prompt}" in command else POLISH_PROMPT + text
    command = command.replace("{prompt}", POLISH_PROMPT.strip())
    try:
        # shell=True because the value IS a shell command line the user wrote in their own config —
        # pipes and flags are the point, and there is no untrusted input here to inject with. The
        # transcript goes on stdin precisely so it never touches the command line and cannot be
        # mangled by quoting, however many apostrophes and umlauts it contains.
        proc = subprocess.run(  # noqa: S602 — user's own configured command, by design
            command,
            shell=True,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return text
    out = (proc.stdout or "").strip()
    return out if proc.returncode == 0 and out else text


# --- injection --------------------------------------------------------------------------------

# How long to let the target app read the pasteboard before we put your own clipboard back.
# Apps read the pasteboard synchronously on Cmd-V, but the event itself is delivered
# asynchronously, so this covers event delivery — and, for a long transcript, the target's own
# insertion of it.
_PASTE_SETTLE = 0.35

# ...and the ceiling on that wait. Only reached by a transcript of several thousand characters.
_PASTE_SETTLE_MAX = 2.0


def paste_settle(text: str) -> float:
    """How long to wait before restoring the clipboard, for a paste of ``text``.

    Grows with the text, because the failure it prevents does. A fixed 0.35s was measured against
    short sentences and is fine for them; a minute of dictation is ~1000 characters, and a target
    that inserts it through a pipe — a terminal writing into a pty, a pty feeding a TUI — is still
    consuming the pasteboard's contents when a fixed wait has already put the old clipboard back.
    What the user sees then is the first part of their sentence and nothing after it, which is
    exactly the report this exists to answer ("when I talk longer, only a few words paste").

    The wait costs nothing anybody is looking at: the paste is already on screen, and this only
    delays the restore of a clipboard they are not using this instant. Capped at
    :data:`_PASTE_SETTLE_MAX` so a pathological transcript cannot hold the clipboard hostage.
    """
    return min(_PASTE_SETTLE_MAX, _PASTE_SETTLE + len(text) / 1000.0)


#: What somebody is told when their words could not be typed for them.
#:
#: It names the COST as well as the recovery, and that is the whole point of it being one string.
#: Both ways a paste can fail put the transcript on the clipboard, which means both of them
#: DESTROY whatever was on it — and the message used to stop at "press paste", which reads as free.
#: It is not free, and it is least free exactly when it fires: Secure Input is macOS's own signal
#: that a password field has focus this second, so the thing being overwritten is more likely than
#: usual to be a password somebody had just copied out of their manager. Putting it back is not
#: available to us (there is no lossless clipboard read here, and no event that says "they have
#: pasted now, it is safe"), so the honest move is to say what happened rather than to be quiet
#: about it.
PASTE_YOURSELF = "Your words are on the clipboard now, in place of what was there. Press paste."


def permission_hint() -> str:
    """How to grant whatever permission typing needs here, or ``""`` when there is nothing to grant.

    Empty is a real answer and not a missing one: Windows has no TCC. The caller prints the row
    either way — see the doctor.
    """
    return platforms.permission_hint()


#: How long a slice of the daemon log :func:`last_paste_verdict` reads. Days of dictation, and
#: bounded so a log nobody ever rotates cannot turn a health check into a full file read.
_LOG_TAIL_BYTES = 65_536

#: The ceiling on the daemon log, and how much of it survives a trim.
#:
#: **Nothing rotates this file**, and that is not a detail on a background daemon: launchd and Task
#: Scheduler both redirect the listener's stdout into it and then never look at it again. A clip
#: costs about 120 bytes, so ordinary use grows it by a few MB a year and nobody would ever notice.
#: The case that is not ordinary is the one that matters: a listener that cannot start writes its
#: reason and exits, `KeepAlive` starts it again ten seconds later (`ThrottleInterval`), and it
#: does that all night. Nobody is dictating in that state, so a trim that only ran per clip would
#: never run at all — which is why this also runs once at daemon start, where a restart loop is
#: guaranteed to reach it.
#:
#: A megabyte is roughly eight thousand clips, which is months of real use, and the tail that
#: survives keeps the last few thousand. Cheap, because the ordinary path is one `stat` and a
#: comparison.
LOG_MAX_BYTES = 1_000_000
LOG_KEEP_BYTES = 400_000


def trim_log(*, cap: int = LOG_MAX_BYTES, keep: int = LOG_KEEP_BYTES) -> bool:
    """Drop the oldest lines once the daemon log passes ``cap``. ``True`` if it had to. Never raises.

    Rewrites the file with its last ``keep`` bytes, from the first line boundary inside them, so no
    half line is ever left at the top of the log.

    Best-effort ON PURPOSE, in both directions. The service holds this file open in append mode and
    may write between the read and the rewrite, so a line can be lost here; a lost diagnostic line
    is a fair price for a log that cannot grow without limit, and the alternative is a lock that a
    keyboard daemon has to take on its hot path. Any failure at all leaves the file exactly as it
    was: a housekeeping routine must never be the reason dictation stops.
    """
    path = config.log_path()
    try:
        if path.stat().st_size <= cap:
            return False
        with path.open("rb") as handle:
            handle.seek(-keep, os.SEEK_END)
            tail = handle.read()
        # Start at a line boundary. Without this the log opens mid-sentence, and the first thing
        # anybody reads when they finally look at it is a fragment.
        newline = tail.find(b"\n")
        if newline != -1:
            tail = tail[newline + 1 :]
        with path.open("wb") as handle:
            handle.write(tail)
    except OSError:
        return False
    return True


def last_paste_verdict() -> bool | None:
    """Did the DAEMON's most recent paste actually land? ``None`` if it has not tried one.

    **The only ground truth there is about the typing permission**, and the reason it is read off a
    log rather than asked of the OS: ``AXIsProcessTrusted`` answers for the process that ASKS, and
    a probe spawned from this CLI is attributed by TCC to whatever is responsible for the CLI — a
    terminal, an editor, whatever launched it — and never to the ``.app`` the launchd agent runs
    as. So the question "does the daemon have Accessibility" came back NOT GRANTED on a Mac where
    dictation had been typing into Slack all afternoon, and the health report sent its owner off to
    switch on a switch that was already on. A confidently wrong diagnostic is worse than none.

    The daemon already writes what happened to every paste, so its last attempt answers the
    question outright — text at the cursor, or the refusal — with nothing to probe and no guessing.

    Secure Input is deliberately NOT read as a refusal: it is a password field somebody had focused
    for a moment, it clears on its own, and it has its own row.
    """
    try:
        with config.log_path().open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
            lines = handle.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("[OK] ") and " · → " in line:
            return True
        if "not permitted to control this Mac" in line:
            return False
    return None


def secure_input_active() -> bool:
    """True if the OS is refusing synthetic keyboard events right now.

    macOS has Secure Input (a password field, some terminals) and while it is on a paste silently
    does nothing — the single most confusing failure this feature can have ("it heard me but typed
    nothing"). Windows has no equivalent, so this is always False there and the message it guards
    is never shown.
    """
    return bool(platforms.input_blocked())


def clipboard_set(text: str) -> bool:
    """Put ``text`` on the clipboard. ``True`` on success.

    Only used when injection is refused before it starts. Everything on the happy path goes through
    :func:`inject`, which has to hold the old clipboard anyway.
    """
    return platforms.clipboard_set(text)


def inject(text: str) -> tuple[bool, str, str]:
    """Type ``text`` into whatever app has focus. ``(ok, problem, note)``; never raises.

    ``note`` is a diagnostic for the daemon log and nothing else: which app the keystroke went to,
    and whether the whole transcript was still on the clipboard when the target read it. A paste
    that lands in full and one that lands as its first eight words returned the identical
    ``(True, "")`` before this, which is why "when I talk longer, only a few words paste" could be
    reported for weeks without the log holding a single fact about it.

    Both platforms do the same thing for the same reason — put the text on the clipboard, send the
    paste chord, wait, put the old clipboard back — because pasting is O(1) where typing the string
    character by character is O(chars), and because a synthetic per-character keystroke is re-mapped
    through the TARGET's keyboard layout, which mangles every umlaut in a German dictation.

    The wait before the restore is :func:`paste_settle` and it grows with the text: the restore is
    what ENDS the window in which the target may read the clipboard, and a long transcript is still
    being inserted when a fixed wait has already closed it.
    """
    text = (text or "").strip()
    if not text:
        return False, "nothing to type", ""
    blocked = platforms.input_blocked()
    if blocked:
        # The copy has to happen HERE, not be assumed. This branch returns before anything is
        # pasted, and it used to promise "the text is on your clipboard" over a clipboard nothing
        # had ever been written to — so the recovery paste gave back whatever was there before, and
        # the dictation was simply gone.
        copied = platforms.clipboard_set(text)
        return False, f"{blocked} {PASTE_YOURSELF if copied else 'The text is lost.'}", ""
    ok, problem, note = platforms.inject(text, paste_settle(text))
    if ok:
        return True, "", note
    # The same sentence, because it is the same event: the injection script writes the clipboard
    # before it sends the chord and aborts before the restore, so a refused paste leaves the
    # transcript there over whatever the user had copied — exactly as the blocked branch does.
    return False, f"{problem} {PASTE_YOURSELF}", note


# --- the flow ---------------------------------------------------------------------------------


# Peak amplitude below which a clip is not quiet speech but a SILENT STREAM. macOS does not fail a
# denied microphone — CoreAudio hands back digital silence — so a process without the grant records
# a perfectly valid, perfectly empty wav, and whisper answers it with confident nonsense ("Nibble,
# Nibble, Nibble"). Room tone from a real mic sits around -46 dBFS; true silence is -90 or below.
SILENT_DBFS = -70.0

# Below this language score, whisper was not listening to speech — see :class:`Heard`. Measured on
# large-v3-turbo: every real utterance scored >= 0.969, every silent or
# noisy clip <= 0.453. 0.75 sits in the empty middle with ~0.22 of margin on both sides.
#
# This is the trap that the word-list in `_HALLUCINATIONS` structurally cannot be: whisper answers
# silence in a DIFFERENT invented language each time. Hit live: the key was pressed, nothing was
# said, and back came two sentences of invented Icelandic ("Ennum, hvað
# er hann?") as though it were a question. No blocklist can grow fast enough to cover that; asking
# whisper how sure it was covers all of it at once.
SPEECH_CONFIDENCE = 0.75


def spoken_languages() -> frozenset[str]:
    """The languages you actually speak (``languages``), or empty = accept every one.

    The last gap `SPEECH_CONFIDENCE` cannot close. It asks "how sure are you it is SOME language",
    and on room tone whisper is sometimes very sure indeed: two clips on 2026-08-17, at -21 and
    -26 dBFS, came back as fluent invented Icelandic and were typed into the operator's window —
    above the quiet floor, above the confidence bar, and not on any blocklist.

    Naming the languages you speak turns that from an open class into a closed one, and it is the
    only check that scales: a hallucination can invent any of whisper's ~99 languages, but a person
    speaks two or three. Deliberately NOT derived from ``language``, which pins the decoder and
    therefore only ever has ONE value — somebody who dictates German and English cannot use it, and
    this is exactly that person. Empty by default, so nothing changes until you say what you speak.
    """
    raw = _cfg().get("languages", [])
    values = raw if isinstance(raw, list) else str(raw).split(",")
    # Written as a code OR as a name: somebody who types `["english"]` means the same thing as
    # `["en"]`, and a gate that silently disagrees with them rejects everything they say.
    return frozenset(language_code(str(v)) for v in values if str(v).strip())


# Peak amplitude below which the microphone WAS working and nobody spoke into it — the trigger got
# pressed and then the room was recorded. `SILENT_DBFS` cannot catch this: room tone is a real
# signal, ~35 dB above digital silence, and `SPEECH_CONFIDENCE` does not catch all of it either.
# Whisper is not asked "was that speech"; `detected_language_probability` answers "WHICH language",
# and on room tone it is sometimes very sure indeed — which is how two sentences of confident
# Icelandic got pasted into whatever window happened to be in front.
#
# Measured, not guessed, from a day of real dictation: eleven clips of 4 to 60 seconds peaked
# between -15 and -5 dBFS, and the quietest was -15. A silent room on the same machine and the same
# microphone measured -47. -35 sits in that gap with 20 dB of margin under the quietest real
# sentence and 12 dB over the room. Raise it if a whisper at arm's length gets dropped; the level
# of every clip is printed in the daemon log, so this is tunable from evidence rather than feel.
#
# -30 and not -35: a second measurement of the same "silent" room came back at -38, because a room
# is not a constant. 8 dB over the loudest room seen, 15 dB under the quietest sentence seen. A
# far-field microphone in a big room is a different machine from this one, so `quietFloor`
# overrides it rather than leaving somebody with a tool that never hears them and no way to say so.
QUIET_DBFS = -30.0


def quiet_floor() -> float:
    """The level below which a clip is a room, not a sentence. ``quietFloor`` overrides."""
    raw = _cfg().get("quietFloor")
    if isinstance(raw, (int, float)):
        return float(raw)
    return QUIET_DBFS


#: What a clip with no voice in it is called. Like `TOO_SHORT` this is a NON-EVENT, not a failure:
#: nothing was asked for, so nothing gets a failure tone. A noise reporting that nothing worked,
#: when nothing was attempted, is the "beeping out of nowhere" that makes a daemon feel broken.
NOTHING_SAID = "nothing was said"


def peak_dbfs(wav: Path) -> float:
    """Peak amplitude of a 16-bit PCM wav in dBFS; ``-inf`` for silence or an unreadable file.

    Pure stdlib (:mod:`wave` + :mod:`array`), cheap enough for the hot path: one pass over a few
    seconds of 16 kHz mono. Worth it because "the transcript is wrong" and "we recorded nothing at
    all" are indistinguishable in a log and have completely different fixes.
    """
    try:
        with wave.open(str(wav), "rb") as handle:
            if handle.getsampwidth() != 2:
                return 0.0  # not 16-bit: no opinion rather than a wrong one
            frames = handle.readframes(handle.getnframes())
    except Exception:  # noqa: BLE001 — a diagnostic must never break a dictation
        return 0.0
    if not frames:
        return float("-inf")
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if not samples:
        return float("-inf")
    # `max(max(...), -min(...))` and not `max(abs(s) for s in samples)`: both find the same peak,
    # but the generator runs one Python-level loop per SAMPLE — a minute of dictation is ~1M of
    # them — where two array scans stay in C. This sits on the hot path between the key coming up
    # and the transcribe starting, which is the one stretch of the product a person is waiting on.
    peak = max(max(samples), -min(samples))
    if peak == 0:
        return float("-inf")
    return 20 * math.log10(min(peak, 32768) / 32768.0)


def audio_seconds(wav: Path) -> float:
    """How long the RECORDING is, from its own header. ``0.0`` if it cannot be read.

    Not the same number as :attr:`Recording.seconds`, and the gap between them is the point.
    ``Recording.seconds`` is wall clock from spawning ffmpeg to stopping it; this is how much
    audio actually reached the file. A daemon log that prints only the first cannot tell "whisper
    mis-heard 21 seconds of speech" from "we captured 15 of the 21 seconds you spoke", which are
    the two halves of every report that a dictation came back short, and they have completely
    different fixes. Both are printed now, and only when they disagree — see the daemon loop.
    """
    try:
        with wave.open(str(wav), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except Exception:  # noqa: BLE001 — a diagnostic must never break a dictation
        return 0.0


@dataclass(frozen=True)
class Result:
    """What one dictation produced, for the CLI and the huddle loop to report on."""

    text: str
    seconds: float
    transcribe_ms: int
    injected: bool
    problem: str = ""
    peak_dbfs: float = 0.0
    #: How much audio actually landed in the file, against ``seconds`` of wall clock. See
    #: :func:`audio_seconds` — the two disagreeing is a capture fault, not a model fault.
    audio_seconds: float = 0.0
    #: What the paste reported about itself: the app it went to, and whether the whole transcript
    #: was still on the clipboard for it. Empty when nothing was pasted. See :func:`inject`.
    paste_note: str = ""
    #: Which transcriber answered — see :attr:`Heard.warm`. ``None`` when none was asked.
    warm: bool | None = None


def log_line(result: Result) -> str:
    """The one line the daemon writes per clip: everything about the clip, and NOT what you said.

    **The transcript itself is deliberately not in it, and that is a privacy fix rather than a
    tidy-up.** It used to be, quoted in full at the end of the line — and under the installed agent
    that line does not scroll past, it is appended to ``~/.murmurflow/listen.log``, which nothing
    ever rotates or trims. So every sentence dictated since the day of install sat in plaintext on
    disk forever: the password read aloud into a form, the medical detail, the message deleted
    before sending. A tool whose whole promise is that your voice never leaves the machine cannot
    also keep a permanent written record of everything said into it, and the README promised in one
    section that the transcript is never logged while admitting in another that it is.

    Everything a person actually debugs with survives, because none of it is the words:

    * the hold, and separately how much audio really landed — the two disagreeing is a CAPTURE
      fault, not a model fault, and printing only the first made every such report unanswerable.
      Shown only when they differ by more than the ~0.6s of device start-up and stop.
    * the peak level. "3.1s at -52 dBFS came back as four words" says the microphone was barely
      picking anything up; "3.1s at -14 dBFS" says it was loud and clear and the model is at fault.
    * whether the COLD path answered. A wedged warm server answers every request with an error and
      every clip silently takes the slower path, where the silence gates are weaker — which is how
      a bump on the desk was transcribed as Japanese and typed.
    * how many characters came back, and which app the paste went to and whether the whole
      transcript was still on the clipboard when it got there. A paste that quietly delivers half a
      sentence is invisible in every other fact on the line, and it is the failure people report.

    ``keepAudio`` brings the words back, because it is already the switch that means "I am
    debugging this one, keep the evidence" — it is what keeps the clip itself. Somebody comparing a
    bad transcription against the audio that produced it needs both, and they have opted in to both.
    """
    captured = ""
    if result.audio_seconds and result.seconds - result.audio_seconds > 1.0:
        captured = f" (captured {result.audio_seconds:.1f}s)"
    went = f" · {result.paste_note}" if result.paste_note else ""
    how = " · cold" if result.warm is False else ""
    said = f' · "{result.text}"' if config.flag("keepAudio", False, cfg=_cfg()) else ""
    return (
        f"[OK] {result.seconds:.1f}s{captured} · {result.peak_dbfs:.0f}dBFS · "
        f"{result.transcribe_ms}ms{how} · {len(result.text)} chars{went}{said}"
    )


def finish(rec: Recording | None = None, *, paste: bool = True) -> Result:
    """Stop the in-flight recording, transcribe it, clean it up and type it. Never raises.

    This is the whole hot path, and the order matters: the audio file is deleted as soon as it has
    been transcribed (before the optional polish call and before the paste), so a crash anywhere
    downstream cannot leave your voice on disk.

    The "got it" cue fires HERE, the instant the microphone closes — not after the paste. Cued by
    the caller afterwards it arrived a full transcribe behind the text it was acknowledging, so the
    user saw the words land and then heard a tone about them, which reads as a glitch rather
    than as feedback. Every surface (the daemon, ``murmurflow toggle``, a huddle) goes through this
    function, so cueing at the seam is also the only way all three stay in step.
    """
    rec = rec or current()
    if rec is None:
        return Result("", 0.0, 0, False, "nothing was recording")
    seconds = rec.seconds
    wav = stop(rec)
    if wav is None:
        return Result("", seconds, 0, False, "no audio was captured")
    if seconds < MIN_CLIP_SECONDS:
        wav.unlink(missing_ok=True)
        # The advice has to match the GESTURE. "hold the key while you talk" is actively wrong
        # in tap mode - which is now the default - and being told to do the thing you are not
        # supposed to do is worse than being told nothing.
        advice = (
            "tap again to stop, not straight away"
            if double_tap_mode()
            else "hold the key while you talk"
        )
        return Result("", seconds, 0, False, f"{TOO_SHORT} — {advice}")
    # After the two ways this can still be a non-event, so a failure cues once rather than being
    # contradicted by a "got it" a moment earlier — and before peak_dbfs and the ~600ms transcribe,
    # which is the whole point: the tone marks the microphone closing, not the text arriving.
    cue(CUE_DONE)
    level = peak_dbfs(wav)
    captured = audio_seconds(wav)

    def retire() -> None:
        # `keepAudio` keeps the LAST clip (one file, overwritten) so a bad transcription can be
        # listened to instead of guessed about — the difference between "the model is wrong" and
        # "we recorded half a sentence" is audible in two seconds and invisible in a log. Off by
        # default: your voice does not outlive its transcription unless you ask.
        if config.flag("keepAudio", False, cfg=_cfg()):
            with contextlib.suppress(OSError):
                shutil.copyfile(wav, _scratch_dir() / "last.wav")
        wav.unlink(missing_ok=True)

    # BOTH LEVEL CHECKS RUN BEFORE THE TRANSCRIBE, and that ordering is the fix rather than a
    # tidy-up: whisper cannot be trusted to answer "was anyone speaking" — it answers "which
    # language is this", confidently, about a recording of a room. Deciding from the waveform
    # first means the invented sentence is never generated, so it can never be pasted. It also
    # returns a silent press in milliseconds instead of after a four-second transcription of noise.
    #
    # A silent STREAM is a different thing from a silent room and keeps its own message: macOS does
    # not fail a denied microphone, it hands back digital silence, so "could not make out any
    # speech" would send somebody off to tune a model when there is no audio at all.
    if level < SILENT_DBFS:
        retire()
        return Result(
            "",
            seconds,
            0,
            False,
            f"the microphone returned silence ({level:.0f} dBFS) — this process has no working "
            f"microphone input. Check System Settings > Privacy & Security > Microphone for "
            f"{Path(sys.executable).name}.",
            level,
            captured,
        )
    if level < quiet_floor():
        retire()
        return Result("", seconds, 0, False, f"{NOTHING_SAID} ({level:.0f} dBFS)", level, captured)

    started = time.monotonic()
    try:
        heard = transcribe(wav)
        raw = heard.text
    finally:
        retire()
    elapsed_ms = int((time.monotonic() - started) * 1000)
    # Whisper's own verdict, and the only one that catches an invented sentence in an invented
    # language. Checked before the word-list because it subsumes it.
    if heard.confidence < SPEECH_CONFIDENCE:
        return Result("", seconds, elapsed_ms, False, NO_SPEECH, level, captured, warm=heard.warm)
    # And the language it landed on, if you have said which ones you speak. Confidence answers
    # "is this SOME language" and passes fluent invented Icelandic; this answers "is it one of
    # YOURS". Only ever consulted when the server reported a language it detected — a pinned
    # `language` or the cold path report none, and an unanswered question is not a refusal.
    allowed = spoken_languages()
    if allowed and heard.language and heard.language not in allowed:
        return Result(
            "",
            seconds,
            elapsed_ms,
            False,
            f"{NO_SPEECH} — heard {heard.language}, which is not one of yours ({', '.join(sorted(allowed))})",
            level,
            captured,
            warm=heard.warm,
        )
    if is_hallucination(raw):
        return Result("", seconds, elapsed_ms, False, NO_SPEECH, level, captured, warm=heard.warm)
    text = tidy(raw)
    if not text or is_hallucination(text):
        return Result("", seconds, elapsed_ms, False, NO_SPEECH, level, captured, warm=heard.warm)
    text = polish(text)  # a no-op unless `polishCommand` is configured
    if not paste:
        return Result(text, seconds, elapsed_ms, False, "", level, captured, warm=heard.warm)
    ok, problem, note = inject(text)
    return Result(text, seconds, elapsed_ms, ok, problem, level, captured, note, heard.warm)


def toggle(*, paste: bool = True) -> Result | None:
    """Press-once-to-start, press-again-to-stop. ``None`` means a recording just STARTED.

    The entry point for a hotkey that only fires on key-down (a Shortcuts/Automator binding), which
    structurally cannot do hold-to-talk. The hold-to-talk daemon uses :func:`start`/:func:`finish`
    directly instead.
    """
    if current() is not None:
        return finish(paste=paste)
    start()
    return None


# --- the daemon -------------------------------------------------------------------------------

# Short system sounds used as the only UI. A menu-bar HUD would need a compiled app bundle; a sound
# needs nothing, cannot steal focus, and is actually BETTER for this job — you are looking at the
# app you are dictating into, not at our indicator. Ready = the mic is live, start talking
# now. Done = the mic just closed, stop talking (NOT "the text has arrived" — the text arriving
# says that far better than a tone can). Fail = something went wrong.
CUE_READY = "ready"
CUE_DONE = "done"
CUE_FAIL = "fail"

#: ``cue`` values that mean "make no sound at all" — see :func:`cues_muted`.
CUE_OFF = frozenset({"off", "none", "silent", "mute", "false", "0"})

# The cue log's ceiling — see `_note_cue`. Trimmed to the last 200 plays, which is days of use.
_CUE_LOG_MAX_BYTES = 200_000

#: The OS's own alert sounds, where it has any to borrow. Empty on a platform that does not name
#: its sounds by PATH (Windows names them by event key), and the generated tones below are then the
#: only cue — which is no loss, because every stock system sound is an ALERT, mixed to interrupt.
_SYSTEM_CUES = platforms.system_cues()

# The default cue is generated rather than borrowed, because every sound in /System/Library/Sounds
# is an ALERT: designed to interrupt, mixed loud, and instantly recognisable as "a Mac just told
# you something". Dictation needs the opposite — you are looking at the app you are talking into,
# so the cue is your only signal, and it should register without announcing itself.
#
# Two things separate a cue that sounds like an INSTRUMENT from one that sounds like a computer
# beeping, and neither is the note you pick:
#
#   * the envelope. A struck object never fades IN — it is loudest at the moment of impact and
#     decays away. An attack of a few ms with an exponential tail is the whole difference; the
#     symmetric raised-cosine swell the first version used is why it read as flute-ish and cheap.
#   * the partials. A bare sine is a test tone. Real bells and bars ring at INHARMONIC ratios
#     (2.76x for a tubular bell, ~3.9x for a marimba bar), which is what the ear hears as wood or
#     glass rather than as an oscillator.
#
# Notes carry an absolute start, so a second note begins while the first is still ringing and the
# two overlap into one gesture instead of sounding like two separate beeps.
_CUE_RATE = 44100
_CUE_REVISION = 2  # bump to reissue the tones — cached files are named for it
_CUE_FADE = 0.004  # forced fade to zero at the tail, so no preset can ever end on a click


@dataclass(frozen=True)
class _Timbre:
    """What the struck thing is made of. ``partials`` are (frequency ratio, relative amplitude)."""

    partials: tuple[tuple[float, float], ...]
    attack: float = 0.004  # seconds to full amplitude; 0 would click
    decay: float = 0.45  # exponential time constant, as a fraction of the note length
    peak: float = 0.16  # of full scale; a macOS alert sound sits around 0.8
    swell: bool = False  # raised-cosine instead of struck — the original `soft` shape


# Each preset is (timbre, {cue: ((frequency, start, length), ...)}). Rising = listening,
# falling = the mic closed, low and slow = something went wrong — that grammar holds across all of
# them, so switching preset never changes what a sound MEANS.
_CUE_PRESETS: dict[str, tuple[_Timbre, dict[str, tuple[tuple[float, float, float], ...]]]] = {
    # Tubular-bell partials, bright and clean. The most "premium" of the set, and it was the default
    # until somebody dictated all day with it: a bright chime twice a sentence, hundreds of times,
    # is the one setting that stops being a nice detail and becomes a reason to turn the tool off.
    "glass": (
        _Timbre(((1.0, 1.0), (2.76, 0.40), (5.40, 0.14)), decay=0.40),
        {
            CUE_READY: ((1174.66, 0.000, 0.30), (1567.98, 0.070, 0.38)),  # D6 -> G6
            CUE_DONE: ((1567.98, 0.000, 0.26), (1046.50, 0.065, 0.40)),  # G6 -> C6
            CUE_FAIL: ((587.33, 0.000, 0.34), (392.00, 0.110, 0.50)),  # D5 -> G4
        },
    ),
    # Wooden bar: warmer, drier, shorter. Reads as a soft knock rather than a chime.
    "marimba": (
        _Timbre(((1.0, 1.0), (3.94, 0.28), (9.60, 0.06)), decay=0.26),
        {
            CUE_READY: ((783.99, 0.000, 0.22), (1174.66, 0.060, 0.28)),  # G5 -> D6
            CUE_DONE: ((1046.50, 0.000, 0.20), (698.46, 0.055, 0.30)),  # C6 -> F5
            CUE_FAIL: ((392.00, 0.000, 0.26), (293.66, 0.090, 0.38)),  # G4 -> D4
        },
    ),
    # One short low blip. The least you can play and still be understood — nearly subliminal, which
    # is exactly what a sound you will hear a few hundred times a day has to be. The DEFAULT.
    "pebble": (
        _Timbre(((1.0, 1.0), (2.0, 0.10)), decay=0.30, peak=0.14),
        {
            CUE_READY: ((659.26, 0.000, 0.16), (987.77, 0.045, 0.20)),  # E5 -> B5
            CUE_DONE: ((659.26, 0.000, 0.18),),  # single note
            CUE_FAIL: ((329.63, 0.000, 0.22), (246.94, 0.080, 0.30)),  # E4 -> B3
        },
    ),
    # The original: pure sine pair under a symmetric swell. Kept because someone may prefer it.
    "soft": (
        _Timbre(((1.0, 1.0),), swell=True),
        {
            CUE_READY: ((784.00, 0.000, 0.065), (1174.66, 0.065, 0.105)),
            CUE_DONE: ((1046.50, 0.000, 0.055), (698.46, 0.055, 0.095)),
            CUE_FAIL: ((392.00, 0.000, 0.085), (311.13, 0.085, 0.150)),
        },
    ),
}
# THE DEFAULT IS THE SYSTEM SOUND (operator, after living with both). The generated presets were
# the default on the argument below — that every sound in /System/Library/Sounds is an ALERT, mixed
# to interrupt. True, and beside the point in practice: a Mac user has heard Tink and Pop for twenty
# years, so they read as "the machine acknowledged you" without being learned, and they are already
# tuned to the output device and the user's alert-volume setting, which a generated tone is not.
#
# `pebble` is the FALLBACK, not a rival: if the system sound is missing — a stripped install, or any
# machine that is not a Mac — a cue must still play. It is the only signal that the microphone is
# live, so losing it to a missing file is not an acceptable failure.
DEFAULT_CUE = "system"
FALLBACK_PRESET = "pebble"


def cue_presets() -> tuple[str, ...]:
    """Every built-in preset name — the CLI auditions these and `cue` is validated against."""
    return tuple(_CUE_PRESETS)


def _render(timbre: _Timbre, notes: tuple[tuple[float, float, float], ...]) -> array.array[int]:
    """Mix ``notes`` into one 16-bit mono buffer. Overlapping tails are summed, then clipped."""
    total = int(_CUE_RATE * max(start + length for _f, start, length in notes)) + 1
    buf = [0.0] * total
    weight = sum(amp for _ratio, amp in timbre.partials) or 1.0
    for freq, start, length in notes:
        count = int(_CUE_RATE * length)
        offset = int(_CUE_RATE * start)
        for i in range(count):
            seconds = i / _CUE_RATE
            if timbre.swell:
                envelope = 0.5 - 0.5 * math.cos(2 * math.pi * i / count)
            else:
                rise = min(1.0, seconds / timbre.attack) if timbre.attack else 1.0
                envelope = rise * math.exp(-seconds / (length * timbre.decay))
            value = sum(
                amp * math.sin(2 * math.pi * freq * ratio * seconds)
                for ratio, amp in timbre.partials
            )
            buf[offset + i] += timbre.peak * envelope * value / weight
    fade = min(int(_CUE_RATE * _CUE_FADE), total)
    for i in range(fade):
        buf[total - fade + i] *= 1.0 - i / fade
    return array.array("h", (max(-32767, min(32767, int(v * 32767))) for v in buf))


def _cue_file(kind: str, preset: str = "") -> Path | None:
    """Path to the generated tone for ``kind``, synthesising it once. ``None`` if it can't be made.

    Cached under ``~/.murmurflow/audio/cues/`` and named for the preset AND ``_CUE_REVISION``, so changing
    a tone ships the new one instead of being masked forever by a stale file, and auditioning every
    preset costs one synthesis each. Written to a temp name and renamed, because a half-written wav
    would be a permanently broken cue.
    """
    timbre_notes = _CUE_PRESETS.get(preset or FALLBACK_PRESET)
    if timbre_notes is None:
        return None
    timbre, cues = timbre_notes
    notes = cues.get(kind)
    if not notes:
        return None
    try:
        path = _scratch_dir() / "cues" / f"{preset or FALLBACK_PRESET}-{kind}-v{_CUE_REVISION}.wav"
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = _render(timbre, notes)
        tmp = path.with_suffix(".part")
        with wave.open(str(tmp), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(_CUE_RATE)
            out.writeframes(frames.tobytes())
        tmp.replace(path)
        return path
    except (OSError, ValueError, wave.Error):
        return None


def custom_cue_dir() -> Path:
    """``~/.murmurflow/audio/cues/custom/`` — drop ``ready``/``done``/``fail`` here to use your own sounds.

    The escape hatch that means we never have to be anyone's taste: any sound file downloaded from
    anywhere works, no code and no rebuild. Named files rather than a config key per cue, because
    "put three files in this folder" is one instruction instead of three.
    """
    return _scratch_dir() / "cues" / "custom"


def _custom_cue(folder: Path, kind: str) -> str:
    """A caller-supplied ``<kind>.<ext>`` in ``folder``, or ``""``. afplay reads all of these."""
    for suffix in (".wav", ".aiff", ".aif", ".m4a", ".mp3", ".caf"):
        candidate = folder / f"{kind}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return ""


def cue_preset_name() -> str:
    """What ``cue`` currently resolves to: a preset name, ``system``, or a folder path.

    Reported by ``murmurflow cues``/``doctor`` so "which sound am I hearing" is answerable without
    reading the config and re-deriving the fallback rules.
    """
    setting = str(_cfg().get("cue", "") or "").strip()
    if not setting:
        return DEFAULT_CUE
    if setting.lower() in CUE_OFF:
        return "off"
    if setting.lower() == "system" or setting.lower() in _CUE_PRESETS:
        return setting.lower()
    return setting  # a folder path, reported verbatim


def _cue_path(kind: str) -> str:
    """Resolve a cue name to a playable file. ``cue`` decides, most specific first:

    1. a file you dropped in ``~/.murmurflow/audio/cues/custom/`` — ALWAYS wins, whatever else is set, so
       a downloaded sound needs no config at all,
    2. a folder path in ``cue`` (``~/Sounds/mine``) holding ``ready``/``done``/``fail``,
    3. ``system`` — the macOS Tink/Pop/Basso,
    4. a built-in preset name (``glass``/``marimba``/``pebble``/``soft``),
    5. the default preset.

    An unreadable or unknown setting degrades to the default rather than to silence: a cue is the
    only signal that the microphone is live, so it must never be possible to lose it by
    typing the wrong thing in a config file.
    """
    setting = str(_cfg().get("cue", "") or "").strip()
    with contextlib.suppress(OSError):
        dropped = _custom_cue(custom_cue_dir(), kind)
        if dropped:
            return dropped
    if setting and setting.lower() not in {"system", *_CUE_PRESETS}:
        folder = Path(setting).expanduser()
        with contextlib.suppress(OSError):
            if folder.is_dir():
                supplied = _custom_cue(folder, kind)
                if supplied:
                    return supplied
    if not setting or setting.lower() == "system":
        system = _SYSTEM_CUES.get(kind, "")
        if system and Path(system).is_file():
            return system
        # A missing system sound must not mean silence. The cue is the only signal that the
        # microphone is live, so the generated fallback answers for it.
        fallback = _cue_file(kind, FALLBACK_PRESET)
        return str(fallback) if fallback else kind
    preset = setting.lower() if setting.lower() in _CUE_PRESETS else FALLBACK_PRESET
    generated = _cue_file(kind, preset)
    return str(generated) if generated else _SYSTEM_CUES.get(kind, kind)


def cue_log_path() -> Path:
    """``~/.murmurflow/audio/cues.jsonl`` — every sound this runtime played, with the file it played."""
    return _scratch_dir() / "cues.jsonl"


def _note_cue(kind: str, sound: str) -> None:
    """Record that a cue was played. Best-effort, and the point is ATTRIBUTION.

    "Random beeps and boops, and not even the ones I selected" has now been diagnosed wrong twice —
    once as our own code, once as macOS's dictation — because a sound leaves no trace and every
    theory sounds equally plausible afterwards. A timestamped line per cue settles it in one look:
    if the log says nothing was played at that second, the sound was not ours, and the hunt moves
    to whatever else is running on the machine.
    """
    try:
        path = cue_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cue": kind, "file": Path(sound).name}
        data = (json.dumps(row) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        if path.stat().st_size > _CUE_LOG_MAX_BYTES:
            tail = path.read_text("utf-8").splitlines()[-200:]
            path.write_text("\n".join(tail) + "\n", "utf-8")
    except Exception:  # noqa: BLE001, S110 — a log about a sound must never cost the sound
        pass


def recent_cues(limit: int = 20) -> list[dict[str, str]]:
    """The last cues this runtime played, oldest first. ``[]`` when it has played none."""
    try:
        lines = cue_log_path().read_text("utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, str]] = []
    for line in lines[-max(1, limit) :]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append({str(k): str(v) for k, v in row.items()})
    return rows


def cues_muted() -> bool:
    """True when ``cue`` is off — no cue sound is made at all.

    An escape hatch that had to exist: a day was lost chasing a beep that could not be placed, and
    the only acceptable answer was all of them gone. The cost is real and yours to accept — the cues
    are the only signal that the microphone is live, since you are looking at the app, not at
    us. `murmurflow config set cue glass` brings them back.
    """
    return str(_cfg().get("cue", "") or "").strip().lower() in CUE_OFF


def cue(sound: str) -> None:
    """Play a short cue by name (``CUE_READY``/``CUE_DONE``/``CUE_FAIL``), non-blocking.

    Silent no-op if audio is unavailable. An existing file path is played as given, so a caller
    that already has a sound file keeps working. Every play is recorded (:func:`_note_cue`) so a
    sound heard can be attributed instead of theorised about.
    """
    if os.environ.get("MURMURFLOW_NO_AUDIO") or cues_muted():
        return
    kind = sound
    sound = sound if Path(sound).is_file() else _cue_path(sound)
    if not Path(sound).is_file():
        return
    platforms.play(sound)
    _note_cue(kind, sound)


def apple_dictation_conflict(trigger: str = "") -> bool:
    """True if the OS's OWN dictation shortcut would fire on the same gesture as ours.

    Apple puts dictation on a double-tap of Control by default. If that is still enabled and the
    user's trigger is a bare Control, both fire: Apple's microphone panel appears on top of this
    one and neither transcript is what was wanted. It is the single most confusing collision this
    tool has, and it was documented in prose nobody reads — so it is a checked row instead.

    Windows puts its own dictation on Win+H, a CHORD, which this tool cannot bind — so there is
    nothing to collide with and the platform answers False. The row still prints.
    """
    return platforms.dictation_conflict(trigger_key(trigger))


def trigger_key(trigger: str = "") -> str:
    """The key this machine talks on: an explicit ``--trigger``, then ``trigger``, then the default.

    One seam, so no caller can forget to read the config and leave the daemon listening on a key
    the user has already rebound away from.
    """
    from . import hotkey

    chosen = trigger or str(_cfg().get("trigger", "") or "")
    if chosen:
        return hotkey.canonical_trigger(chosen)
    # No explicit choice: THE GESTURE PICKS THE KEY, because the two gestures do not have the same
    # problem. A hold starts on the same key-down a shortcut does, so on a bare modifier it fires
    # for ⌃C — it needs a combo. A double-tap does not: two deliberate taps inside half a second is
    # not a shape any shortcut has, and the one that could imitate it is thrown out by the chord
    # guard. So a tap gets ONE ordinary key — easier to perform, and the same key on every OS.
    # Double-tapping two modifiers at once was the price of having only one default, and it is not
    # a price worth paying for safety the gesture already had.
    return hotkey.DEFAULT_TAP_TRIGGER if double_tap_mode() else hotkey.DEFAULT_TRIGGER


def double_tap_mode() -> bool:
    """True when ``doubleTap`` is set: tap the trigger twice to start, twice again to stop.

    **This is the DEFAULT, and holding is the opt-out.** It shipped the other way round and the
    other way round is wrong for a first run: a Mac reports some keys unreliably while they are
    held, so the very first thing a new user experiences is a clip that records and then reports
    `too short` — which reads as "I can start it but I cannot stop it", i.e. broken. A tap has no
    such failure: the key is down for a normal keypress and the OS reports that reliably. It is
    also the only gesture that survives a long sentence, and dictating a long sentence is the
    thing this tool is for.

    The trigger follows the gesture (see :func:`trigger_key`), so flipping this also moves the
    default key onto ONE ordinary key that is the same on every OS, instead of a two-modifier
    combo a hold is forced to use.
    """
    return config.flag("doubleTap", True, cfg=_cfg())


def start_and_cue() -> Recording | None:
    """Begin recording and cue the user once the microphone is genuinely live.

    The cue runs on a THREAD, never inline. :func:`ready` blocks for the ~300ms CoreAudio needs, and
    doing that on the poll loop's own thread means the loop is deaf while it waits — it cannot see a
    chord (so the abort fires late, after the sound has already played) and it cannot see a fast
    release. Off-thread the loop keeps polling, and the cue simply checks the clip is still alive
    before making a noise.
    """
    rec = start()
    if rec is None:
        return None

    def _cue_when_live() -> None:
        # Generous timeout. The device normally opens in ~300ms everywhere — the "launchd takes
        # SECONDS" reading this cap was widened for turned out to be launchd THROTTLING the agent
        # (fixed by `ProcessType: Interactive`, see `platforms.macos.render_plist`), not a slow CoreAudio.
        # The wide cap stays as cheap insurance on unknown client hardware: this is a daemon thread
        # with the recording already running, so waiting longer costs nothing and giving up early
        # costs the user the cue, and with it the sentence.
        if ready(rec, timeout=8.0) and current() is not None:
            cue(CUE_READY)

    threading.Thread(target=_cue_when_live, daemon=True).start()
    return rec


def bind_trigger(
    on_press: object,
    on_release: object,
    *,
    trigger: str = "",
    on_abort: object | None = None,
    should_stop: object | None = None,
    on_tap: object | None = None,
) -> str:
    """Run the key listener in whichever mode ``doubleTap`` selects. Blocks. Returns a description.

    One seam, so dictation and a huddle can never end up on different keys or different gestures —
    which is exactly what happened while each surface bound the keyboard for itself.
    """
    from . import hotkey

    key = trigger_key(trigger)
    if double_tap_mode():
        hotkey.listen_double_tap(
            on_press,  # type: ignore[arg-type]
            on_release,  # type: ignore[arg-type]
            trigger=key,
            should_stop=should_stop if callable(should_stop) else None,
            on_tap=on_tap if callable(on_tap) else None,
        )
        return f"double-tap {key} to start, tap once to stop"
    hotkey.listen(
        on_press,  # type: ignore[arg-type]
        on_release,  # type: ignore[arg-type]
        trigger=key,
        on_abort=on_abort if callable(on_abort) else None,
        should_stop=should_stop if callable(should_stop) else None,
    )
    return f"hold {key}"


#: How each trigger name reads to a person. A combo is two keys held TOGETHER and the name has to
#: say so, or ``control_option`` reads as a choice between them.
TRIGGER_LABELS = {
    "control_option": "Control+Option (⌃⌥) together",
    "control_command": "Control+Command (⌃⌘) together",
    "command_option": "Command+Option (⌘⌥) together",
    "control_shift": "Control+Shift (⌃⇧) together",
}


def trigger_label(trigger: str = "") -> str:
    """The trigger as a person would say it out loud."""
    key = trigger_key(trigger)
    return TRIGGER_LABELS.get(key, key)


def trigger_hint(trigger: str = "") -> str:
    """How to work the trigger right now, for a banner printed BEFORE the listener blocks."""
    key = trigger_label(trigger)
    return f"double-tap {key} to start, tap once to stop" if double_tap_mode() else f"hold {key}"


# --- lending the key ---------------------------------------------------------------------------
#
# The listener LOCK above answers "may I start a second listener", and its answer is no. That is a
# REFUSAL, and a refusal is the wrong shape for the thing people actually want, which is to borrow
# the key for a moment: a voice assistant that wants the same double-tap for a conversation, a
# screen recorder that must not have a microphone taken out from under it, a meeting.
#
# So there is a second, separate idea: a PAUSE. While one is held, the listener sees the trigger and
# does not record. It is not a stop — the daemon stays up, the whisper server stays warm, and the
# key comes back on its own.
#
# EVERY PAUSE EXPIRES, and that is the whole design rather than a safety net. A borrower that
# crashes while holding the key would otherwise leave dictation silently dead with nothing on
# screen to explain it, which is the worst failure this tool can have: the key is pressed, the cue
# does not play, and there is nothing to read. A deadline means the worst case is a few minutes of
# no dictation that fixes itself, and `murmurflow doctor` names the holder the whole time.

#: How long a pause lasts when the borrower does not say. Long enough for a conversation, short
#: enough that a crashed borrower is an annoyance rather than an outage.
DEFAULT_PAUSE_SECONDS = 300.0

#: And the ceiling on one, however long the borrower asks for. An hour of silently no dictation is
#: already past the point where somebody would file a bug instead of waiting.
MAX_PAUSE_SECONDS = 3600.0


def pause_path() -> Path:
    return config.home_root() / "paused"


def pause(seconds: float = DEFAULT_PAUSE_SECONDS, *, who: str = "") -> float:
    """Stand the listener down until a deadline, and return that deadline as a unix time.

    ``who`` is whoever is borrowing the key, in words a person would recognise ("a Zyx huddle").
    It costs nothing to pass and it is the entire difference between ``murmurflow doctor`` saying
    "paused" and it saying "paused by a Zyx huddle for another 4 minutes" — one of those is a
    diagnosis and the other is a new question.
    """
    until = time.time() + min(max(1.0, float(seconds)), MAX_PAUSE_SECONDS)
    with contextlib.suppress(OSError):
        pause_path().write_text(
            json.dumps({"until": until, "pid": os.getpid(), "who": who.strip()}), "utf-8"
        )
    return until


def resume() -> bool:
    """Give the key back. ``True`` if it had been borrowed. Idempotent, and safe from anyone."""
    existed = pause_path().is_file()
    with contextlib.suppress(OSError):
        pause_path().unlink(missing_ok=True)
    return existed


def paused() -> tuple[bool, str]:
    """``(is the key lent out, who has it and for how long)``. Never raises.

    An expired pause is NOT paused, and the marker is cleaned up on the way past — a borrower that
    died holding the key must not need anybody to know that a file exists. Unreadable or malformed
    is also not paused, for the same reason: every failure here resolves to "the key is yours",
    because the failure mode on the other side is a microphone that never opens again.
    """
    try:
        data = json.loads(pause_path().read_text("utf-8"))
        until = float(data.get("until", 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return False, ""
    remaining = until - time.time()
    if remaining <= 0:
        resume()
        return False, ""
    who = str(data.get("who", "") or "another program").strip()
    return True, f"{who}, for another {int(remaining) // 60}m {int(remaining) % 60}s"


def listener_lock_path() -> Path:
    return config.home_root() / "listener.pid"


def listener_pid() -> int:
    """The PID of a live listener OTHER than this process, or 0 when the trigger is free.

    A lock left behind by a crash or a hard reboot reads as free, because the process it names is
    gone. Nobody is ever told to delete a lock file.
    """
    try:
        pid = int(listener_lock_path().read_text("utf-8").strip())
    except (OSError, ValueError):
        return 0
    if pid <= 0 or pid == os.getpid() or _exited(pid):
        return 0
    return pid


def _pgrep(pattern: str) -> list[int]:
    """Live PIDs whose full command line contains ``pattern``, this process excluded.

    ponytail: ``pgrep`` is POSIX, so on Windows this answers ``[]`` for "none" and ``[]`` for "I
    cannot look" alike. The ceiling is that three things degrade quietly there — orphan reaping
    (bounded anyway by :data:`MAX_CLIP_SECONDS`), stopping the warm server on uninstall, and the
    doctor's listener count, which reads as "dictation is off" on a machine where it is on. The
    upgrade is one seam function over ``tasklist``, and it is worth writing the first time somebody
    actually runs this on Windows — not blind from a Mac, where it cannot be tested even once.
    """
    try:
        found = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for token in found.stdout.split():
        with contextlib.suppress(ValueError):
            pids.append(int(token))
    return [pid for pid in pids if pid != os.getpid()]


def listener_pids() -> list[int]:
    """Every murmurflow listener process on this Mac, whether or not it took the lock.

    ``listener_pid`` answers "may I start", which is the lock's question. This answers "how many
    are there", which is the user's — and it is deliberately not read from the lock file, because
    the setup worth reporting is the broken one where something is listening that never claimed it.
    """
    return _pgrep("murmurflow listen")


#: Dictation daemons that are NOT murmurflow and fire on the SAME double-tap key. murmurflow was
#: extracted from zyx's ``core.dictate``, so a machine that runs both hears every cue twice — in
#: two different presets, which is exactly what the report sounds like ("several different start
#: and fail sounds at once"). The listener lock cannot see this: it is one program's lock file and
#: the other program never asks for it, so the only honest place to catch it is the health report.
#: (pgrep pattern, what it is, how to switch it off.)
RIVAL_LISTENERS: tuple[tuple[str, str, str], ...] = (
    ("zyx voice listen", "zyx", "zyx voice uninstall"),
    ("anton voice listen", "anton", "anton voice uninstall"),
)


def rival_listeners() -> list[tuple[str, list[int], str]]:
    """(name, pids, how to stop it) for every non-murmurflow listener on the same trigger."""
    rivals = []
    for pattern, name, fix in RIVAL_LISTENERS:
        pids = _pgrep(pattern)
        if pids:
            rivals.append((name, pids, fix))
    return rivals


def claim_listener() -> int:
    """Take the trigger for this process. Returns 0 when it is ours, else the PID that holds it.

    Two listeners on one key is not a half-working setup, it is a doubled one: two chimes, two
    recorders on the same microphone, and the sentence pasted twice. It arrives the ordinary way —
    the login agent is running and you start ``murmurflow listen`` in a terminal to watch it, or a
    second install lands its own agent — so the second one stands down and says why.

    ``O_EXCL`` rather than read-then-write, because two agents CAN start in the same instant at
    login and a check that is not atomic is exactly the race that produces the doubled sound it
    was added to prevent.
    """
    path = listener_lock_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            holder = listener_pid()
            if holder:
                return holder
            with contextlib.suppress(OSError):
                path.unlink()  # stale: whoever wrote it is gone
            continue
        except OSError:
            return 0  # a home we cannot write to is no reason to refuse to work
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        return 0
    return listener_pid()


# How many clips in a row may fall through to the cold path before the warm server is presumed
# WEDGED and bounced. Two, not one: a single cold clip is what a server that is still loading its
# model looks like, and bouncing it for that would restart the load it was in the middle of.
#
# The failure this exists for is not a server that DIED — that one is visible, because `server_up`
# says no. It is a server that keeps answering and answers wrongly. Live (2026-08-17): whisper-
# server replied "FFmpeg conversion failed" to every single request, so every clip silently took
# the cold path, where the confidence gate reports 1.0 and can judge nothing — and a bump on the
# desk came back as two sentences of Japanese and was typed into a terminal. Nothing on screen
# said the fast path was gone; the daemon simply got slower and less safe for the rest of the day.
COLD_CLIPS_BEFORE_RESTART = 2


def listen_loop(
    *,
    trigger: str = "",
    on_event: object | None = None,
    should_stop: object | None = None,
) -> None:
    """Run the press-to-talk daemon: hold the trigger, speak, release, text appears. Blocks.

    Spawns the warm whisper-server first so the FIRST dictation is as fast as the hundredth — the
    ~1s model load is paid once here, at daemon start, instead of on the first real sentence.

    ``on_event`` (optional callable taking a short string) receives progress lines so the CLI can
    print them; the daemon itself never prints, so it is usable from launchd.

    It never prints anything itself, which is what makes it usable from launchd.
    """

    def emit(line: str) -> None:
        if callable(on_event):
            # A noisy printer must never kill the daemon.
            with contextlib.suppress(Exception):
                on_event(line)

    ready_ok, hint = available()
    if not ready_ok:
        emit(hint)
        return
    holder = claim_listener()
    if holder:
        emit(
            f"[!] another murmurflow listener already has {trigger_key(trigger)} "
            f"(pid {holder}) — this one stands down rather than double every sentence. "
            "Stop the other one first: murmurflow uninstall"
        )
        return
    reaped = reap_orphans()
    if reaped:
        emit(f"[!] stopped {reaped} orphaned recorder(s) left by a previous run")
    # HERE as well as after each clip, and the restart loop is the reason: a listener that cannot
    # start writes its reason, exits, and is started again ten seconds later, all night. Nobody
    # dictates in that state, so a per-clip trim would never run on the one machine that needs it.
    trim_log()
    warm_expected = start_server()
    if warm_expected:
        emit(f"whisper warm on :{port()}")
    else:
        emit("whisper-server unavailable — falling back to cold whisper-cli (~1s slower)")

    #: Consecutive clips that took the cold path while a warm server was supposed to be answering.
    cold_streak = [0]

    def watch_warm(warm: bool | None) -> None:
        """Bounce the warm server once it has stopped answering. See COLD_CLIPS_BEFORE_RESTART.

        On a THREAD, because the restart waits for a 1.6 GB model to load and the poll loop that
        calls this is the only thing watching the trigger key — a listener deaf for ten seconds
        after every dictation is a worse bug than the one being healed. The clip that noticed is
        already pasted, so nothing is waiting on this.
        """
        if warm is None or not warm_expected:
            return  # nothing was decoded, or there was never a warm server to lose
        if warm:
            cold_streak[0] = 0
            return
        cold_streak[0] += 1
        if cold_streak[0] < COLD_CLIPS_BEFORE_RESTART:
            return
        cold_streak[0] = 0

        def bounce() -> None:
            stop_server()
            emit("[!] the warm whisper-server stopped answering — restarting it")
            emit(
                f"whisper warm again on :{port()}"
                if start_server(wait=30.0)
                else "[!] the warm whisper-server would not come back — staying on cold whisper-cli"
            )

        threading.Thread(target=bounce, daemon=True).start()

    # WHICH recording is ours. `current()` is a machine-wide marker — it is how `murmurflow toggle`
    # and a stale clip from a crashed run are both visible — so the daemon finishes only the
    # Recording it started itself rather than whatever happens to be in flight.
    mine: list[Recording] = []

    def on_press() -> None:
        # THE LENT KEY IS CHECKED HERE AND NOT IN `start_and_cue`, and the difference is the whole
        # rule: a pause stands the DAEMON down, it does not disable recording. Somebody who types
        # `murmurflow toggle` while the key is lent is asking for a recording on purpose and gets
        # one; only the trigger stands down.
        #
        # No cue, deliberately. The person pressing it knows they lent the key out — they are
        # talking to whatever borrowed it — and a sound here would be this program interrupting the
        # conversation it stood down for. The log says so once per press, because a daemon that
        # does nothing still has to be explainable after the fact.
        lent, holder = paused()
        if lent:
            emit(f"[--] the trigger is lent to {holder} — not recording")
            return
        rec = start_and_cue()
        if rec is not None:
            mine.append(rec)

    def on_release() -> None:
        if not mine:
            return  # nothing of OURS was recording
        result = finish(mine.pop())
        watch_warm(result.warm)
        if result.problem:
            # A clip too short to hold a word is not a FAILURE, it is a gesture that was never a
            # sentence — a brushed key, or a hand that changed its mind. Chiming at it is the
            # "beeping that comes back out of nowhere" that makes a background daemon feel broken:
            # nothing was asked for, so a noise reporting that it did not work is pure confusion.
            # It is still logged. Every OTHER problem (a silent microphone, a failed transcription)
            # followed a real attempt and still gets its tone.
            if not result.problem.startswith((TOO_SHORT, NOTHING_SAID)):
                cue(CUE_FAIL)
            emit(f"[!] {result.problem}")
            return
        # No cue here: `finish` already played it when the microphone closed, and the text landing
        # at the cursor is a better confirmation than any second tone.
        emit(log_line(result))
        # One stat per dictation, off the felt path: the text is already at the cursor by now.
        trim_log()

    def on_abort() -> None:
        """A keyboard shortcut, not speech: throw the audio away without transcribing it."""
        if not mine:
            return  # same ownership rule as on_release
        wav = stop(mine.pop())
        if wav is not None:
            wav.unlink(missing_ok=True)

    _, device = resolve_input()
    emit(f"listening — {trigger_hint(trigger)} (mic: {device})")
    bind_trigger(on_press, on_release, trigger=trigger, on_abort=on_abort, should_stop=should_stop)
