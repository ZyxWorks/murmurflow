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
no Xcode, no code signing and no notarization. No always-on microphone: it decodes the clip only
once you have stopped talking, so nothing is listening between sentences. No LLM rewrite on the hot path
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

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from . import config, platforms, speech, whisper

# ONE COPY, TWO TOOLS. Everything below is the same file in zyx's `core.speech` - byte for byte,
# checked by a digest in both repos and copied by one command (`make voice-sync`, run from zyx).
# It is the layer that is true of AUDIO and of a TRANSCRIPT rather than of either product: this
# module was extracted from zyx's `core.dictate`, the two drifted for three weeks, and the same two
# bugs had to be found twice. Re-exported rather than reached through the module, so every call
# site and every test that says `dictate.X` keeps saying it.
from .speech import (  # noqa: F401
    AUTO_STOP_SECONDS,
    BYTES_PER_SECOND,
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    MIN_HEADER_BYTES,
    NO_SPEECH,
    QUIET_DBFS,
    SAMPLE_RATE,
    SILENCE_STOP_SECONDS,
    SILENT_DBFS,
    SPEECH_CONFIDENCE,
    TOO_SHORT,
    TRIM_BLOCK_SECONDS,
    TRIM_KEEP_SECONDS,
    Heard,
    Recording,
    Setup,
    audio_seconds,
    data_offset,
    is_hallucination,
    join_segments,
    language_code,
    multipart,
    peak_dbfs,
    repair_punctuation,
    repair_wav,
    resolve_bin,
    strip_trailing_hallucination,
    tail_dbfs,
)

# Homebrew's bin dirs. launchd hands an agent a minimal PATH that excludes them, so a bare
# shutil.which() finds nothing when the listener runs from a plist while working fine in a shell
# (the TUNNEL-PATH-1 lesson, generalized here rather than re-learned).

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


#: The highest ``port`` that leaves the one above it free. Nothing of ours listens there any
#: more, but :func:`stop_server` still sweeps it to reap the small live server an older MurmurFlow
#: ran, and a port set to 65535 would make that sweep ask about 65536, which is not a port.
MAX_PORT = 65534


def port() -> int:
    """The loopback port for the warm whisper-server (``port``)."""
    try:
        chosen = int(str(_cfg().get("port", "") or DEFAULT_PORT))
    except (TypeError, ValueError):
        return DEFAULT_PORT
    return chosen if 1 <= chosen <= MAX_PORT else DEFAULT_PORT


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


def kept_audio() -> Path:
    """The one clip ``keepAudio`` holds on to. Named here so the switch-off can find it again.

    ``reap_orphans`` only globs ``dictate-*.wav``, so nothing else in this module will ever delete
    it — turning the setting off is the only thing that can, and that is the promise the README
    makes.
    """
    return _scratch_dir() / "last.wav"


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
    devices = list_inputs()
    resolved = speech.pick_input(input_name(), devices, default=platforms.default_input())
    if devices:  # never cache a failed enumeration — ffmpeg may simply not have been ready
        _INPUT_CACHE, _INPUT_CACHED_AT = resolved, time.monotonic()
    return resolved


# --- capture ----------------------------------------------------------------------------------


#: What a wav header is when nothing else is in it: `RIFF....WAVE` + a 16-byte `fmt ` chunk.
#: A FLOOR, never the answer — see :func:`data_offset`.


# --- warm transcription -----------------------------------------------------------------------


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
            body, content_type = multipart(probe, {"response_format": "json", "temperature": "0"})
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


# --- the recorder, in this install's vocabulary --------------------------------------------------
#
# The engine below is byte-identical with zyx's copy and knows nothing about avfoundation or dshow:
# `capture` is this platform's own input arguments with the device already resolved, which is
# exactly the seam that made the Windows port four files instead of a fork.


def current() -> Recording | None:
    """The in-flight recording, or ``None``. Stale markers (dead pid) are cleaned up and ignored."""
    return speech.current(state_path())


def start() -> Recording | None:
    """Capture the mic to a fresh 16kHz mono wav; ``None`` if already recording or unable."""
    device, _name = resolve_input()
    return speech.start(
        ffmpeg=resolve_bin("ffmpeg"),
        capture=platforms.capture_args(device),
        scratch=_scratch_dir(),
        state=state_path(),
    )


def ready(rec: Recording, *, timeout: float = 2.0) -> bool:
    """Block until the wav has actual audio frames in it (or ``timeout``). ``True`` if it does."""
    return speech.ready(rec, timeout=timeout)


def stop(rec: Recording | None = None) -> Path | None:
    """Stop the capture and return the finished wav (``None`` if nothing was recording)."""
    return speech.stop(rec, state=state_path())


def reap_orphans() -> int:
    """Stop every recorder left behind by a previous run, and delete its audio. Returns how many."""
    return speech.reap_orphans(scratch=_scratch_dir(), state=state_path())


def _setup(model: str = "") -> speech.Setup:
    """This install's answer to every question the shared engine asks. See :class:`speech.Setup`.

    ONE place where MurmurFlow's vocabulary meets the engine's, so nothing below reads a setting and
    the engine can stay one file - byte-identical with zyx's copy of it.
    """
    return speech.Setup(
        whisper_server=resolve_bin("whisper-server"),
        model=model or whisper.model(),
        threads=whisper.threads(),
        port=port(),
        scratch=_scratch_dir(),
        vocabulary=whisper.vocabulary(),
        language=whisper.language(),
    )


def server_url() -> str:
    return speech.server_url(port())


def server_up() -> bool:
    """True if anything answers on the loopback port - see :func:`ours` for WHO."""
    return speech.server_up(port())


def ours() -> bool:
    """Is a whisper-server what holds our port. See :func:`speech.ours`."""
    return speech.ours(port())


def forget_ownership() -> None:
    """Drop the cached ownership verdict (a server was just started or stopped)."""
    speech.forget_ownership()


def serve_command(model: str = "") -> list[str] | None:
    """The argv that starts the warm whisper-server, or ``None`` if it cannot be built."""
    return speech.serve_command(_setup(model))


def start_server(*, wait: float = 60.0) -> bool:
    """Spawn the warm whisper-server if it is not already up; block until it answers."""
    return speech.start_server(_setup(), wait=wait)


def stop_server() -> int:
    """Stop the warm whisper-server this install started. Returns how many were stopped."""
    return speech.stop_server(port())


def transcribe_warm(wav: Path, *, timeout: float = 60.0, language: str = "") -> Heard:
    """Transcribe via the warm server; empty text if it is not up or errors. Never raises."""
    return speech.transcribe_warm(wav, _setup(), timeout=timeout, language=language)


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

# SOUNDS ONLY, never words. This list used to hold `hey`, `so`, `well`, `right`, `alright`,
# `you know`, `i mean` and `basically` as well, and every one of them is also an ordinary English
# word: "Do you know what time it is?" came back as "Do what time it is?", "I mean what I said." as
# "what I said.", and a leading "hey" was deleted from a real sentence in front of a real user —
# the incident `tidy` documents below. No syntactic rule separates the two uses. Comma-bounding
# does not: that "hey" HAD a comma.
#
# What is left is the non-lexical sounds, which are never a word in any sentence and so cannot eat
# one. A missed filler costs the reader one keystroke; a deleted word is invisible until they
# re-read their own sentence. Anything smarter than this needs a language model, which is
# `polishCommand` and is the user's choice, not a default.
#
# One regex, not two: a leading filler is just a filler at the start of the line, and `\b` with
# `\s*` already handles both positions.
#
# ponytail: `um` is also an ordinary GERMAN word ("um 5 Uhr"), and this list cannot see the clip's
# language. Germans say "äh"/"ähm" rather than "um", so the entry earns little here and risks the
# same word-eating in the other direction; gate the list on the clip language if that ever lands.
_FILLER_WORD = re.compile(r"\b(?:um+|uh+|erm+|hmm+)\b[,.]?\s*", re.IGNORECASE)


def strip_fillers(transcript: str) -> str:
    """Remove spoken filler SOUNDS from a transcript. Removes only — never rewrites or re-cases.

    Sounds, not words: see :data:`_FILLER_WORD` for why "hey", "so" and "you know" are not on the
    list and cannot be. Falls back to the original when the strip would eat the whole line:
    "um, uh" must not become "". A filler-free transcript passes through byte-identical.
    """
    original = transcript.strip()
    text = _FILLER_WORD.sub("", original)
    text = " ".join(text.split())
    return text or original


# A clip this short cannot contain a word — it is a brushed key or an aborted chord. Transcribing
# it wastes a second and, worse, invites the hallucination below.
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
def trim_trailing_quiet(wav: Path) -> bool:
    """Cut the silence off the end of a clip, at this install's own floor. See `speech`."""
    return speech.trim_trailing_quiet(wav, quiet_floor())


def tidy(transcript: str) -> str:
    """Deterministic cleanup of a raw transcript. VERBATIM unless ``stripFillers`` is turned on.

    Costs ~0 ms, which is why it — and not an LLM — runs on the hot path. The judgement calls an
    LLM would make (resolving "no wait, scratch that", reflowing into bullets) live in
    :func:`polish`, off by default.

    **Filler stripping is OFF by default, and that is a correction, not a preference.** It shipped
    on, and it deleted a leading "hey" — exactly as designed, and still wrong. (The design has since
    been corrected too: "hey" is no longer on the list at all. See :data:`_FILLER_WORD`.) Somebody dictating
    "hey, it seems like our murmurflow is still not working" watched the first word of their own
    sentence vanish, with no way to know why or that a setting existed. The contract of a dictation
    tool is that what you said is what appears. A tool that silently edits you is not a faster
    keyboard, it is an unpredictable one. An "um" left in is visible and deletable in one keystroke;
    a word removed is invisible, and you only find it by re-reading your own sentence.
    """
    stripping = config.flag("stripFillers", False, cfg=_cfg())
    transcript = join_segments(transcript)
    text = strip_fillers(transcript) if stripping else transcript.strip()
    # WHISPER SEGMENTS ARE NOT LINE BREAKS, and `join_segments` above is the whole fix for "it puts
    # line breaks in random places". whisper-server returns one `\n` per SEGMENT, and it segments on
    # decoder budget and pauses — never on sentences — so a single dictated sentence comes back as
    # "...than in the\n other, you know,\n" and the paste lands with a hard break mid-clause.
    # This line only collapses what is left (doubled spaces, tabs, a stray non-breaking space).
    # Nobody can speak a newline, so collapsing them is not an edit to what was said: it is undoing
    # a transport artefact, and it belongs OUTSIDE the `stripping` branch. It used to be inside
    # `strip_fillers`, which is exactly how it went missing — defaulting filler-stripping off (the
    # right call) silently took the flattening with it, and the two have nothing to do with each
    # other.
    text = " ".join(text.split())
    # Seam repair is ONLY meaningful after a strip: it exists to close the ", ," a removed filler
    # leaves behind. Run unconditionally it would quietly eat a trailing comma somebody dictated on
    # purpose, which is the same class of bug as the strip itself - so `stripping` decides it, and
    # zyx, which always strips, always closes it.
    return repair_punctuation(
        text,
        close_dangling=stripping,
        ended=transcript.rstrip().endswith((".", "!", "?")),
    )


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
    """Type ``text`` into whatever app has focus. ``(ok, problem, note)``. Never raises.

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
    # Emptiness is tested on the STRIPPED text but the text itself is pasted exactly as given —
    # `tidy` already returns stripped text, so a caller that skips it still gets its own text back.
    text = text or ""
    if not text.strip():
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
def auto_stop_seconds() -> float:
    """How long a forgotten microphone stays open. ``maxHold`` overrides; ``0`` switches it off."""
    raw = _cfg().get("maxHold")
    if isinstance(raw, (int, float)) and float(raw) >= 0:
        return float(raw)
    return AUTO_STOP_SECONDS


def silence_stop_seconds() -> float:
    """How long a silent microphone stays open. ``silenceStop`` overrides; ``0`` switches it off."""
    raw = _cfg().get("silenceStop")
    if isinstance(raw, (int, float)) and float(raw) >= 0:
        return float(raw)
    return SILENCE_STOP_SECONDS


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
    # HERE, the instant the microphone closes — not after the transcribe, and not after the paste.
    # Cued afterwards it arrives a full transcription behind the thing it is acknowledging, so the
    # silence in between reads as the tool having missed you rather than as a wait. Every surface
    # goes through this function, so cueing at the seam is also the only way they stay in step.
    # After the two ways above this can still be a non-event, so nothing chimes at a brushed key.
    # A clip with a waveform open says this by drawing it, so it stays silent.
    if str(rec.wav) not in _LEVELS:
        cue_done()
    # BEFORE the level gates and the transcribe: silence on the end is what whisper invents into,
    # and a clip closed by the silence watchdog ends with fifteen seconds of it. See
    # :func:`trim_trailing_quiet`.
    trim_trailing_quiet(wav)
    level = peak_dbfs(wav)
    captured = audio_seconds(wav)

    def retire() -> None:
        # `keepAudio` keeps the LAST clip (one file, overwritten) so a bad transcription can be
        # listened to instead of guessed about — the difference between "the model is wrong" and
        # "we recorded half a sentence" is audible in two seconds and invisible in a log. Off by
        # default: your voice does not outlive its transcription unless you ask.
        if config.flag("keepAudio", False, cfg=_cfg()):
            with contextlib.suppress(OSError):
                shutil.copyfile(wav, kept_audio())
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


# --- the head start ----------------------------------------------------------------------------
#
# THE FIRST HALF-SECOND OF EVERY SENTENCE WAS MISSING, and whisper was never the reason: CoreAudio
# takes that long to hand ffmpeg its first buffer, so the audio does not exist to transcribe.
# Measured on an M4 Pro — from :func:`start` to the first sample in the file is 0.35-0.65s, and
# after that the capture tracks real time to within 0.3%. A FIXED head loss, not a drift, and one
# no amount of decoding can bring back.
#
# So the recorder is opened one gesture EARLY. The FIRST tap of the double-tap opens it; the second
# tap, a quarter of a second later, then starts a microphone that is already live. If the pair never
# completes the pre-roll is stopped and its file deleted — see :data:`PREROLL_SECONDS`, which is the
# whole privacy cost of this: a stray single tap of the trigger holds the microphone open for that
# long and writes a clip nobody ever reads.

#: How long an unclaimed pre-roll may live. A double-tap is bounded by ``hotkey.DOUBLE_TAP_WINDOW``
#: press-to-press plus ``hotkey.TAP_MAX`` for the second tap's own duration — about a second in the
#: worst case — so anything the second tap could still claim has arrived well inside this.
PREROLL_SECONDS = 1.5

#: How long :func:`preroll_claim` will wait for a pre-roll that is still opening.
#:
#: Nearly always zero: :func:`start` returns as soon as ffmpeg is spawned, in about 7ms, and that
#: happened a quarter of a second ago on the first tap. The exception is the one case worth waiting
#: for — :func:`resolve_input` re-listing the devices when its cache has expired, which shells out
#: to ffmpeg and takes about a second. Waiting is right there: the alternative is starting a SECOND
#: recorder on a microphone that is already opening.
PREROLL_CLAIM_SECONDS = 3.0


@dataclass
class _Preroll:
    """A microphone opening for a gesture that has not finished yet."""

    #: Set once :func:`start` has returned, whatever it returned.
    open: threading.Event
    rec: Recording | None = None


_PREROLL_LOCK = threading.Lock()
_PREROLL: _Preroll | None = None


def preroll() -> None:
    """Open the microphone speculatively, for a gesture that has not finished yet. Returns at once.

    **Every line of the actual work is on a thread, and that is the whole point of this function.**
    It is called from the key poll loop, on a key-DOWN, in the middle of the gesture that loop is
    trying to measure — and the loop has 16ms per tick to notice the key moving. Opening a
    recording is a config read, a liveness probe and a process spawn, and once in a while a
    device re-list that takes a second. Doing that inline made the loop deaf for as long as it
    took, so taps went missing and the daemon could not reliably tell whether it was listening.

    Safe to call on any keypress: it does nothing while something is already recording or already
    pre-rolling, and an unclaimed pre-roll stops itself after :data:`PREROLL_SECONDS`.
    """
    global _PREROLL
    with _PREROLL_LOCK:
        if _PREROLL is not None:
            return
        pending = _Preroll(threading.Event())
        _PREROLL = pending

    def _open() -> None:
        with contextlib.suppress(Exception):
            pending.rec = None if current() is not None else start()
        pending.open.set()

    threading.Thread(target=_open, daemon=True).start()
    # A timer and not a deadline checked by the poll loop: the loop only runs a callback when a KEY
    # moves, and the case to clean up is precisely the one where no key moves again.
    threading.Timer(PREROLL_SECONDS, _preroll_expire, args=(pending,)).start()


def preroll_claim() -> Recording | None:
    """Take the pre-rolled recording, if there is one. It becomes the caller's to finish."""
    global _PREROLL
    with _PREROLL_LOCK:
        pending, _PREROLL = _PREROLL, None
    if pending is None:
        return None
    pending.open.wait(PREROLL_CLAIM_SECONDS)
    return pending.rec


def _preroll_expire(pending: _Preroll) -> None:
    """Stop and delete the pre-roll unless it has been claimed in the meantime. Never raises."""
    global _PREROLL
    with _PREROLL_LOCK:
        if _PREROLL is not pending:
            return
        _PREROLL = None
    # Behind the opener, or the recorder it is in the middle of spawning outlives this and holds
    # the microphone open with nobody left holding a reference to it.
    pending.open.wait(PREROLL_CLAIM_SECONDS)
    with contextlib.suppress(Exception):
        if pending.rec is not None:
            wav = stop(pending.rec)
            if wav is not None:
                wav.unlink(missing_ok=True)


#: The open waveform per clip, keyed by its wav. A clip with one makes no sound: the pill already
#: says the microphone opened, shows it hearing you, and shows it working once it closed.
_LEVELS: dict[str, object] = {}


def show_level(rec: Recording) -> None:
    """Open the waveform for ``rec``, where the platform has one. Never raises, never in the suite."""
    if os.environ.get("MURMURFLOW_NO_AUDIO"):
        return
    with contextlib.suppress(Exception):
        handle = platforms.show_level(rec.wav)
        if handle is not None:
            _LEVELS[str(rec.wav)] = handle


def hide_level(wav: Path) -> None:
    """Close the waveform for ``wav``, if one is open. Never raises."""
    handle = _LEVELS.pop(str(wav), None)
    if handle is not None:
        with contextlib.suppress(Exception):
            platforms.hide_level(handle)


def cue_ready() -> None:
    """The microphone is live, start talking. Never raises.

    The words arriving at the cursor say "it heard you", perfectly — but they arrive a second and
    a half in, which is a second and a half of not knowing whether to start, every single time.
    Fired the moment the device is genuinely live (see :func:`ready`), so it is also the signal
    that stops the first word being spoken into a microphone that is not open yet.
    """
    _sound(platforms.play_ready)


def cue_done() -> None:
    """The microphone just closed, stop talking. Never raises.

    **Not "the text has arrived".** It fires the instant the recorder stops, which is a second or
    more before the transcription lands and the text is pasted — and that gap is precisely why it
    exists: without it, the silence between the tap and the paste reads as the tool having missed
    you, rather than as a wait.

    Two sounds, and no third: a failure is a line in the log, not a noise in a meeting.
    """
    _sound(platforms.play_done)


def _sound(play: object) -> None:
    """Make one noise, or none. Never raises, and never anything the test suite can hear."""
    if os.environ.get("MURMURFLOW_NO_AUDIO") or not callable(play):
        return
    with contextlib.suppress(Exception):
        play()


def bind_trigger(
    on_press: object,
    on_release: object,
    *,
    trigger: str = "",
    on_abort: object | None = None,
    should_stop: object | None = None,
    on_tap: object | None = None,
    is_recording: object | None = None,
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
            is_recording=is_recording if callable(is_recording) else None,
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
    if pid <= 0 or pid == os.getpid() or speech._exited(pid):
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
    #: Held for the instant a clip is claimed. `on_release`, `on_abort` and the forgotten-key
    #: watchdog all end the same recording, and without this a tap landing while the watchdog was
    #: deciding transcribed one clip twice.
    claiming = threading.Lock()

    def claim(rec: Recording | None = None) -> Recording | None:
        """Take the in-flight recording, or ``None`` if something else already took it."""
        with claiming:
            if not mine or (rec is not None and mine[0] is not rec):
                return None
            return mine.pop()

    def _say_ready(rec: Recording) -> None:
        """Sound the one cue, once the microphone is genuinely live and the clip is still alive."""
        if ready(rec, timeout=8.0) and current() is not None and str(rec.wav) not in _LEVELS:
            cue_ready()

    def on_tap(what: str) -> None:
        """Open the microphone on the first tap of the pair — see :func:`preroll`.

        On the PRESS and not the release, because every millisecond here is a millisecond of the
        first word: the device needs ~0.6s and the rest of the gesture only supplies ~0.5s of it.

        A LENT trigger opens nothing. This runs before `on_press` gets to check, so without the
        check here a paused daemon still opened the microphone for up to
        :data:`PREROLL_SECONDS` on every press of a key it had promised not to listen to.
        """
        if what == "press" and not mine and not paused()[0]:
            preroll()

    def on_press() -> None:
        # THE LENT KEY IS CHECKED HERE AND NOT IN `preroll`, and the difference is the whole rule:
        # a pause stands the DAEMON down, it does not disable recording. Somebody who types
        # `murmurflow toggle` while the key is lent is asking for a recording on purpose and gets
        # one; only the trigger stands down. The log says so once per press, because a daemon that
        # does nothing still has to be explainable after the fact.
        lent, holder = paused()
        if lent:
            emit(f"[--] the trigger is lent to {holder} — not recording")
            return
        # The pre-roll FIRST: it is already `current()`, so a plain `start()` would decline.
        rec = preroll_claim() or start()
        if rec is not None:
            mine.append(rec)
            show_level(rec)
            threading.Thread(target=_forgot, args=(rec,), daemon=True).start()
            # ON A THREAD, never inline: `ready` blocks until the device hands over its first
            # buffer, and this runs on the poll loop, which is the only thing watching the key.
            # Usually instant — the pre-roll opened the microphone a quarter of a second ago.
            threading.Thread(target=_say_ready, args=(rec,), daemon=True).start()

    def _forgot(rec: Recording) -> None:
        """Close the microphone for a clip nobody tapped to stop, and type what was said.

        A thread per clip and not a timer in the poll loop: the poll loop is the only thing
        watching the trigger key, so anything that blocks there is a listener that misses a tap.
        """
        limit = auto_stop_seconds()
        quiet = silence_stop_seconds()
        if limit <= 0 and quiet <= 0:
            return
        deadline = time.monotonic() + (limit if limit > 0 else float("inf"))
        why = ""
        while not why:
            if not mine or mine[0] is not rec:
                return  # a tap, or an abort, already ended it
            if time.monotonic() >= deadline:
                why = f"after {limit:.0f}s"
            elif quiet > 0 and tail_dbfs(rec.wav, quiet) < quiet_floor():
                why = f"after {quiet:.0f}s of silence"
            else:
                time.sleep(0.5)
        if claim(rec) is None:
            return
        emit(f"[--] closed the microphone {why} — you did not tap to stop")
        _land(rec)

    def _land(rec: Recording) -> None:
        """Transcribe and type ``rec``, then close its waveform whatever happened."""
        try:
            result = finish(rec)
        finally:
            hide_level(rec.wav)
        watch_warm(result.warm)
        if result.problem:
            emit(f"[!] {result.problem}")
            return
        emit(log_line(result))
        # One stat per dictation, off the felt path: the text is already at the cursor by now.
        trim_log()

    def on_release() -> None:
        rec = claim()
        if rec is None:
            return  # nothing of OURS was recording, or the watchdog got there first
        _land(rec)

    def on_abort() -> None:
        """A keyboard shortcut, not speech: throw the audio away without transcribing it."""
        rec = claim()
        if rec is None:
            return
        hide_level(rec.wav)
        wav = stop(rec)
        if wav is not None:
            wav.unlink(missing_ok=True)

    _, device = resolve_input()
    emit(f"listening — {trigger_hint(trigger)} (mic: {device})")
    bind_trigger(
        on_press,
        on_release,
        trigger=trigger,
        on_abort=on_abort,
        should_stop=should_stop,
        on_tap=on_tap,
        # The microphone can close itself now, so the gesture has to be able to find that out
        # without being told by a tap — see :func:`hotkey.listen_double_tap`.
        is_recording=lambda: bool(mine),
    )
