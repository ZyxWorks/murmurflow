"""speech — the physics of a microphone and the hygiene of a transcript. ONE COPY, TWO TOOLS.

**This file is byte-identical in MurmurFlow and in zyx, and that is enforced.** MurmurFlow was
extracted from zyx's ``core.dictate`` and the two ship separately on purpose — they do different
jobs, one types at your cursor and one hands what you said to a runtime — but the layer underneath
both is not a product decision at all. It is what a wav header looks like, what silence measures,
and what whisper invents when it is handed nothing. That layer drifted for three weeks and cost
two of the same bugs found twice, which is what this file exists to stop.

**What belongs here:** anything that is true of the AUDIO or of the TRANSCRIPT, needs no config,
opens no device, and has no opinion about what happens to the words afterwards.

**What does NOT:** every reader of a setting (the floors are passed in as arguments, never read),
the recorder, the key, the server, the cues, and anything that types, pastes, speaks or answers.
A function here that needs to ask a question about THIS install is in the wrong file.

Stdlib only, and it must stay that way: it is copied verbatim into a runtime whose first invariant
is a stdlib core.

Licence: MIT, as MurmurFlow is. The copy in zyx is vendored under it — see
docs/legal/THIRD-PARTY.md.
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
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# --- the wav on disk ------------------------------------------------------------------------

MIN_HEADER_BYTES = 44


def data_offset(wav: Path) -> int:
    """Where the samples actually start. :data:`MIN_HEADER_BYTES` when the header cannot be read.

    **44 is a guess and on this machine it is the wrong one.** Measured 2026-09-09 with the
    recorder's own argv: ffmpeg writes a 26-byte ``LIST``/``INFO`` chunk between ``fmt `` and
    ``data``, so the first sample is at byte **78**. Everything that seeks past "the header" by a
    constant — the trailing-quiet trim, the tail level read — is then reading 34 bytes of encoder
    metadata as audio and cutting the clip 34 bytes early. (MurmurFlow assumes 44 in both places
    and has the same off-by-a-chunk; it is small enough to have gone unnoticed there, which is
    exactly why it was measured here rather than copied.)

    Works on a file STILL BEING RECORDED: the chunk headers are written when ffmpeg opens the file,
    and only the two SIZE fields are left for the exit that may never come.
    """
    try:
        with wav.open("rb") as handle:
            if handle.read(4) != b"RIFF":
                return MIN_HEADER_BYTES
            handle.seek(8)
            if handle.read(4) != b"WAVE":
                return MIN_HEADER_BYTES
            size = wav.stat().st_size
            offset = 12
            while offset + 8 <= size:
                handle.seek(offset)
                name = handle.read(4)
                declared = int.from_bytes(handle.read(4), "little")
                if name == b"data":
                    return offset + 8
                if declared <= 0:
                    break  # a chunk with no length: the walk cannot go on honestly
                offset += 8 + declared + (declared % 2)  # chunks are padded to an even length
    except (OSError, ValueError):
        pass
    return MIN_HEADER_BYTES


def repair_wav(wav: Path) -> bool:
    """Patch a RIFF header whose lengths were never written. ``True`` if it had to. Never raises.

    **ffmpeg writes the real lengths when it EXITS, and it does not always get to.** The SIGKILL
    fallback in :func:`stop` is one way; :func:`trim_trailing_quiet` is the other, and there it is
    not a failure at all — cutting the tail off a clip leaves a header still claiming the length it
    had before the cut.

    A killed ffmpeg leaves ``0xFFFFFFFF`` in both size fields rather than zero, so the audio still
    DECODES (``-flush_packets 1`` already wrote every sample and readers stop at the end of file).
    What breaks is every reader that believes the header — :func:`peak_dbfs` reads a frame count
    that is not there, and a clip something already went wrong with is exactly the clip whose
    diagnostics go blind.

    Cheaper than what it protects: a stat and twelve bytes on the ordinary path, where the header
    is already right and this returns immediately. (Ported from MurmurFlow 2026-09-09.)
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


# --- what a sample measures ----------------------------------------------------------------

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2


def _peak_dbfs(frames: bytes) -> float:
    """Peak of 16-bit PCM ``frames`` in dBFS. ``-inf`` for silence or nothing."""
    if not frames:
        return float("-inf")
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if not samples:
        return float("-inf")
    # `max(max(...), -min(...))` and not `max(abs(s) for s in samples)`: both find the same peak,
    # but the generator runs one Python-level loop per SAMPLE — a minute of speech is ~1M of them —
    # where two array scans stay in C. This sits between the key coming up and the transcribe
    # starting, which is the one stretch of this a person is waiting on.
    peak = max(max(samples), -min(samples))
    if peak == 0:
        return float("-inf")
    return 20 * math.log10(min(peak, 32768) / 32768.0)


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
    return _peak_dbfs(frames)


#: Silence left on the end of a clip, in blocks this long, is cut before anything transcribes it.
#: Small enough to find the end of the last word closely, large enough that one loud sample of

TRIM_BLOCK_SECONDS = 0.2

#: ...and this much is kept after the last block that had sound in it. A word's decay is part of
#: the word, and whisper reads a hard cut at the end of a syllable as a different syllable.
TRIM_KEEP_SECONDS = 0.4


def trim_trailing_quiet(wav: Path, floor: float) -> bool:
    """Cut silence off the end of a clip. True if anything was cut. Never raises.

    **Whisper invents words when it is handed audio with nothing in it**, and this is the fix for
    it — measured in MurmurFlow, after two that were not. The same 12 seconds of speech, three ways:

        speech alone                 "...but just in this text box,"
        + 20s of digital silence     "...but just in this text box, Thank you."
        + 20s of faint room noise    "...but just in this text box.."

    So the invention is not a property of the speech, the model or the prompt. It is the silence,
    and the cure is not to hand it over. It matters more here than it did there: with
    :data:`SILENCE_STOP_SECONDS` a forgotten microphone now ALWAYS ends in fifteen seconds of
    nothing, so without this every auto-closed turn would arrive with an invented tail on it.

    What was tried first and refused, both measured: no word list can catch it (whisper answers
    silence in a different invented language each time — see :data:`SPEECH_CONFIDENCE`), and
    whisper's own ``no_speech_prob``/``avg_logprob`` do not either; the invented " Thank you." came
    back at 0.000 and -0.28, sitting among real speech at -0.05 to -0.12.

    Nothing is cut when the clip is quiet all the way through — that is a clip with no speech in it,
    and the level gates in :func:`finish` are what should judge it and say so.
    (Ported from MurmurFlow 2026-09-09.)
    """
    block = int(TRIM_BLOCK_SECONDS * BYTES_PER_SECOND)
    head = data_offset(wav)  # NOT 44: this ffmpeg writes a LIST chunk too
    try:
        size = wav.stat().st_size
        with wav.open("rb") as handle:
            handle.seek(head)
            audio = handle.read()
    except OSError:
        return False
    last = -1
    for index in range(len(audio) // block):
        if _peak_dbfs(audio[index * block : (index + 1) * block]) >= floor:
            last = index
    if last < 0:
        return False  # nothing above the floor anywhere: not ours to judge
    keep = head + (last + 1) * block + int(TRIM_KEEP_SECONDS * BYTES_PER_SECOND)
    if keep >= size:
        return False
    try:
        with wav.open("r+b") as handle:
            handle.truncate(keep)
    except OSError:
        return False
    repair_wav(wav)  # the RIFF header still claims the length it had before the cut
    return True


def tail_dbfs(wav: Path, seconds: float) -> float:
    """Peak of the LAST ``seconds`` of a clip that is STILL BEING RECORDED. ``0.0`` = no opinion.

    Read by seeking from the end of the file rather than through :mod:`wave`, because while ffmpeg
    is still appending, the RIFF header holds the lengths it was born with — zero — so every
    header-respecting reader sees an empty file. (:func:`repair_wav` is what undoes that, and it
    must not be run against a file the recorder is still writing.)

    Every byte past the header is a sample, so "the last fifteen seconds" is the last
    ``15 * BYTES_PER_SECOND`` bytes and a seek is the whole implementation. A clip that has not yet
    run that long has no opinion — ``0.0``, which is well above any real floor and so is never read
    as silence. (Ported from MurmurFlow 2026-09-09.)
    """
    want = int(seconds * BYTES_PER_SECOND)
    if want <= 0:
        return 0.0
    try:
        size = wav.stat().st_size
        if size < want + data_offset(wav):
            return 0.0  # not enough audio yet to have been quiet for that long
        with wav.open("rb") as handle:
            handle.seek(size - want)
            return _peak_dbfs(handle.read(want))
    except OSError:
        return 0.0


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


# --- the thresholds, each with the measurement behind it -----------------------------------

MIN_CLIP_SECONDS = 0.4

# The one problem that is NOT worth a sound: see the daemon's release handler.
TOO_SHORT = "too short"

# The clip was long enough and loud enough to be a sentence, but there was no speech in it. ONE
# string for every way we reach that conclusion (whisper's language score, the boilerplate word
# list, an empty transcript) because callers act on it rather than print it: the huddle counts
# consecutive occurrences to decide when to stop trusting the microphone. Kept in the operator's
# own words — he does not care which of the three traps fired.
NO_SPEECH = "I didn't hear anything"

# The longest single clip the recorder will ever produce (see the `-t` flag in `start`). Ten minutes
# is far beyond any real hold — the longest sentence the operator has dictated is ~60s — and short
# enough that an orphaned recorder costs ~20 MB and ten minutes of open microphone instead of hours.
MAX_CLIP_SECONDS = 600

# How long a turn may run before the huddle closes the microphone ITSELF and answers what was said.
# `MAX_CLIP_SECONDS` above is the recorder's own fuse and stays where it is: it bounds an ORPHAN —
# a clip nobody is waiting for — and it throws the audio away. This is the opposite case. The
# operator is right there and simply forgot the second tap, and the right answer is not to discard
# two minutes of his voice but to finish the turn exactly as the tap would have.

SILENT_DBFS = -70.0

# Below this language score, whisper was not listening to speech — see :class:`Heard`. Measured on
# this operator's own model (large-v3-turbo): every real utterance scored >= 0.969, every silent or
# noisy clip <= 0.453. 0.75 sits in the empty middle with ~0.22 of margin on both sides.
#
# This is the trap that the word-list in `_HALLUCINATIONS` structurally cannot be: whisper answers
# silence in a DIFFERENT invented language each time. The operator hit it live on 2026-08-09 —
# he pressed the key, said nothing, and Zyx answered two turns of invented Icelandic ("Ennum, hvað
# er hann?") as though it were a question. No blocklist can grow fast enough to cover that; asking
# whisper how sure it was covers all of it at once.
SPEECH_CONFIDENCE = 0.75


# The level below which a stretch of audio is a ROOM and not a sentence. Measured: a loud silent
# room reads -38 dBFS and the quietest real speech -15, so -30 sits between them. Do NOT lower it
# to -40 — that is inside the room.
#
# **It is not a gate on a clip, and that distinction is the reason zyx may have this number at all**
# (the voice contract, `deliberately_divergent.quiet_floor`). MurmurFlow DROPS a clip
# under its floor, because it types into a document where transcribing a room is worse than losing
# a sentence; zyx hands text to a model that can decline to answer, so it drops nothing. Here the
# floor answers two different questions: "has he stopped talking" (:data:`SILENCE_STOP_SECONDS`)
# and "is this tail worth transcribing" (:func:`trim_trailing_quiet`).
# Each tool names the setting that moves it in its own vocabulary; this is the default.
QUIET_DBFS = -30.0


# Each tool names the setting that moves it in its own vocabulary; 0 switches it off.
AUTO_STOP_SECONDS = 120.0

# How long the microphone stays open with nothing being said before it closes itself and answers.
#
# Fifteen seconds, and the dictation apps that stop at two or three are WRONG HERE: the gesture is
# a tap, not a held key, so nothing is telling the microphone the operator is still there — and he
# thinks mid-sentence. A clip cut at three seconds would end half his turns mid-thought, with the
# rest spoken into a closed microphone. That failure is worse than the one this fixes, because a
# forgotten microphone loses nothing and a truncated sentence loses the sentence.
# Each tool names the setting that moves it in its own vocabulary; 0 switches it off.
SILENCE_STOP_SECONDS = 15.0


# --- the transcript -------------------------------------------------------------------------

_SEGMENT_SEAM = re.compile(r"([^\S\n]*)\n+([^\S\n]*)")


def join_segments(transcript: str) -> str:
    """Flatten whisper's per-segment newlines WITHOUT inventing a space in the middle of a word."""
    return _SEGMENT_SEAM.sub(lambda seam: " " if seam.group(1) or seam.group(2) else "", transcript)


HALLUCINATIONS: frozenset[str] = frozenset(
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
        # The structural guards elsewhere are the real fix; this is the list of exact strings
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
# "(wind blowing)", a run of music notes (U+266A). Observed live on this operator's quiet
# room (it produced "*sad*"). These
# are an open CLASS, not a word list — a blocklist would need a new entry forever — so the whole
# class is matched structurally: a transcript that is ENTIRELY one bracketed/asterisked annotation
# was a description of a sound, not something the operator said, and must never be typed.
_ANNOTATION_ONLY = re.compile(r"^[\s\u266a]*[\*\[\(]([^\]\)\*]*)[\*\]\)][\s\u266a.]*$")

# An annotation is a LABEL ("sad", "wind blowing", "MUSIC"), never a sentence. Without this cap the
# trap also eats a real dictated line that happens to be fully parenthesised — "(That said, ship it
# anyway.)" — which is the worse bug of the two: a hallucination that slips through is visible and
# deletable, whereas silently swallowing what the operator actually said looks like the mic failed.
# Bias deliberately toward letting text through.
_MAX_ANNOTATION_WORDS = 3


def is_hallucination(text: str, extra: frozenset[str] = frozenset()) -> bool:
    """True if ``text`` is whisper's output for silence rather than something that was said.

    **``extra`` is the one thing here that is NOT shared, and it must not become shared.**
    :data:`HALLUCINATIONS` holds only what a PERSON NEVER SAYS - subtitle credits, `[BLANK_AUDIO]`,
    the credit line in each language whisper invents it in. The polite one-word sentences
    ("thank you", "so", "bye") are a different bet in each tool: MurmurFlow types into a document,
    where swallowing a sentence somebody did say reads as broken hardware, so it carries none of
    them; zyx hands text to a model that can decline to answer, so it carries them all. That is
    `deliberately_divergent.hallucination_list` in the voice contract, and a test in MurmurFlow
    fails the moment one of those words reaches this table.

    Three shapes: the fixed boilerplate lines it emits for pure silence, a transcript of nothing
    but MUSIC NOTES, and the open class of non-speech ANNOTATIONS it emits for room noise. All
    three must be trapped — any of them typed into the operator's document is a word he did not say.
    """
    stripped = text.strip().lower().strip("\u266a ")
    # A bare run of music notes (U+266A) with no brackets around it. The annotation pattern
    # below cannot see it because that
    # one requires a bracket or an asterisk, and the boilerplate set above cannot either because
    # stripping the notes leaves "", which is not a member. So it fell through both traps and was
    # PASTED. Found 2026-08-14 while extracting this module into a standalone tool; whisper emits
    # bare note runs for music and for room tone, so this was live.
    if not stripped:
        return bool(text.strip())
    if stripped in HALLUCINATIONS or stripped in extra:
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

# Sentence end, for the trailing-boilerplate trap below. Deliberately crude: it only has to find
# the seam between "...make it public." and an appended "Thanks for watching!".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Polite closings whisper invents for the silence at the END of a clip, in both languages the
# operator speaks. These may only ever be matched as a TRAILING sentence with a real one in front
# of them — never on a whole transcript, because every one of them is also a complete thing a
# person says on purpose, and swallowing it looks like broken hardware. That is why they live here
# and not in :data:`_HALLUCINATIONS`. (Ported from MurmurFlow 2026-09-09.)
TRAILING_BOILERPLATE: frozenset[str] = frozenset(
    {
        "thank you",
        "thank you.",
        "thank you!",
        "thank you very much",
        "thank you very much.",
        "thanks",
        "thanks.",
        "thank you for watching",
        "thank you for watching.",
        "thanks for listening",
        "thanks for listening.",
        "bye",
        "bye.",
        "bye!",
        "bye bye",
        "bye-bye.",
        "goodbye",
        "goodbye.",
        "danke",
        "danke.",
        "danke schön",
        "danke schön.",
        "vielen dank",
        "vielen dank.",
        "tschüss",
        "tschüss.",
        "auf wiedersehen",
        "auf wiedersehen.",
        "untertitel im auftrag des zdf",
    }
)


def strip_trailing_hallucination(text: str, extra: frozenset[str] = frozenset()) -> str:
    """Drop whisper's boilerplate when it is APPENDED to a real sentence.

    :func:`is_hallucination` judges the WHOLE line, which is the right shape for a clip that was
    nothing but silence. It is the wrong shape for the other half of the same failure: a real
    sentence followed by trailing silence comes back as the sentence *plus* the credit line
    ("...so only agent flow is public now. Thanks for watching!"). Every gate before this one reads
    the transcript as a whole — the confidence score, the language score and the blocklist all see
    a confident, real, in-language sentence — so the invented tail passes all three.

    Only whole trailing SENTENCES that :func:`is_hallucination` already recognises are removed, and
    never the last one standing — same one-directional bias as everything else here.
    """
    parts = _SENTENCE_END.split(text)
    while len(parts) > 1 and (
        is_hallucination(parts[-1], extra) or parts[-1].strip().lower() in TRAILING_BOILERPLATE
    ):
        parts.pop()
    return " ".join(parts)


def repair_punctuation(
    text: str, *, close_dangling: bool, ended: bool, extra: frozenset[str] = frozenset()
) -> str:
    """The seam repair every transcript wants, in the one order both tools ran it in.

    ``close_dangling`` is the one genuine difference and it belongs to the caller: a filler strip
    can leave the sentence hanging on the comma that preceded it ("...fixed, you know." ->
    "...fixed,"), so a tool that strips fillers must close that seam and a tool that does not must
    NOT — run unconditionally it quietly eats a trailing comma somebody dictated on purpose, which
    is the same class of bug as the strip itself. ``ended`` says whether the raw transcript ended on
    a full stop, so the seam is closed with one rather than with nothing.
    """
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DOUBLED_PUNCT.sub(r"\2", text)
    # The credit line whisper appends to trailing silence. The whole-transcript case is trapped by
    # `is_hallucination`; this is the same invention riding along behind a real sentence.
    text = strip_trailing_hallucination(text, extra)
    if close_dangling:
        text = _DANGLING_TAIL.sub("." if ended else "", text)
    return text.strip()


# --- the engine: one microphone, one server, one transcript --------------------------------------


@dataclass(frozen=True)
class Setup:
    """What the engine needs to know about THIS install, asked ONCE by the caller.

    **This is the seam that lets the engine be one file.** The two tools name their settings
    differently (`voicePort` against `port`, `whisperModel` against `model`), read them from
    different homes, and answer to different vocabularies - so a single `config.get` in here is a
    line that has to differ, and one line that differs is a file that is no longer shared. Each tool
    builds one of these from its own configuration and hands it over.

    Every field has a default that is safe rather than clever: an empty binary path or model makes
    the function that needs it decline, exactly as a missing binary always did.
    """

    #: Absolute path to the `whisper-server` binary, or "" when it is not installed.
    whisper_server: str = ""
    #: Absolute path to the model weights, or "" when none was found.
    model: str = ""
    #: How many decode threads, as the string the CLI wants.
    threads: str = "4"
    #: The loopback port the warm server listens on.
    port: int = 8477
    #: A writable directory. THE SERVER IS STARTED IN IT - see :func:`start_server`.
    scratch: Path = Path(".")
    #: The proper-noun glossary sent with every request; "" sends none.
    vocabulary: str = ""
    #: The language to decode as, or "auto".
    language: str = "auto"


def server_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def server_up(port: int) -> bool:
    """True if a warm whisper-server answers on the loopback port."""
    try:
        with urllib.request.urlopen(f"{server_url(port)}/", timeout=0.5):
            return True
    except (urllib.error.URLError, OSError):
        return False


#: How long an ownership answer is trusted for. See :func:`ours` — the question is "who holds this
#: port", which changes only when a process starts or dies, and asking it costs a `pgrep`.


#: How long an ownership answer is trusted for. See :func:`ours` — the question is "who holds this
#: port", which changes only when a process starts or dies, and asking it costs a `pgrep`.
OWNERSHIP_SECONDS = 30.0

_OWNERSHIP: dict[int, tuple[float, bool]] = {}


def ours(port: int) -> bool:
    """Is the thing listening on our port a whisper-server, rather than whatever got there first.

    **Because the answer decides where recorded audio is sent.** The port is predictable, so any
    local process can bind it first, receive every clip, and answer with text that gets typed at
    the cursor. A socket that accepts a connection proves nothing about who is on the other end.

    So the port has to be held by a `whisper-server` process. Only one process can bind a port, so
    finding one there IS the answer. Cached for :data:`OWNERSHIP_SECONDS` because this sits on the
    partial path, which asks it about once a second, and `pgrep` is a process spawn.
    """
    at = port
    now = time.monotonic()
    cached = _OWNERSHIP.get(at)
    if cached is not None and now - cached[0] < OWNERSHIP_SECONDS:
        return cached[1]
    try:
        found = subprocess.run(
            ["pgrep", "-f", f"whisper-server.*--port {at}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        verdict = bool(found.stdout.split())
    except (OSError, subprocess.SubprocessError):
        verdict = False  # cannot tell: fail CLOSED, because the cost of being wrong is the audio
    _OWNERSHIP[at] = (now, verdict)
    return verdict


def forget_ownership() -> None:
    """Drop every cached ownership verdict — a server was just started, stopped, or faked in a test.

    A module-level cache is per PROCESS, and a test process runs hundreds of tests: without this, a
    test that stubs `pgrep` leaves its answer standing for everything that follows it in the same
    worker, in file order, which is not an order anybody reads.
    """
    _OWNERSHIP.clear()


def serve_command(setup: Setup) -> list[str] | None:
    """The argv that starts a warm whisper-server, or ``None`` if it cannot be built.

    ONE server, and it answers both the live passes and the final transcription. There used to be
    a second one holding a small model for the partials; it was retired when the live pass began
    typing punctuation, because the marks it chose were the marks the operator kept.
    """
    binary, model = setup.whisper_server, setup.model
    if not binary or not model:
        return None
    return [
        binary,
        "-m",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(setup.port),
        "-t",
        setup.threads,
        "--convert",  # let the server transcode anything ffmpeg reads, not just wav
        # Stop whisper emitting "*sad*"/"[MUSIC]"-style sound annotations at the SOURCE rather than
        # filtering them afterwards. `is_hallucination` stays as the backstop: -sns reduces these
        # but does not eliminate them.
        "-sns",
        # EVERY CLIP IS ITS OWN CLIP. whisper.cpp keeps the text it decoded as context for what it
        # decodes next, and a SERVER keeps it across REQUESTS — so yesterday's sentence primes
        # today's, and under streaming, where one dictation is ~100 overlapping passes over the
        # same growing audio, it primes itself with a hundred near-copies of what it just said.
        # Measured on a 145s clip, same audio, same prompt, twice in a row: 543 characters one
        # run and 1108 the next, one of them collapsing into "And. Your. Job as a founder." nine
        # times over. That is the report — "a lot of points in between, it cuts the logic of the
        # sentence" — and it is also why the same words came out well before the stream existed.
        # `-mc 0` stores no text context, and the same two runs then came back CHARACTER FOR
        # CHARACTER identical, in whole clauses, with no repetition: "...that fits your workflow,
        # that you connect with that company, you like how they do things, and then go from there."
        #
        # What it costs is real and small: past 30s whisper decodes each window without the
        # previous window's words to lean on. A dictation is one window, the vocabulary prompt is
        # sent per request and still applies, and an unstable transcript is not worth a smoother
        # seam at 0:30.
        "-mc",
        "0",
    ]


def start_server(setup: Setup, *, wait: float = 60.0) -> bool:
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

    So the server is started in the caller's own scratch dir, which is writable, and already the one
    place recorded voice lives — whisper-server deletes its converted copy when it is done, so the
    "empty between sentences" promise there still holds.
    """
    if server_up(setup.port):
        # Adopted only if a whisper-server holds the port — see :func:`ours`. Anything
        # else answering there would be handed recorded audio and believed about what was said.
        return ours(setup.port)
    cmd = serve_command(setup)
    if cmd is None:
        return False
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(setup.scratch),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    deadline = time.time() + wait
    while time.time() < deadline:
        if server_up(setup.port):
            return True
        time.sleep(0.1)
    return False


def stop_server(port: int) -> int:
    """Stop the warm whisper-server this install started. Returns how many were stopped.

    The port ABOVE ours is swept too, and it is not a second server of ours: an older MurmurFlow
    ran a small model there for the live pass, and a version that no longer starts one must still
    stop the one it finds, or ~488 MB stays resident until the machine is next restarted.

    ``start_server`` detaches it with ``start_new_session=True`` so it outlives the listener, which
    is the whole point while dictation is installed — and a leak the moment it is not: 1.8 GB
    resident with nothing left to ask it anything, until the next reboot. BOTH of ours, matched on
    OUR two ports, because a whisper-server on any other port belongs to somebody else.
    """
    stopped = 0
    for which in (port, port + 1):
        try:
            found = subprocess.run(
                ["pgrep", "-f", f"whisper-server.*--port {which}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for token in found.stdout.split():
            with contextlib.suppress(ValueError, ProcessLookupError, PermissionError):
                os.kill(int(token), signal.SIGTERM)
                stopped += 1
    return stopped


def multipart(wav: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    """Build a multipart/form-data body for whisper-server's ``/inference`` (stdlib only)."""
    boundary = f"----speech{uuid.uuid4().hex}"
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
    """``"german"``, ``"German"`` and ``"de"`` are one answer; anything else passes through.

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


def confidence(payload: str) -> tuple[str, float, str]:
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


def transcribe_warm(wav: Path, setup: Setup, *, timeout: float = 60.0, language: str = "") -> Heard:
    """Transcribe via the warm server; empty text if it is not up or errors. Never raises.

    Reuses :mod:`whisper`'s language and vocabulary decisions, so every surface that transcribes
    hears your own proper nouns the same way.

    Asks for ``verbose_json`` rather than ``text`` purely to get the language score back; the
    transcript is identical either way.

    ``language`` overrides the configured one for THIS request, and it exists for streaming.
    Detecting the language is a whole extra encoder pass — measured at 0.75s of every 2.2s request
    on an M4 Pro — and a clip does not change language halfway through, so the partials after the
    first pin themselves to what the first one heard. See :func:`_stream_loop`.

    **It asks who holds the port before it sends anything** (:func:`ours`). Adopting a server was
    guarded and SENDING was not, which is the wrong half: a process that binds :func:`port` first is
    handed every clip you record and believed about what was in it — and what comes back is TYPED
    AT YOUR CURSOR. Cached, so this is a dict read on the partial path and a `pgrep` twice a minute.
    """
    if not wav.is_file() or not ours(setup.port):
        return Heard("")
    fields = {
        "response_format": "verbose_json",
        "language": language or setup.language,
        "prompt": setup.vocabulary,
        "temperature": "0",
    }
    try:
        body, content_type = multipart(wav, fields)
    except OSError:
        return Heard("")
    request = urllib.request.Request(
        f"{server_url(setup.port)}/inference",
        data=body,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError, ValueError):
        return Heard("")
    return Heard(*confidence(payload), warm=True)


# --- finding the pieces this install has -------------------------------------------------------

#: Homebrew's bin dirs. launchd hands an agent a minimal PATH that excludes them, so a bare
#: `shutil.which()` finds nothing when the listener runs from a plist while working fine in a
#: shell. Both tools are installed as launchd/Task Scheduler agents, and both learned this the
#: same way.
FALLBACK_BIN_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")


def resolve_bin(name: str) -> str:
    """Absolute path to ``name``, searching PATH then :data:`FALLBACK_BIN_DIRS`; ``''`` if absent.

    Never raises. The fallback dirs matter only under an agent (see the constant).
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in FALLBACK_BIN_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def pick_input(
    want: str, devices: Sequence[tuple[str, str]], *, default: str = "default"
) -> tuple[str, str]:
    """Choose the microphone from an enumerated list. ``(id, name)``, never raises.

    **Matched by NAME, never by position.** Device ids shift the moment a USB interface is plugged
    in — pinning one would silently start recording the wrong device, which is the failure nobody
    notices until they read the transcript. A name miss prefers anything calling itself a
    microphone over an aggregate interface, and failing that falls back to the system default, so
    the capture degrades to "wrong-ish mic" rather than to "broken".

    The ENUMERATION is the caller's, and so is ``default`` — an avfoundation index on macOS, a
    dshow name on Windows. Only the choosing is here, because the rule is about how people name
    microphones, not about an API.
    """
    wanted = want.strip().lower()
    for device, name in devices:
        if wanted and wanted in name.lower():
            return str(device), name
    for device, name in devices:  # name miss: prefer a real mic over an aggregate interface
        # ...in both spellings, because a German Windows calls it "Mikrofon" and the operator's
        # clients run German machines. It costs one `or` and it is the difference between picking
        # the built-in microphone and picking whatever aggregate device sorted first.
        lowered = name.lower()
        if "microphone" in lowered or "mikrofon" in lowered:
            return str(device), name
    return default, "system default"


# --- the recorder ------------------------------------------------------------------------------
#
# WHAT IS NOT HERE, and why: which microphone, where the scratch dir is, what the marker file is
# called, and how this platform names an audio input. Every one of those is a question about an
# INSTALL, so the caller answers them and passes the answer in — `capture` is the platform's own
# input arguments with the device already resolved, which is the seam a Windows port needs and the
# reason this file has no idea what avfoundation is.

#: The recorder this process spawned, when it spawned one. Held so :func:`_exited` can REAP it: a
#: child that has exited but not been waited on is a zombie, and signalling a zombie succeeds.
_PROC: subprocess.Popen[bytes] | None = None


@dataclass(frozen=True)
class Recording:
    """An in-flight capture: the ffmpeg pid and the wav it is writing."""

    pid: int
    wav: Path
    started_at: float

    @property
    def seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)


def current(state: Path) -> Recording | None:
    """The in-flight recording, or ``None``. Stale markers (dead pid) are cleaned up and ignored."""
    path = state
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


def start(
    *,
    ffmpeg: str,
    capture: Sequence[str],
    scratch: Path,
    state: Path,
    max_seconds: int = MAX_CLIP_SECONDS,
) -> Recording | None:
    """Begin capturing the mic to a fresh 16kHz mono wav; ``None`` if already recording or unable.

    Returns as soon as ffmpeg is spawned — CoreAudio needs ~300ms more before the first sample
    actually lands (measured; it is a device-start floor, not an ffmpeg tax, so a compiled helper
    would not beat it). Callers that cue the operator should cue on :func:`ready`, not on this
    return, or the first word is clipped.
    """
    if current(state) is not None:
        return None
    if not ffmpeg:
        return None
    wav = scratch / f"dictate-{int(time.time())}-{uuid.uuid4().hex[:8]}.wav"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        # A HARD CEILING on one clip, and it is a privacy control, not a convenience. ffmpeg is
        # spawned with `start_new_session=True` so it survives its parent: kill the daemon (launchd
        # restart, a crash, `kickstart -k`) while a clip is running and nothing ever stops it. Found
        # live on the operator's machine — THREE orphaned recorders, 4.5 hours each, 1.4 GB of his
        # voice on disk and the microphone hot the whole time, which is the exact incident this
        # product exists not to cause. `-t` makes the recorder bound its own life, with no
        # supervisor needed and nothing to remember.
        "-t",
        str(max_seconds),
        *capture,
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
        # until long after it did — and the operator gets his "start talking" cue two seconds late,
        # by which point he has already said the first half of his sentence.
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
        state.write_text(
            json.dumps({"pid": rec.pid, "wav": str(rec.wav), "started_at": rec.started_at}),
            "utf-8",
        )
    return rec


def ready(rec: Recording, *, timeout: float = 2.0) -> bool:
    """Block until the wav has actual audio frames in it (or ``timeout``). ``True`` if it does.

    The ~300ms CoreAudio start-up is invisible to the operator only if he is cued when the mic is
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
    our own child we reap it with ``waitpid``; when it is not (one CLI invocation starts it and
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


def stop(rec: Recording | None, *, state: Path) -> Path | None:
    """Stop the in-flight capture and return the finished wav (``None`` if nothing was recording).

    SIGINT (not SIGKILL) so ffmpeg writes the RIFF trailer — a killed ffmpeg leaves a wav whose
    header claims zero length and whisper decodes it as silence. This sits directly on the felt
    latency (it runs the instant the operator lets go of the key), so exit is detected by REAPING
    the child rather than polling it — see :func:`_exited`.
    """
    global _PROC
    rec = rec or current(state)
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
    _clear_state_for(rec.pid, state)
    return rec.wav if rec.wav.is_file() else None


def reap_orphans(*, scratch: Path, state: Path) -> int:
    """Stop every recorder left behind by a previous run, and delete its audio. Returns how many.

    A recorder is spawned with ``start_new_session=True`` so it outlives the call that started it —
    which also means it outlives the DAEMON. A launchd restart, a crash or a ``kickstart -k`` in the
    middle of a clip leaves ffmpeg running forever with the microphone open: found live on the
    operator's machine as three recorders, 4.5 hours each, 1.4 GB of his voice on disk. `-t`
    (:data:`MAX_CLIP_SECONDS`) bounds the damage; this ends it, because a daemon that is only now
    starting cannot own a clip from before it existed.

    Matched on the exact scratch-path pattern this module writes, so no other ffmpeg on the machine
    is ever a candidate. Best-effort and silent on any failure — a reaper must never keep the daemon
    from starting.
    """
    reaped = 0
    try:
        pattern = f"-y {scratch}/dictate-"
        found = subprocess.run(
            # `--` IS LOad-BEARING, and its absence is why this reaper never reaped anything. The
            # pattern begins with `-y`, so without the guard pgrep parses it as its own option and
            # exits with "illegal option -- y" before matching a single process. It failed silently
            # in exactly the shape this function is written to tolerate — empty stdout, no raise —
            # so every daemon start reported nothing to reap while an orphan held the microphone
            # open. Found live on the operator's machine: one recorder open since 4:03pm, writing
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
    # contract is that the operator's voice does not sit on disk (`voiceKeepAudio` keeps ONE file,
    # deliberately, and it is not named like these).
    with contextlib.suppress(OSError):
        for wav in scratch.glob("dictate-*.wav"):
            wav.unlink(missing_ok=True)
    state.unlink(missing_ok=True)
    return reaped


def _clear_state_for(pid: int, state: Path) -> None:
    """Drop the in-flight marker, but ONLY if it still describes ``pid``.

    A huddle answers on a worker thread so the operator can interrupt, which means his NEXT
    recording can already be running by the time the previous one is stopped. Unlinking
    unconditionally orphaned it — ffmpeg still capturing, ``current()`` reporting nothing — so the
    clip he was in the middle of speaking could never be finished. An unreadable marker is cleared,
    since a marker nobody can parse is worse than none.
    """
    try:
        raw = json.loads(state.read_text("utf-8"))
        if int(raw["pid"]) != pid:
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    state.unlink(missing_ok=True)
