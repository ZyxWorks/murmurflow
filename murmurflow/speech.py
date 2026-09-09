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
import math
import re
import wave
from pathlib import Path

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
