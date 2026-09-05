"""The transcriber: which whisper binary, which model, which language — and the COLD fallback.

The hot path is not here. A warm ``whisper-server`` on the loopback (see :mod:`murmurflow.dictate`)
answers in ~0.6s because the model is already in memory; this module is what runs when no server
could be started, and it pays the model load on every clip (~1s slower). Everything here is
best-effort by contract: a missing binary, a missing model, a timeout and an empty transcript all
come back as ``''`` and never as an exception, because the caller is a keyboard daemon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

# Supported STT binaries, in preference order. whisper.cpp builds (``whisper-cli``/``whisper-cpp``)
# are fastest and dep-free; ``whisper`` is openai-whisper; ``whisper-ctranslate2`` shares its CLI.
BINARIES: tuple[str, ...] = ("whisper-cli", "whisper-cpp", "whisper", "whisper-ctranslate2")

# openai-whisper-style CLIs use a positional file + --output_dir; whisper.cpp uses -f/-of flags.
_CPP_BINARIES = frozenset({"whisper-cli", "whisper-cpp"})

# ggml model basenames, BEST FIRST. `model()` returns the best one actually PRESENT, so dropping a
# bigger file into ~/.murmurflow/models/ upgrades transcription with no config edit, and deleting it
# degrades back down. Ordered by accuracy-per-second on Apple Silicon: large-v3-turbo is
# near-large-v3 quality at ~8x the speed, which is why it outranks plain large. The `.en` variants
# are English-only builds — better on English, useless on anything else — so each ranks just below
# its multilingual twin.
MODEL_PREFERENCE: tuple[str, ...] = (
    "ggml-large-v3-turbo.bin",
    "ggml-large-v3.bin",
    "ggml-large-v2.bin",
    "ggml-medium.bin",
    "ggml-medium.en.bin",
    "ggml-small.bin",
    "ggml-small.en.bin",
    "ggml-base.bin",
    "ggml-base.en.bin",
)

# The model that answers the LIVE PARTIALS while you are still talking, best first — and it is a
# SMALL one on purpose. Measured on an M4 Pro against a warm server, same clip, same prompt:
# large-v3-turbo answers a partial in 2.2-2.4s, `ggml-small` in 0.35-0.44s, `ggml-base` in 0.15-0.23s.
#
# The speed is only half of why this exists. whisper-server answers ONE request at a time, so a
# partial still decoding when you stop talking is time the FINAL transcription spends queued behind
# it — measured live at 1-2.3s added to the end of every sentence, which is exactly the moment
# somebody is waiting. A model that answers in 0.4s cannot cost more than 0.4s of that.
#
# `small` and not `base`, and that is a measurement rather than caution: on the same German clip
# base typed "das" where the speaker said "dass" and dropped a plural, while small returned
# character-for-character what large-v3-turbo did. A partial is PASTED and there is no un-paste, so
# a model that quietly rewords is not cheaper, it is wrong. `.en` variants rank below their
# multilingual twins for the same reason as in MODEL_PREFERENCE.
PARTIAL_PREFERENCE: tuple[str, ...] = (
    "ggml-small.bin",
    "ggml-small.en.bin",
    "ggml-base.bin",
    "ggml-base.en.bin",
)

#: What `murmurflow setup` fetches for the partials. ~488 MB, next to the ~1.6 GB main model.
DEFAULT_PARTIAL_MODEL = "ggml-small.bin"

# Where a ggml model may live, in search order. `~/.murmurflow/models/` is ours; the others are what
# `brew install whisper-cpp` lays down.
_MODEL_DIRS: tuple[str, ...] = (
    "/usr/local/share/whisper-cpp",
    "/opt/homebrew/share/whisper-cpp",
)

MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# What `murmurflow setup` downloads with no argument. large-v3-turbo is ~1.6GB and is the reason
# this tool transcribes proper nouns correctly; the smaller models are a real accuracy cliff.
DEFAULT_MODEL = "ggml-large-v3-turbo.bin"


def found_binary() -> str:
    """The first supported STT binary on PATH, or ``''`` if none is installed."""
    for name in BINARIES:
        if shutil.which(name):
            return name
    return ""


def available() -> bool:
    """True if any supported local speech-to-text binary is on PATH."""
    return bool(found_binary())


def _search_dirs() -> list[Path]:
    """Every directory a ggml model may live in, ours first."""
    return [config.home_root() / "models", *(Path(directory) for directory in _MODEL_DIRS)]


def model() -> str:
    """Path to the BEST available ggml model, or ``''`` if none is configured or found.

    whisper.cpp needs an explicit ``-m <model>`` — without one it cannot initialize and produces no
    transcript at all, which surfaces as "it heard nothing" rather than as an error. An explicit
    ``model`` path in the config always wins; otherwise the best model PRESENT is chosen by
    :data:`MODEL_PREFERENCE` rather than whichever happens to be found first.
    """
    configured = str(config.load().get("model", "")).strip()
    if configured and Path(configured).expanduser().is_file():
        return str(Path(configured).expanduser())
    for name in MODEL_PREFERENCE:  # best model first, wherever it lives
        for directory in _search_dirs():
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return ""


def partial_model() -> str:
    """Path to a SMALL model for the live partials, or ``''`` when none is installed.

    Empty is a working answer and not a failure: the partials then go to the same warm server the
    final transcription uses, exactly as they did before this existed. Slower, and never wrong.

    Deliberately ignores the ``model`` config override, which names the model that writes what you
    keep. Pinning that to a small model is a choice about the transcript; it must not also silently
    become the choice about the partials, and vice versa.
    """
    for name in PARTIAL_PREFERENCE:
        for directory in _search_dirs():
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return ""


def openai_model_name() -> str:
    """The ``--model`` NAME for an openai-whisper-style CLI, which downloads its own weights.

    Shares the one ``model`` knob with :func:`model`: a value that looks like a filesystem
    path (a ``/`` or a ``.bin``) belongs to whisper.cpp and is ignored here. The default stays
    ``base`` because these CLIs fetch weights on demand, and silently pulling 1.5GB during
    someone's first dictation is not a default to inflict on anyone.
    """
    configured = str(config.load().get("model", "")).strip()
    ok = configured and "/" not in configured and not configured.endswith(".bin")
    return configured if ok else "base"


def language() -> str:
    """The spoken language passed to whisper — ``language``, else ``auto``.

    **Pinning is the single cheapest speed win here, worth ~0.7s on every utterance.** Measured
    2026-08-14 against a warm whisper-server: ``language=en`` transcribed in ~1.2s where ``auto``
    took ~1.9s, consistently, on a 1.7s clip and an 11s one alike. Auto-detect also misfires on
    short clips, where one language decoded as another comes back as garbage.

    The default is still ``auto``, because pinning the WRONG language is worse than not pinning at
    all: forcing ``en`` onto German speech makes whisper *translate* rather than transcribe, and a
    fluent English paraphrase of something you said in German is a far more confusing failure than
    a slow transcription. Set this the moment you know you only ever speak one language into it.
    """
    return str(config.load().get("language", "")).strip() or "auto"


def threads() -> str:
    """Decode threads for whisper.cpp. Its own default of 4 leaves most of a modern Mac idle, which
    is felt on the large models. Capped at 8: whisper.cpp decode throughput flattens and then
    regresses past roughly that many threads.
    """
    return str(min(8, os.cpu_count() or 4))


def vocabulary() -> str:
    """The initial-prompt sentence biasing whisper toward the user's own proper nouns, or ``''``.

    This is context, not a find/replace — whisper still transcribes what was actually said, it is
    just primed to expect these words. It is the single cheapest accuracy win available: names,
    acronyms and jargon a general model has never heard are exactly what it guesses at.

    Keep the list SHORT. whisper caps the initial prompt at n_text_ctx/2 (~224 tokens) and silently
    truncates past it, so only words that actually get mangled belong here. A natural sentence
    primes better than a bare comma list, because it is decoded as preceding speech.
    """
    raw = config.load().get("vocabulary", [])
    words = [str(w).strip() for w in raw] if isinstance(raw, list) else str(raw).split(",")
    seen: dict[str, None] = {}
    for word in words:
        clean = word.strip()
        if clean:
            seen.setdefault(clean, None)
    if not seen:
        return ""
    return "Glossary: " + ", ".join(list(seen)[:60]) + "."


def _run(cmd: list[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Run a transcription command; ``None`` on any failure. Never raises, by contract."""
    try:
        return subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _first_txt(directory: Path) -> str:
    """Read the first ``*.txt`` a whisper binary produced in ``directory``; ``''`` if none."""
    for txt in sorted(directory.glob("*.txt")):
        try:
            return txt.read_text("utf-8", errors="ignore")
        except OSError:
            return ""
    return ""


def _json_language(sidecar: Path) -> str:
    """The language code whisper.cpp wrote beside the transcript (``result.language``); ``''``.

    Never raises: a missing, truncated or unparsable sidecar is "no opinion", which the caller
    treats as "do not judge" rather than as a rejection.
    """
    try:
        data = json.loads(sidecar.read_text("utf-8", errors="ignore"))
    except (OSError, ValueError):
        return ""
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return ""
    return str(result.get("language") or "").strip().lower()


def transcribe(audio: Path, *, timeout: float = 180.0) -> str:
    """The transcript alone, for callers that do not judge the language. See :func:`transcribe_heard`."""
    return transcribe_heard(audio, timeout=timeout)[0]


def transcribe_heard(audio: Path, *, timeout: float = 180.0) -> tuple[str, str]:
    """Transcribe a local file; ``(text, language code)``, ``("", "")`` on failure.

    THE LANGUAGE IS HALF THE POINT. The cold path used to report only text, so
    :func:`dictate.finish`'s "is that one of the languages you speak" gate was inert on it — and
    the cold path is exactly where a wedged warm server silently leaves you. Live (2026-08-17):
    MurmurFlow's own whisper-server answered every request "FFmpeg conversion failed", every clip
    fell through to here, and a bump on the desk came back as two sentences of Japanese and was
    typed into the terminal. Both gates that exist to stop that were warm-only.

    The COLD path — it loads the model on every call. Work happens in a private temp dir that is
    deleted before returning, so the binary's scratch output never lingers next to the caller's
    file. The caller's own audio is not copied or moved; deleting it stays the caller's job,
    because only the caller knows when the transcript is safely in hand.
    """
    binary = found_binary()
    source = Path(audio)
    if not binary or not source.is_file():
        return "", ""
    prompt = vocabulary()
    with tempfile.TemporaryDirectory(prefix="murmurflow-") as tmp:
        work = Path(tmp)
        if binary in _CPP_BINARIES:
            # whisper.cpp: -f <file> -otxt -of <stem> writes <stem>.txt; stdout is the fallback.
            # --carry-initial-prompt re-primes EVERY segment — without it the vocabulary bias dies
            # after the first ~30s window and a long clip drifts back to guessing. -np keeps the
            # stdout fallback free of progress noise.
            stem = work / "transcript"
            # `-oj` ALONGSIDE `-otxt`: the JSON sidecar carries `result.language`, the one thing
            # the cold path could never report — see :func:`transcribe_heard`. It costs a small
            # file write, not a second decode.
            cmd = [binary, "-f", str(source), "-otxt", "-oj", "-of", str(stem), "-np"]
            cmd += ["-l", language(), "-t", threads()]
            if prompt:
                cmd += ["--prompt", prompt, "--carry-initial-prompt"]
            found = model()
            if found:
                cmd += ["-m", found]
            proc = _run(cmd, cwd=work, timeout=timeout)
            spoken_code = _json_language(stem.with_suffix(".json"))
            txt = stem.with_suffix(".txt")
            if txt.is_file():
                try:
                    produced = txt.read_text("utf-8", errors="ignore").strip()
                except OSError:
                    produced = ""
                if produced:
                    return produced, spoken_code
            return (proc.stdout.strip() if proc and proc.stdout else ""), spoken_code
        # openai-whisper / whisper-ctranslate2: positional file + --output_dir writes <stem>.txt.
        cmd = [binary, str(source), "--model", openai_model_name(), "--output_format", "txt"]
        cmd += ["--output_dir", str(work)]
        if prompt:
            cmd += ["--initial-prompt", prompt]
        spoken = language()
        if spoken != "auto":  # these CLIs auto-detect when --language is omitted
            cmd += ["--language", spoken]
        _run(cmd, cwd=work, timeout=timeout)
        # These CLIs write no language sidecar in txt mode, so this path still reports none —
        # and reporting none is a "no opinion", never a rejection (`dictate.finish`).
        return _first_txt(work).strip(), ""
