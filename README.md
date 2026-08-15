# MurmurFlow

**Hold a key, talk, let go. The text appears at your cursor.** In Slack, in your terminal, in a
browser, in your notes app — anywhere. Nothing you say ever leaves your Mac.

A free, local alternative to the paid cloud dictation apps. No account, no subscription, no server,
no telemetry, and no Python dependencies at all.

```sh
brew install whisper-cpp ffmpeg
pipx install murmurflow      # or: uv tool install murmurflow
murmurflow setup             # downloads the speech model (~1.6 GB, once)
murmurflow install           # dictation is live now, and after every login
```

Hold **left Control**, say something, let go. That's the whole product.

---

## Why it exists

Dictation is the fastest input method most people never use, and the good implementations are all
cloud products: your microphone streams to someone else's servers, behind a subscription, under a
privacy policy that can change. The local pieces to do it properly are all right there —
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcribes better than most cloud APIs, and
macOS will tell you which keys are held and let you paste into the frontmost app.

So this is roughly 2,000 lines of standard library gluing those together.

### Speed, honestly

Measured on an M4 Pro, macOS 26, `large-v3-turbo`, an 11 second clip, 8 threads, nothing else
resident:

| step | seconds |
|---|---|
| transcribe, warm server, `language` **pinned** | **1.2** |
| transcribe, warm server, `language` on `auto` | 1.9 |
| transcribe, cold `whisper-cli`, model page-cached | 1.7-2.2 |
| first transcription after boot (1.6 GB model off disk) | 13.3 |
| microphone open, first ever | 9.9 |
| microphone open, thereafter | 0.3 |

Two things in that table are worth more than the headline number:

**The warm server is not what makes this fast in steady state.** Once the model file is in the OS
page cache, cold and warm are within noise of each other. What a resident `whisper-server` actually
buys is the *first* transcription of the day — 13.3s down to 1.2s — which is exactly the one you
form your opinion on, and exactly the one a freshly booted laptop serves. So the daemon starts it at
launch and pays that cost while nobody is waiting.

**Pinning the language is the real lever, worth ~0.7s per sentence.** It is the one setting most
worth changing:

```sh
murmurflow config set language en
```

Only do it if you really do speak one language into it. Pinning the *wrong* language is worse than
leaving it on `auto`: forcing `en` onto German speech makes whisper translate rather than
transcribe, and a fluent English paraphrase of what you said is far more confusing than a slow
transcript. That is why the default stays `auto`.

These numbers move a lot with your machine and what else is resident — two whisper-servers on one
Mac roughly doubled every row. Measure your own before believing any of them, including ours.

## What it does not do

- **No compiled helper.** Key state is polled through `CGEventSourceFlagsState` with stdlib
  `ctypes`. No Xcode, no code signing, no notarization, and no TCC grant that a rebuild invalidates.
- **No dependencies.** `pip install murmurflow` pulls in nothing. Two Homebrew binaries and macOS
  itself do the work.
- **No streaming.** It would cost a permanently resident microphone to save a fraction of a batch
  pass.
- **No LLM on the hot path**, unless you ask for one — see [polish](#polish-optional).
- **No Linux or Windows.** The capture (`avfoundation`) and the key polling are both macOS-only.
  This isn't a stub waiting to be filled in; it's what the tool is.

## Permissions

macOS will ask for two the first time, and neither can be granted from a script:

1. **Microphone** — to hear you.
2. **Accessibility** — to type into the app you're using.
   *System Settings → Privacy & Security → Accessibility*

If the key is never detected at all, **Input Monitoring** is the third — but test with
`murmurflow keytest` before granting it, because most Macs don't need it.

> **The apps you dictate into need nothing.** Every permission goes to this one process; Slack, your
> browser and your editor just receive a paste. You are not opening up your machine app by app.

If dictation transcribes but nothing appears, Accessibility is the reason nine times out of ten.

## The model is the accuracy

Model size is the single biggest lever on proper nouns and jargon. `murmurflow setup` gets
`large-v3-turbo` because it's near-`large-v3` quality at roughly 8x the speed. Anything smaller is a
real cliff — names come back as plausible nonsense.

Dropping a different ggml file into `~/.murmurflow/models/` upgrades or downgrades it with no config
change: the best model **present** wins.

```sh
murmurflow setup base            # smaller and faster, noticeably worse
```

Teach it your own words — the cheapest accuracy win there is:

```sh
murmurflow config set vocabulary '["Kubernetes", "Postgres", "Anthropic", "Reinsch"]'
```

## Settings

`~/.murmurflow/config.json`, one flat object. `murmurflow config` prints every key.

| key | what it does |
|---|---|
| `trigger` | the hold key. Default `left_control` — where macOS puts dictation, so your hand knows it |
| `doubleTap` | `true` = tap twice to start, twice to stop, instead of holding |
| `language` | `en`, `de`, … Default `auto`. Pinning saves ~0.7s per sentence, so pin it if you can |
| `inputName` | part of a microphone name. Default: system default. `murmurflow devices` lists them |
| `vocabulary` | proper nouns to bias the transcriber toward |
| `cue` | tone preset: `glass`, `soft`, `pure`, `wood`, `bell`, `system`, or `off` |
| `polishCommand` | see below |
| `keepAudio` | keep the last clip for debugging a bad transcription |

**Right Option is deliberately not the default.** On a German layout it's AltGr — the dead key for
`@ € \ | ~ [ ] { }` — so binding dictation there fires the microphone on every email address and
code bracket.

### Polish (optional)

Resolving spoken self-corrections ("go left, no wait, right" → "go right") is the one thing a
language model does that a regex cannot, and it's most of what people mean when a paid dictation app
"reads their mind". It's off by default because it isn't free: spawning a model costs seconds on
top of a transcription measured in low single digits.

`polishCommand` takes the transcript on **stdin** and prints the cleaned text. Anything of that
shape works:

```sh
murmurflow config set polishCommand "ollama run llama3.2"       # stays local
murmurflow config set polishCommand "claude -p --model haiku"   # sends text to Anthropic
```

> **This is the only way your words can leave your machine, and only because you sent them there.**
> A local runner keeps everything local. A command pointing at a hosted API does not. Unset — the
> default — means the answer to "does this phone home" stays no.

A broken polish command degrades to the plain transcript. It never costs you the sentence.

## Commands

```
murmurflow install      install the login agent so dictation is always live
murmurflow listen       run the daemon in this terminal instead (blocks)
murmurflow doctor       what is missing, and the one command that fixes each thing
murmurflow keytest      does this Mac actually see your trigger key?
murmurflow devices      list microphones
murmurflow cues         play the three tones, or switch preset
murmurflow setup        download a speech model
murmurflow config       show or change settings
murmurflow toggle       start/stop one recording (bind this to a macOS Shortcut)
murmurflow transcribe   transcribe an audio file and print the text
murmurflow uninstall    stop dictation and remove it from login
```

`doctor` and `keytest` exist because dictation fails in exactly four ways — the key isn't seen, the
microphone isn't heard, the model isn't found, the text isn't typed — and from the outside those are
indistinguishable. One run of each separates them.

## Privacy, concretely

- Audio is written to `~/.murmurflow/audio/` and deleted **the instant it's transcribed** — before
  the optional polish call and before the paste, so a crash downstream can't leave your voice on
  disk.
- The recorder bounds its own life at 10 minutes, so a daemon killed mid-clip can't leave the
  microphone hot. (This is not hypothetical: it was found in development as three orphaned
  recorders, 4.5 hours each, 1.4 GB of audio, microphone open the whole time.)
- The transcript is never inspected, logged or filtered. It goes to your clipboard, then your
  cursor, and your previous clipboard contents are put back — all of them, not just text. Copy a
  screenshot while a dictation is in flight and the screenshot is still on your clipboard
  afterwards.
- Nothing in this repo makes a network request except `murmurflow setup`, which downloads the model
  from Hugging Face.

## Development

```sh
uv run pytest        # hermetic: no microphone, no whisper, no network
uv run ruff check .
```

The hardware paths (`start`/`finish`/`inject`) have no unit tests on purpose — a mock of CoreAudio
only ever tests the mock. `murmurflow doctor` and `murmurflow keytest` are how those are exercised.

## License

MIT. See [LICENSE](LICENSE).

---

*MurmurFlow is not affiliated with, endorsed by, or connected to Wispr AI, Inc. or any other
dictation product. Product names mentioned are the trademarks of their respective owners.*
