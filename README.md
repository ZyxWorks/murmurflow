# MurmurFlow

### Write at the speed you talk.

Tap a key twice and talk. The words land at your cursor **while you are still talking**, in
whatever app you are already in: Slack, your terminal, a browser, your notes. It transcribes on
your own machine, so nothing you say ever leaves it.

[![CI](https://github.com/ZyxWorks/murmurflow/actions/workflows/ci.yml/badge.svg)](https://github.com/ZyxWorks/murmurflow/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-c9903f)](LICENSE)
![Platform: macOS and Windows](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-6a6c77)
![Dependencies: none](https://img.shields.io/badge/python%20dependencies-none-6a6c77)

A free, local alternative to the paid cloud dictation apps. No account, no subscription, no server,
no telemetry, and no Python dependencies at all. **macOS and Windows.**

[**The product page**](https://zyxworks.github.io/murmurflow/) ·
[Our other tools](https://zyxworks.com/) ·
[What we do for companies](https://zyxworks.com)

**macOS** — paste into Terminal:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/ZyxWorks/murmurflow/main/install.sh)"
```

**Windows** — paste into PowerShell:

```powershell
irm https://raw.githubusercontent.com/ZyxWorks/murmurflow/main/install.ps1 | iex
```

Either one works on a machine with nothing on it: no package manager, no Python, no developer
tools, and on Windows no Administrator either. It installs what is missing, downloads the speech
speech models (~2.1 GB, once), and turns dictation on for every login. On macOS it also opens the one
permission switch the OS will not let a script flip for you; Windows has no such switch.

Then double-tap **Control**, say something, and tap it once more to stop. That's the whole product,
and it is what macOS puts its own dictation on. (Rather hold a key while you talk?
`murmurflow config set doubleTap false` — then it is **Control+Option** held together, Windows
**Control+Alt**.)

### Why bother

| 150 wpm | 40 wpm | 0 € / mo |
|---|---|---|
| how fast you speak, without trying | how fast an average person types | MIT licensed, with no paid tier above it |

A 200-word message is five minutes of typing. Say it instead and you are done in eighty seconds,
hands still on the desk. Ten of those a day is most of an hour back, every day.

Averages, not promises: conversational speech runs about 130 to 150 words a minute, and an average
typist about 40. Your own numbers are yours to check, and [the product
page](https://zyxworks.github.io/murmurflow/) shows the same figures with the measured speed
table beside them.

<details>
<summary>Rather do it by hand?</summary>

macOS ships Python 3.9 and MurmurFlow needs 3.11+, so `uv` (which brings its own) is the path with
the fewest ways to go wrong:

```sh
brew install whisper-cpp ffmpeg uv
uv tool install --python 3.13 git+https://github.com/ZyxWorks/murmurflow
murmurflow setup             # downloads both speech models (~2.1 GB, once)
murmurflow install           # dictation is live now, and after every login
```

On Windows the only difference is whisper: there is no package for it, so the installer downloads
[whisper.cpp's own release build](https://github.com/ggml-org/whisper.cpp/releases) and unzips it
into `%LOCALAPPDATA%\MurmurFlow\bin`. Everything else is the same:

```powershell
winget install Gyan.FFmpeg astral-sh.uv
uv tool install --python 3.13 git+https://github.com/ZyxWorks/murmurflow
murmurflow setup
murmurflow install
```

If `murmurflow` is not found afterwards, `~/.local/bin` is not on your `PATH` — `uv tool
update-shell` fixes that for the next terminal you open.

</details>

<details>
<summary>What is different on Windows</summary>

Four things, and none of them is a setting you have to find. The trigger names, the config file,
every command and every message are identical — `alt`, `win`, `super` and `meta` are accepted
spellings of the same keys, so a config file moves between machines unchanged.

| | macOS | Windows |
|---|---|---|
| permission to type | Accessibility, granted per executable | **none** — there is nothing to grant |
| the trigger `fn` | works | not available: the key is handled in keyboard firmware and never reaches the OS |
| the trigger `right_command` | not available: these Macs report the right-side keys as the left ones | works |
| the clipboard, while a dictation lands | every flavour is restored, images included | text only |

One ceiling worth knowing: Windows does not let an ordinary program type into a window running as
Administrator. Dictating into an elevated console needs MurmurFlow running elevated too.

**Linux is not supported yet**, and that is a decision rather than an oversight. Recording and
typing are both small there; the hotkey is not, because Wayland exposes no global hotkey API at
all. `murmurflow doctor` runs on Linux and says exactly that.

</details>

### Two things worth knowing

**It types while you talk, and there is no setting for it.** Tap twice and start speaking: the
words begin landing at your cursor about two seconds in, in lumps, and the rest arrives when you
tap again to stop. It used to be an opt-in flag called `stream` and that flag is gone — opt-in
meant almost nobody ever saw the good version of the tool. See
[how it works](#how-the-words-arrive-while-you-talk).

**It makes exactly one sound.** A short tick the moment the microphone is genuinely live, so you
know when to start. There is no setting, because there is nothing to choose: nothing marks the end
(the text landing already does) and nothing marks a failure (that is a line in the log, not a noise
in a meeting). The three configurable tones and the preset system they needed are gone.

**And the first word is not missing any more.** CoreAudio takes about 0.6s to hand over its first
buffer, so a recorder opened on the second tap starts half a sentence late and no decoder can get
that audio back. The recorder is now opened on the **first** tap instead and the second tap claims
it, already live — which is also why the tick usually sounds the instant you finish tapping.
Measured over a five-second sentence: 0.35s of speech lost before, 0.04s after. A tap that never
becomes a pair stops the recorder within 1.5s and deletes what it caught.

**What you said is what you get.** Nothing is removed, reworded or reordered. If you say
"hey, ship it on Friday", "hey" appears. There is an opt-in filler strip, and it removes **sounds
only** — never a word. "hey", "so", "well" and "you know" were on that list once, and each of them
is also an ordinary word: "Do you know what time it is?" came back as "Do what time it is?". A word
deleted is invisible; an "um" left in costs one keystroke.

```sh
murmurflow config set stripFillers true   # deletes um / uh / erm / hmm, and nothing else
```

**And what you did NOT say never appears.** Handed a recording of a room, whisper does not answer
"nothing" — it answers fluently, in a language picked at random, and that sentence gets typed
into whatever window is in front. Level and confidence catch most of it. Naming the languages you
speak catches the rest, and costs nothing (it does not pin the decoder):

```sh
murmurflow config set languages '["de", "en"]'
```

**Double-tap, or hold.** Double-tap is the default: tap `left Control` twice to start, tap again
to stop. It is what macOS's own dictation does, it is the only gesture that survives a long
sentence, and it is the safer of the two for the reason in *The trigger* below. Holding is there
for one short sentence at a time:

```sh
murmurflow config set doubleTap false   # hold the key while you talk instead
```

Both take effect immediately — `config set` restarts the listener for you.

### Lending the key to another program

Something else wants the same double-tap for a moment — a voice assistant taking a turn, a screen
recorder that must not have the microphone pulled out from under it. That is a **pause**, not a
stop: the listener stays up and the whisper server stays warm, and only the trigger stands down.

```sh
murmurflow pause --seconds 120 --who "a Zyx huddle"
murmurflow resume                                   # or just wait
```

**Every pause expires**, and that is the design rather than a safety net. A borrower that crashes
holding the key would otherwise leave dictation silently dead with nothing on screen to explain it,
which is the worst failure this tool can have. The default is five minutes, the ceiling is an hour,
and `murmurflow doctor` names the holder and the time left for as long as it lasts.

An explicit `murmurflow toggle` still records while the key is lent — a pause stands the *trigger*
down, it does not disable dictation.

### The trigger: the gesture picks the key

| gesture | its key | why |
|---|---|---|
| **double-tap** (default) | **left Control** | two deliberate taps is a shape no shortcut has |
| **hold** | **⌃⌥ together** | a hold starts on the same key-down a shortcut does, so it needs a combo |

That is one rule, and it is worth understanding before you rebind.

The listener **polls** key state rather than intercepting it — that is what keeps this
dependency-free and out of Input Monitoring — so it can never take a keypress away from anything
else. A **hold** on a bare modifier therefore fires on every `⌃C` and every `⌃←`: the microphone
opens before the chord guard can tell it was a shortcut. It discards the audio correctly, but the
microphone did open.

A **double-tap** has no such problem. Two taps of one key inside half a second is not a shape any
shortcut has, and the one that could imitate it (`⌃C` then `⌃C`) is thrown out by the chord guard.
So double-tap gets one ordinary key — easier to perform than two at once, and the same key on every
keyboard there is.

```sh
murmurflow config set trigger left_control      # one key
murmurflow config set trigger command_option    # ⌘⌥
murmurflow config set trigger ctrl_alt          # the same as control_option, spelled the other way
```

`ctrl`/`alt`/`win` are accepted everywhere `control`/`option`/`command` are, so a config written on
one keyboard reads on another.

> **macOS has its own "press Control twice for dictation", and it is on by default on many Macs.**
> On a Control trigger both fire, and Apple's microphone panel lands on top of this one.
> `murmurflow doctor` checks for it and prints the fix; you do not have to remember this.
> *System Settings → Keyboard → Dictation → Shortcut → Off.*

> **Every sentence typed twice?** Something else is listening on the same key. `murmurflow doctor`
> names it and prints the one command that stops it. The usual one is `zyx voice listen` —
> murmurflow was extracted from zyx, and murmurflow's own lock cannot see another program's daemon.

`murmurflow keytest` shows what this Mac actually reports for every bindable key. Use it before
believing any of the above about your hardware — some MacBooks report the right-side Command and
Option keys as the left ones, so a `right_*` trigger can never fire there.

---

## Why this one is local

Dictation is the fastest input method most people never use, and the good implementations are all
cloud products: your microphone streams to someone else's servers, behind a subscription, under a
privacy policy that can change. The local pieces to do it properly are all right there —
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcribes better than most cloud APIs, and
macOS will tell you which keys are held and let you paste into the frontmost app.

So this is a few thousand lines of standard library gluing those together.

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
| speech lost at the start of a clip (pre-roll on) | 0.04 |

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
- **No always-on microphone.** Nothing is listening between sentences. Typing the words out while
  you talk decodes the clip you are already recording, and nothing else — see
  [how the words arrive](#how-the-words-arrive-while-you-talk). The one exception is the half
  second the recorder opens early, on the first of your two taps, so the first word is not lost;
  a tap that never becomes a pair stops it and deletes what it caught.
- **No LLM on the hot path**, unless you ask for one — see [polish](#polish-optional).
- **No Linux.** Recording and typing are both small there; the hotkey is not, because Wayland
  exposes no global hotkey API at all. `murmurflow doctor` runs on Linux and says exactly that.

## Permissions

macOS will ask for two the first time, and neither can be granted from a script:

1. **Microphone** — to hear you.
2. **Accessibility** — to type into the app you're using.
   *System Settings → Privacy & Security → Accessibility*

**Both rows are called `MurmurFlow`.** `murmurflow install` builds a small app bundle at
`~/Applications/MurmurFlow.app` purely so that is true. Without it macOS names the row after the
*interpreter* — `python3.13` — which nobody scrolling for "murmurflow" finds, and switching that on
would hand the microphone and your keyboard to every other Python tool sharing it.

If the key is never detected at all, **Input Monitoring** is the third — but test with
`murmurflow keytest` before granting it, because most Macs don't need it.

> **The apps you dictate into need nothing.** Every permission goes to this one program; Slack, your
> browser and your editor just receive a paste. You are not opening up your machine app by app.

`murmurflow doctor` answers both questions for real — it asks the bundle, not itself. If dictation
transcribes but nothing appears, Accessibility is the reason nine times out of ten.

## The model is the accuracy

Model size is the single biggest lever on proper nouns and jargon. `murmurflow setup` gets
`large-v3-turbo` because it's near-`large-v3` quality at roughly 8x the speed. Anything smaller is a
real cliff — names come back as plausible nonsense.

Dropping a different ggml file into `~/.murmurflow/models/` upgrades or downgrades it with no config
change: the best model **present** wins.

```sh
murmurflow setup base            # smaller and faster, noticeably worse
```

`setup` also fetches `small`, which is not a downgrade of the transcript: it is the separate model
that answers the live pass while you talk. See
[how the words arrive](#how-the-words-arrive-while-you-talk).

Teach it your own words — the cheapest accuracy win there is:

```sh
murmurflow config set vocabulary '["Kubernetes", "Postgres", "Anthropic", "Reinsch"]'
```

## Settings

`~/.murmurflow/config.json`, one flat object. `murmurflow config` prints every key.

| key | what it does |
|---|---|
| `trigger` | the key. Default follows the gesture: `left_control` for double-tap, `control_option` for hold |
| `doubleTap` | tap twice to start, tap again to stop. **On by default.** `false` = hold the key while you talk |
| `language` | `en`, `de`, … Default `auto`. Pinning saves ~0.7s per sentence, so pin it if you can |
| `languages` | the languages you actually speak, e.g. `["de","en"]`. A clip whisper reads as any other one is dropped. Empty = accept all |
| `inputName` | part of a microphone name. Default: system default. `murmurflow devices` lists them |
| `vocabulary` | proper nouns to bias the transcriber toward |
| `polishCommand` | see below |
| `stripFillers` | `true` = delete the sounds `um` / `uh` / `erm` / `hmm`, and nothing else. **Off** — you get verbatim |
| `quietFloor` | peak dBFS below which a clip is a room and not a sentence. Default `-30` |
| `model` | path to a ggml model file, overriding the best one found in `~/.murmurflow/models/` |
| `port` | loopback port for the warm whisper-server. Default `8479`. The live model's server takes the next one up |
| `keepAudio` | keep the evidence for one bad transcription: the last clip, and the transcript in the log. Off, so your sentences are not written down |

**Right Option is deliberately not offered as a default.** On a German layout it's AltGr — the dead
key for `@ € \ | ~ [ ] { }` — so binding dictation there fires the microphone on every email
address and code bracket.

### How the words arrive while you talk

**On by default, with no setting to find.** The words land at your cursor in lumps while you are
still talking, instead of arriving in one paste when you stop.

What it actually does: it re-decodes the clip you are currently recording, over and over, and types
the words that two passes in a row agreed on. Agreement is the safety. Whisper revises — give it
another second of audio and it re-reads what it already had, now that it knows how the sentence
ends — so typing each pass's best guess would type words the next pass withdraws, and nothing can
un-type them. The first pass has nothing to agree with, so it holds back its last four words
instead.

**Two models, because they do different jobs.** `large-v3-turbo` (~1.6 GB) writes the transcript
you keep. `small` (~488 MB) answers the live pass while you are still talking, on its own
whisper-server, on its own port. `murmurflow setup` fetches both.

That is not only about speed. **whisper-server answers one request at a time**, so a live pass
still decoding when you stop talking is time the *final* transcription spends queued behind it —
measured at 1 to 2.3 seconds added to the end of every sentence, at exactly the moment somebody is
waiting for it. A separate process cannot queue against itself.

`small` and not `base`, and that is a measurement rather than caution: on the same German clip
`base` typed *das* where the speaker said *dass* and dropped a plural, while `small` returned
character-for-character what `large-v3-turbo` did. A live word is pasted and **there is no
un-paste**, so a model that quietly rewords is not cheaper, it is wrong.

**Measured** on an M4 Pro, macOS 26, `language` on `auto`, one 10.7 second sentence:

| | one big model | with the live model |
|---|---|---|
| one live pass | 2.2s | **0.4s** |
| first words at the cursor | 3.7s | **1.4s** |
| lumps of text during the sentence | 5 | **12** |
| the final transcription, after you stop | 2.9s | **1.7s** |
| still left to paste when you stop | 62 of 195 chars | **21 of 195** |

Without the live model everything still works — the live pass goes to the big server, in ~2s lumps,
as it did before. `murmurflow doctor` says which one you are on, and every clip's line in the daemon
log ends with what streaming actually did: `stream 12x → 11 typed`.

**Detecting the language is a whole extra encoder pass** — 0.75s of every 2.2s, measured — and one
clip does not change language halfway through, so every pass after the first pins itself to what
the first one heard. It costs nothing: the pin is only taken from a pass that already cleared the
confidence gate, only to a language on your `languages` list when you have one, and the **final**
transcription is never pinned, so the language gate still judges the real clip.

Two things to know:

- **It needs the tap gesture, and that is not a preference.** A paste is a synthetic ⌘V, and in
  hold-to-talk the trigger is a modifier that is physically down — every paste would be sent as
  ⌥⌘V into whatever app you are in. With `doubleTap false` it stands down, and the daemon says so
  on the line it prints at start-up.
- **It needs the warm whisper-server.** Partials never fall back to the cold `whisper-cli`: that
  would spawn a model every 1.2s and make your final transcription slower, not faster. No warm
  server, and the words simply arrive at the end as they always did.
- **The last pass can still reword what is already typed.** Usually punctuation or a capital. There
  is no un-paste and deliberately no attempt at one: synthesising backspaces into an app whose
  cursor may have moved since would delete text that was never ours. A word left as first heard is
  a cosmetic loss; a doubled half-sentence is not.

Shrinking the encoder window per request (`audio_ctx`) was tried instead and thrown out: it does
cut a pass to 0.8s, and on some clips it returns
fluent invented text — *"the final pass has to line a line. So the final pass has to line a line"* —
at every window size tried, deterministically enough that two passes agree on it and it gets typed.
Fast and occasionally making things up is the one trade a dictation tool cannot take.

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
murmurflow install      install (or UPDATE) dictation, and keep it live after every login
murmurflow listen       run the daemon in this terminal instead (blocks)
murmurflow doctor       what is missing, and the one command that fixes each thing
murmurflow keytest      does this Mac actually see your trigger key?
murmurflow devices      list microphones
murmurflow setup        download both speech models (the big one, and the fast live one)
murmurflow config       show or change settings
murmurflow toggle       start/stop one recording (bind this to a macOS Shortcut)
murmurflow transcribe   transcribe an audio file and print the text
murmurflow pause        lend the trigger key to another program for a while
murmurflow resume       take it back
murmurflow trigger      print the trigger key this install is on, for another program to read
murmurflow uninstall    stop dictation and remove it from login
```

Every clip the daemon handles is logged — how long you held the key, how much audio actually
landed, the peak level, the transcribe time, how many characters came back and the app the paste
went to. **Not what you said.** That file is `~/.murmurflow/listen.log`, `murmurflow doctor` prints
the path, and nothing rotates it — which is exactly why your sentences are not in it. Debugging one
bad dictation and want them? `keepAudio` keeps the clip and the transcript together for as long as
you leave it on, and deletes the clip when you turn it off.

`config set` refuses a setting it does not recognise and a value that cannot work — a trigger name
this machine cannot poll, a model path that is not there, a language you did not say you speak. All of those
used to be accepted and then fail silently, which is the same symptom as broken hardware.

**`install` is also `update`.** The listener does not run your checkout — `uv tool install` made
a copy of the package and launchd runs that — so a `git pull` alone changes a directory the running
program never reads, silently. `murmurflow install` therefore re-installs the package from wherever
it came from first (a local checkout, a git URL, PyPI), then re-executes itself out of the new copy
and registers the agent. Two commands, and only because the first one is git:

```sh
git pull && murmurflow install
```

An update that cannot run never blocks the install: no `uv`, a source that has moved, a network
that is down — each prints a line and carries on with the copy already on the machine.

`doctor` and `keytest` exist because dictation fails in exactly four ways — the key isn't seen, the
microphone isn't heard, the model isn't found, the text isn't typed — and from the outside those are
indistinguishable. One run of each separates them.

## Privacy, concretely

- Audio is written to `~/.murmurflow/audio/` and deleted **the instant it's transcribed** — before
  the optional polish call and before the paste, so a crash downstream can't leave your voice on
  disk.
- Each pass copies the clip so far to a second file next to it and deletes that
  copy the moment the pass ends, pass or fail. The copy exists because the recorder has not finished
  writing the original's header yet and reading it means patching one — never the file ffmpeg is
  still appending to. It lives for the length of one decode and it is inside the same directory the
  clip is, so it is covered by the same deletion and the same 10-minute ceiling.
- The recorder bounds its own life at 10 minutes, so a daemon killed mid-clip can't leave the
  microphone hot. (This is not hypothetical: it was found in development as three orphaned
  recorders, 4.5 hours each, 1.4 GB of audio, microphone open the whole time.)
- The transcript is never inspected, filtered, or written down. It goes to your clipboard, then
  your cursor, and your previous clipboard contents are put back — all of them, not just text. Copy
  a screenshot while a dictation is in flight and the screenshot is still on your clipboard
  afterwards. The one exception is `keepAudio`, which you turn on yourself to debug a bad
  dictation. It keeps that clip until you turn it off — switching it off deletes it. The transcript
  it also writes into the daemon log is a line in a log, and turning the setting off does not go
  back and remove it; delete the log yourself if you want it gone.
- If a paste **cannot** land — Secure Input is on, or Accessibility was never granted — your words
  go to the clipboard so you can paste them by hand, and that replaces what was on it. The daemon
  says so rather than leaving you to find out: there is no way to put the old contents back once
  the recovery paste is the only copy of what you just said.
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

## Where this comes from

MurmurFlow is one of the tools **[ZyxWorks](https://zyxworks.com)**, a product studio and forward
deployed engineering practice, built for itself and gave away. It was extracted from
[Zyx](https://zyxworks.com#zyx), the OS the studio runs on, which is also why a Mac running both
types every sentence twice until you turn one off.

The other one is **[Agent Office](https://zyxworks.github.io/agent-office/)**: several coding
agents in one tmux window, each in its own git worktree, and the one that has stopped and is waiting
on you says so on its border.

**Product:** [page](https://zyxworks.github.io/murmurflow/) ·
[all our tools](https://zyxworks.com/) ·
[issues](https://github.com/ZyxWorks/murmurflow/issues)

**Studio:** [what we do for companies](https://zyxworks.com) ·
[Zyx](https://zyxworks.com#zyx) ·
[GitHub](https://github.com/ZyxWorks)

**Legal:** [MIT licence](LICENSE) ·
[privacy](https://zyxworks.com/legal#privacy) ·
[imprint](https://zyxworks.com/legal#imprint)

*MurmurFlow is not affiliated with, endorsed by, or connected to Wispr AI, Inc. or any other
dictation product. Product names mentioned are the trademarks of their respective owners.*
