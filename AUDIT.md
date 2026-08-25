# Audit — 2026-08-25

A full audit of this repo at `main` (`865e1c8`), run as twelve agents: six finders, one per
dimension (core engine, platform layer, CLI/config, privacy promises, installers, simplification),
and six adversarial verifiers that re-read the cited code and tried to refute every claim. 21
findings survived verification; none were refuted. Deduplicated, they are the 18 items below.

This file is the handoff to the implementation session. It records findings only — nothing in this
branch changes behaviour. Each item names the file and line as of `865e1c8`, what is wrong, and the
smallest verified fix.

**Baseline at `865e1c8`:** 154 tests pass, `ruff check` clean, `mypy --strict` clean.
One local note: after the repo moved to `.zyx/code/murmurflow`, the checked-out `.venv` still
pointed at the old path; `rm -rf .venv && uv sync` fixes it (already done in this checkout,
`.venv` is gitignored).

---

## High

### 1. `stripFillers` deletes real words, not just fillers
`murmurflow/dictate.py:867` — reproduced live.

`_FILLER_LEAD` unconditionally strips a leading `hey`/`so`/`well`/`right`/…, and `_FILLER_WORD`
strips `you know`/`i mean`/`basically` **anywhere** they occur:

- "Do you know what time it is?" → "Do what time it is?"
- "Let me know if you know a good restaurant." → "Let me know if a good restaurant."
- "I mean what I said." → "what I said."

The docstring of `tidy()` (≈1044) already tells the story of this exact bug shipping once with a
leading "hey" — the response then was to default the feature off, not to fix the regex.

**Fix:** no syntactic heuristic separates filler-"hey" from real-"hey" (comma-bounding does NOT
fix the documented incident — that "hey" had a comma). Drop the ambiguous lexical entries
entirely: remove `yo`, `hey`, `okay so`, `ok so`, `so`, `well`, `alright`, `right` from
`_FILLER_LEAD` and `basically`, `you know`, `i mean` from `_FILLER_WORD`. Keep only the
non-lexical sounds (`um`, `uh`, `erm`, `hmm`), which are never real words. Update the README lines
that sell the feature (≈124–129 and the settings-table row ≈329) to match.
*This narrows a documented feature — flag it in the PR description so it can be vetoed in review.*

### 2. Windows clipboard/paste truncates 64-bit pointers
`murmurflow/platforms/windows.py:344` (also 315, 318, 347).

No `.restype`/`.argtypes` are declared anywhere in windows.py. `GlobalAlloc`, `GlobalLock`, and
`GetClipboardData` return pointer/handle-sized values, but ctypes' default restype is a 32-bit
`int` — on 64-bit Windows the truncated pointer feeds `ctypes.memmove` (line 351): crash on the
first paste, and clipboard restore silently fails. macos.py sets restype for every pointer-returning
call; windows.py sets none.

**Fix:** set `.restype = ctypes.c_void_p` on `_kernel32.GlobalAlloc`, `_kernel32.GlobalLock`, and
`_user32.GetClipboardData` right after the DLL handles are constructed, mirroring macos.py's
pattern. While there, sweep every other `_kernel32`/`_user32` call in the file for
pointer-or-larger return types.

### 3. `config set quietFloor` accepts a positive value and silently kills all dictation
`murmurflow/cli.py:571` — reproduced live.

The branch checks the value is numeric but never the sign, though its own error message says
"below zero". `quietFloor: 30` is accepted; peak dBFS is always ≤ 0, so the gate at
`dictate.py:1597` then discards **every** clip forever. Exactly the silent-misconfiguration class
this validation exists to prevent. The existing test (tests/test_murmurflow.py:918) only covers
the non-numeric case.

**Fix:** `if isinstance(value, bool) or not isinstance(value, (int, float)) or value >= 0:` —
plus a test for the sign.

### 4. install.ps1 reports success after a failed model download
`install.ps1:75` (also 31–32, 78).

`$ErrorActionPreference = "Stop"` does not check native .exe exit codes. Both `winget` calls
discard output, and `& $mf setup` (the 1.6 GB model download) and `& $mf install` are never
checked against `$LASTEXITCODE` — a network drop mid-download still ends in "Done." install.sh
gets this right (`set -eu` aborts on external-command failure).

**Fix:** a small check of `$LASTEXITCODE` after each of the four native calls, using the existing
`Die` helper.

### 5. The quickstart teaches the wrong default gesture
`README.md:38` and `docs/index.html:254`.

The quickstart says "hold Control+Option … (Prefer hands-free? `config set doubleTap true`)".
But `doubleTap` defaults to **true** (`dictate.py:2029`) and the default trigger is a double-tap of
left Control — holding Control+Option on a fresh install does nothing. README's own gesture table
(≈174) states the default correctly, contradicting its own quickstart.

**Fix:** rewrite both quickstarts: double-tap Control to start, tap again to stop; the hold
gesture is the opt-out via `config set doubleTap false`.

## Medium

### 6. `ctrl`/`alt`/`win` spellings are not accepted everywhere, despite the README promise
`murmurflow/hotkey.py:104` — reproduced live.

README:196 promises "ctrl/alt/win are accepted everywhere control/option/command are".
`TRIGGER_ALIASES` is a hand-list; `left_ctrl`, `right_ctrl`, `left_win`, `right_win`, `ctrl_win`,
`win_alt`, `ctrl_shift` are all rejected while their canonical twins exist.

**Fix:** replace the alias dict with per-token substitution:
`_WORD_ALIASES = {"ctrl": "control", "alt": "option", "win": "command", "super": "command",
"meta": "command"}` applied word-wise over `key.split("_")`. Smaller than the current dict and
covers every counterexample. Add a test over the full alias matrix.

### 7. `config set languages` accepts junk codes; one bad entry discards every clip
`murmurflow/cli.py:584` — reproduced live.

Only the length is checked, not that the code is alphabetic — the sibling `language` branch
(line 579) checks both. `["d3"]` passes, and then the clip-language gate throws away every
dictation.

**Fix:** reuse the alpha check from the `language` branch on each entry, plus a test.

### 8. Turning `keepAudio` off does not delete the kept clip
`murmurflow/dictate.py:1570`, handler in `cli.py` `_config` (≈677).

README (≈403) promises keepAudio "keeps that clip and its transcript **until you turn it off**".
`last.wav` is written in exactly one place and deleted in none; `reap_orphans()` only globs
`dictate-*.wav`. The recording outlives the opt-out indefinitely.

**Fix:** in the `config set` handler, when `keepAudio` is set falsy, delete
`~/.murmurflow/audio/last.wav` (`missing_ok=True`). Also soften the README's "and its transcript"
clause: the transcript line already written to the log is not retroactively purged.

### 9. Installers and package metadata still point at the pre-rename GitHub owner
`install.sh:4,19`, `install.ps1:3,20`, `pyproject.toml:26–27`.

The repo moved to the `ZyxWorks` org (README, CI badge, docs, and `git remote` all agree).
The two installers' `SOURCE` defaults and pyproject's Homepage/Issues still say
`hannesreinsch/murmurflow` — alive only via GitHub's transfer redirect, which breaks if the old
name is ever reclaimed. Then both installers 404 at once.

**Fix:** plain string replacement to `ZyxWorks/murmurflow` at all five sites.

### 10. CI never syntax-checks the installers
`.github/workflows/ci.yml:15`.

The only job is macos-latest running ruff/mypy/pytest over the package. install.sh and install.ps1
— the actual onboarding path, fetched raw from `main` — have zero coverage; a typo ships behind a
green badge. (The Windows *Python* backend IS unit-tested via mocks; only the entry scripts are
blind.)

**Fix:** add `bash -n install.sh` and a PowerShell-parser syntax check of install.ps1 to the
existing job. No OS matrix needed for that.

### 11. `doctor`'s hint suggests a no-op
`murmurflow/cli.py:386`.

The `_VERBS` hint offers `config set doubleTap true` — already the default, so it changes nothing.

**Fix:** change to `("config set doubleTap false", "hold the key instead of tapping it twice")`.

## Low

### 12. `flags()` is dead code with a false docstring
`murmurflow/hotkey.py:221`, `platforms/__init__.py:67`, plus the three platform impls.
Both docstrings claim keytest uses it; keytest polls `is_trigger_down` and nothing calls
`flags()` anywhere. **Fix:** delete it from all five files.

### 13. Stale comment contradicts the shipped default gesture
`murmurflow/hotkey.py:143` says "Hold stays the default"; double-tap is the default.
**Fix:** correct the comment.

### 14. `min_hold` sleeps the full floor instead of the remainder
`murmurflow/hotkey.py:291`. `if elapsed < min_hold: time.sleep(min_hold)` nearly doubles the
intended floor for a press just under it. **Fix:** `time.sleep(min_hold - elapsed)`.

### 15. README settings table omits `model` and `port`
`README.md:319`. `config.KEYS` has 13 keys; the table lists 11. **Fix:** add both rows, copying
the descriptions from `config.py`.

### 16. Package metadata says macOS-only; the product is macOS and Windows
`pyproject.toml:4` (and classifiers/keywords), `murmurflow/__init__.py:1`. **Fix:** mention
Windows in both descriptions, add the Windows OS classifier and a `windows` keyword.

### 17. "roughly 2,000 lines" is now ~5,700
`README.md:223`. **Fix:** say "a few thousand lines" so it does not drift again.

### 18. `recent_cues()` is dead code
`murmurflow/dictate.py:1934`. The reader side of the cue log; no caller anywhere. The writer
(`_note_cue`) is used and stays. **Fix:** delete the function.

---

## Suggested implementation order

Small, separately revertable PRs; every one keeps the gate green (pytest, ruff, mypy --strict):

1. **Validation + trigger aliases** — items 3, 6, 7, 11 (+ tests). Pure Python, all reproduced.
2. **stripFillers** — item 1 (+ README rows, + tests). The one product-behaviour change.
3. **Windows restype** — item 2. Cannot be run on this Mac; the mocked tests must still pass.
4. **keepAudio delete-on-off** — item 8 (+ test, + README clause).
5. **Installers + CI + metadata** — items 4, 9, 10, 16.
6. **Docs + dead code** — items 5, 12, 13, 14, 15, 17, 18.

What was audited and came back clean: the recording lifecycle and its 10-minute bound, orphan
reaping, the pause/resume lease, clipboard save/restore on macOS, the whisper server lifecycle,
audio deletion timing (before polish and paste, on error paths too), the no-network promise, and
the log's metadata-only content.
