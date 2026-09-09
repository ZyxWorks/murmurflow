"""Nothing in this suite may reach the real keyboard, clipboard or microphone.

**This file exists because the suite typed into the operator's screen for a whole day.** Two tests
drive `_stream_loop` with a fixed `Heard` and monkeypatch `dictate._inject` — the CLIPBOARD path —
believing that was the way out to the machine. It is not the only one: :func:`dictate.place` tries
`platforms.type_text` FIRST, and that is a real ``CGEventKeyboardSetUnicodeString`` with nothing in
front of it. So every run of the suite typed both fixtures into whatever window had focus, back to
back and with no space between them:

    hello there my friend and also yougokigen you desu ne totemo ii tenki

Reported as "I keep getting this same random paste everywhere, even tho I'm not using murmurflow",
which is exactly right: it was not MurmurFlow, it was MurmurFlow's tests, and the giveaway was that
the text was byte-identical every time — a hallucination is different every time, a fixture is not.

A per-test monkeypatch cannot fix this class of bug, because the bug is a test that did not know it
had a second door. So the doors are shut here for EVERY test, autouse, and a test that wants to
watch one opens its own double over the top.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murmurflow import platforms


@pytest.fixture(autouse=True)
def _never_type_on_the_real_keyboard(monkeypatch):
    """Shut the door that was open: ``platforms.type_text`` and nothing else.

    ONE door, deliberately. The clipboard and the recorder have tests that drive them on purpose,
    with their own doubles over the OS calls inside; stubbing those here would replace the thing
    under test with this stub and the assertions would pass against nothing — a suite that reports
    green over a function it never ran is worse than the bug being fixed.

    ``""`` is what a keyboard that typed the whole chunk returns, so :func:`dictate.place` reports
    success and never falls through to the clipboard behind it.
    """
    monkeypatch.setattr(platforms, "type_text", lambda _text: "")
