"""Test collection setup.

Keeps ``murmurflow`` importable from a bare ``pytest`` (no ``uv run``, nothing installed) by
putting the package root on ``sys.path`` before ``tests/test_murmurflow.py`` can import it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from murmurflow import platforms


@pytest.fixture(autouse=True)
def _never_paste_on_the_real_machine(monkeypatch):
    """The suite once typed its fixtures into the operator's screen for a whole day.

    ``finish`` pastes through ``platforms.inject`` (a real ⌘V). This shuts that door for every test
    by RAISING, so a test that reaches it fails loudly instead of passing against a stub; a test
    that wants to watch the seam patches its own double over this one.
    """

    def _refuse(*_args, **_kwargs):
        raise AssertionError("a test reached the real paste")

    monkeypatch.setattr(platforms, "inject", _refuse)
    monkeypatch.setattr(platforms, "clipboard_set", _refuse)
