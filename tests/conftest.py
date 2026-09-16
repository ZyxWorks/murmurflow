"""Test collection setup.

Keeps ``murmurflow`` importable from a bare ``pytest`` (no ``uv run``, nothing installed) by
putting the package root on ``sys.path`` before ``tests/test_murmurflow.py`` can import it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
