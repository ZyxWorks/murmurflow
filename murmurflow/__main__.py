"""``python -m murmurflow`` — the same entry point as the console script, so the launchd agent
works from a checkout that was never `pip install`ed.
"""

import sys

from .cli import main

sys.exit(main())
