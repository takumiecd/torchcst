"""Existing shared-L4 entry point; implementation lives in tools/colab."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.colab.scripts.pool import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F405
