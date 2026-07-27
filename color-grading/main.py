"""Top-level launcher for the colour-grading analysis engine.

Thin wrapper so the project can be run as ``python main.py <image> -o outputs``
from the repository root.  All logic lives in :mod:`color_analyzer.main`.
"""

from __future__ import annotations

from color_analyzer.main import run

if __name__ == "__main__":
    raise SystemExit(run())
