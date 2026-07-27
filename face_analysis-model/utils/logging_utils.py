"""Logging setup shared by GUI, CLI and benchmark entry points.

A single call to :func:`setup_logging` configures a coloured console handler and
(optionally) a rotating file handler. Modules just do ``log = get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional

_CONFIGURED = False

_LEVEL_COLORS = {
    logging.DEBUG: "\033[38;5;245m",
    logging.INFO: "\033[38;5;39m",
    logging.WARNING: "\033[38;5;214m",
    logging.ERROR: "\033[38;5;196m",
    logging.CRITICAL: "\033[1;38;5;196m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Console formatter that colours the level name when a TTY is attached."""

    def __init__(self, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-26s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)
        color = _LEVEL_COLORS.get(record.levelno, "")
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    to_file: bool = True,
    filename: Optional[str] = None,
) -> Path | None:
    """Configure the root logger exactly once.

    Returns the log file path when file logging is enabled, else ``None``.
    """
    global _CONFIGURED
    root = logging.getLogger()
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    if _CONFIGURED:
        return None

    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(_ColorFormatter(use_color))
    console.setLevel(numeric)
    root.addHandler(console)

    log_path: Path | None = None
    if to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        name = filename or f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / name
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-30s | %(threadName)-16s | %(message)s"
            )
        )
        file_handler.setLevel(numeric)
        root.addHandler(file_handler)

    # Third-party noise reduction.
    for noisy in ("matplotlib", "PIL", "urllib3", "libav", "av"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (``setup_logging`` may be called before or after)."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
