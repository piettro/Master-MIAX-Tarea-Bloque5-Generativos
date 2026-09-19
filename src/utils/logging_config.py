"""Root logging configuration for the project.

`configure_logging` is called once, from `main.py`, before any other project
module does meaningful work. Every module then obtains its own logger with
`logging.getLogger(__name__)`, which inherits the handlers and formatting
configured here.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a single, human-readable handler.

    Args:
        level: Minimum severity that will be emitted. Defaults to
            ``logging.INFO``, which is verbose enough to follow training
            progress without flooding the console with debug detail.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if this is called more than once (for
    # instance, from a test suite that imports main multiple times).
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
