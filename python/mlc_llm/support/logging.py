"""
Logging support for MLC. It derives from Python's logging module, and in the future,
it can be easily replaced by other logging modules such as structlog.
"""

import logging
import os


def enable_logging():
    """Enable MLC's default logging format"""
    if os.getenv("MLC_UNSET_LOGGING"):
        return
    logging.basicConfig(
        level=logging.INFO,
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
    )


def getLogger(name: str):  # pylint: disable=invalid-name
    """Get a logger according to the given name"""
    return logging.getLogger(name)
