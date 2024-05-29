"""Utils to better use tqdm"""

import contextlib
import inspect
import io

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm as _redirect_logging


@contextlib.contextmanager
def _redirect_print():
    old_print = print

    def new_print(*args, **kwargs):
        with io.StringIO() as output:
            kwargs["file"] = output
            kwargs["end"] = ""
            old_print(*args, **kwargs)
            content = output.getvalue()
        tqdm.write(content)

    try:
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


@contextlib.contextmanager
def redirect():
    """Redirect tqdm output to logging and print."""

    with _redirect_logging():
        with _redirect_print():
            yield


__all__ = ["tqdm", "redirect"]
