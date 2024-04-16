"""Printing styles."""

from enum import Enum


class Styles(Enum):
    """Predefined set of styles to be used.

    Reference:
    - https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
    - https://stackoverflow.com/a/17303428
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def red(text: str) -> str:
    """Return red text."""
    return f"{Styles.RED.value}{text}{Styles.END.value}"


def green(text: str) -> str:
    """Return green text."""
    return f"{Styles.GREEN.value}{text}{Styles.END.value}"


def yellow(text: str) -> str:
    """Return yellow text."""
    return f"{Styles.YELLOW.value}{text}{Styles.END.value}"


def blue(text: str) -> str:
    """Return blue text."""
    return f"{Styles.BLUE.value}{text}{Styles.END.value}"


def purple(text: str) -> str:
    """Return purple text."""
    return f"{Styles.PURPLE.value}{text}{Styles.END.value}"


def cyan(text: str) -> str:
    """Return cyan text."""
    return f"{Styles.CYAN.value}{text}{Styles.END.value}"


def bold(text: str) -> str:
    """Return bold text."""
    return f"{Styles.BOLD.value}{text}{Styles.END.value}"


def underline(text: str) -> str:
    """Return underlined text."""
    return f"{Styles.UNDERLINE.value}{text}{Styles.END.value}"
