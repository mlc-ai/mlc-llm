"""An enhanced argument parser for mlc-chat."""

import argparse
import sys

TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("0", "false", "no", "off")


def boolean(value: str) -> bool:
    """Parse a boolean value taken from the command line.

    `type=bool` cannot be used for this: argparse applies the callable to the raw string,
    and `bool("false")` is `True`, so every non-empty value turns the flag on and the flag
    can never be switched off.
    """
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Expected one of {', '.join(TRUE_VALUES + FALSE_VALUES)}."
    )


class ArgumentParser(argparse.ArgumentParser):
    """An enhanced argument parser for mlc-chat."""

    def error(self, message):
        """Overrides the behavior when erroring out"""
        print("-" * 25 + " Usage " + "-" * 25)
        self.print_help()
        print("-" * 25 + " Error " + "-" * 25)
        print(message, file=sys.stderr)
        sys.exit(2)
