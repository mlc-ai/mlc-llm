"""An enhanced argument parser for mlc-chat."""

import argparse
import sys


class ArgumentParser(argparse.ArgumentParser):
    """An enhanced argument parser for mlc-chat."""

    def error(self, message):
        """Overrides the behavior when erroring out"""
        print("-" * 25 + " Usage " + "-" * 25)
        self.print_help()
        print("-" * 25 + " Error " + "-" * 25)
        print(message, file=sys.stderr)
        sys.exit(2)
