"""Namespace of callback functions in Python API."""
#! pylint: disable=unused-import, invalid-name, unnecessary-pass

from .base import get_delta_message


class DeltaCallback:
    """Base class that fetches delta callback"""

    def __init__(self):
        r"""Initialize the callback class."""
        self.curr_message = ""

    def __call__(self, message: str = "", stopped: bool = False):
        r"""Process newly generated message using callback functions.

        Parameters
        ----------
        message : str
            The newly generated message.
        stopped : bool
            Whether generation reaches an end. If True, clear the state of current message.
        """
        if stopped:
            self.stopped_callback()
            self.curr_message = ""
        else:
            delta = get_delta_message(self.curr_message, message)
            self.curr_message = message
            self.delta_callback(delta)

    def delta_callback(self, delta_message: str):
        r"""Perform a callback action on the delta message.
        This vary depending on the callback method.

        Parameters
        ----------
        delta_message : str
            The delta message.
        """
        raise NotImplementedError

    def stopped_callback(self):
        r"""Perform a callback action when we receive a "stop generating" signal.
        Can optionally ignore this function if no action need to be done when
        generation stops."""
        pass


class StreamToStdout(DeltaCallback):
    """Stream the output of the chat module to stdout."""

    def __init__(self, callback_interval: int = 2):
        r"""Initialize the callback class with callback interval.

        Parameters
        ----------
        callback_interval : int
            The refresh rate of the streaming process.
        """
        super().__init__()
        self.callback_interval = callback_interval

    def delta_callback(self, delta_message: str):
        r"""Stream the delta message directly to stdout.

        Parameters
        ----------
        delta_message : str
            The delta message (the part that has not been streamed to stdout yet).
        """
        print(delta_message, end="", flush=True)

    def stopped_callback(self):
        r"""Stream an additional '\n' when generation ends."""
        print()
