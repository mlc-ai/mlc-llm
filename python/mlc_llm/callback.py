"""Namespace of callback functions in Python API."""
# pylint: disable=unused-import, invalid-name, unnecessary-pass
from queue import Queue
from typing import Optional


def _get_delta_message(curr_message: str, new_message: str) -> str:
    r"""Given the current message and the new message, compute the delta message
    (the newly generated part, the diff of the new message from the current message).

    Parameters
    ----------
    curr_message : str
        The message generated in the previous round.
    new_message : str
        The message generated in the new round.

    Returns
    -------
    delta_message : str
        The diff of the new message from the current message (the newly generated part).
    """
    from tvm._ffi import get_global_func  # pylint: disable=import-outside-toplevel

    f_get_delta_message = get_global_func("mlc.get_delta_message")
    return f_get_delta_message(curr_message, new_message)


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
            delta = _get_delta_message(self.curr_message, message)
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


class StreamIterator(DeltaCallback):
    """Stream the output using an iterator.
    A queue stores the delta messages"""

    def __init__(self, callback_interval: int = 2, timeout: Optional[float] = None):
        r"""Initialize the callback class with callback interval and queue timeout.

        Parameters
        ----------
        callback_interval : int
            The refresh rate of the streaming process.
        timeout : Optional[float]
            Timeout for put and get from the delta messages queue
        """
        super().__init__()
        self.delta_messages: Queue = Queue()
        self.callback_interval = callback_interval
        self.timeout = timeout

    def delta_callback(self, delta_message: str):
        r"""Stream the delta message to iterator (adding).

        Parameters
        ----------
        delta_message : str
            The delta message (the part that has not been added to queue yet).
        """
        self.delta_messages.put(delta_message, timeout=self.timeout)

    def stopped_callback(self):
        """Using None as the stop signal for the iterator"""
        self.delta_messages.put(None, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.delta_messages.get(timeout=self.timeout)
        if value:
            return value
        raise StopIteration()
