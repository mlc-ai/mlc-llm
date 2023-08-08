"""Namespace of callback functions in Python API."""
#! pylint: disable=unused-import, invalid-name


class stream_to_stdout(object):
    """Stream the output of the chat module to stdout."""

    def __init__(self, interval: int = 2):
        r"""Initialize the callback class.

        Parameters
        ----------
        interval : int
            The refresh rate of the streaming process.
        """
        self.interval = interval

    def __call__(self, message: str = "", stopped: bool = False):
        r"""Stream the message to stdout without any buffering.

        Parameters
        ----------
        message : str
            The new piece of message that is not streamed to stdout yet.
        stopped : bool
            Whether streaming reaches the end. If so, print out an additional "\n".
        """
        if stopped:
            print()
        else:
            print(message, end="", flush=True)


class update_var(object):

    def __init__(self):
        r"""Initialize the callback class.
        """
        self.var = ""
        self.interval = 0

    def __call__(self, message: str = "", stopped: bool = False):
        r"""Update `var` with the delta message.

        Parameters
        ----------
        message : str
            The new piece of message that has not been added to `var` yet.
        stopped : bool
            Whether streaming reaches the end.
        """
        if not stopped:
            self.var += message
