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

    def __call__(self, message: str):
        pass
