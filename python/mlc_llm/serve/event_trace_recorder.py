"""The event trace recorder in MLC LLM serving"""

import tvm.ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm.ffi.register_object("mlc.serve.EventTraceRecorder")  # pylint: disable=protected-access
class EventTraceRecorder(Object):
    """The event trace recorder for requests."""

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        """Initialize a trace recorder."""
        self.__init_handle_by_constructor__(
            _ffi_api.EventTraceRecorder  # type: ignore  # pylint: disable=no-member
        )

    def add_event(self, request_id: str, event: str) -> None:
        """Record a event for the input request in the trace recorder.

        Parameters
        ----------
        request_id : str
            The subject request of the event.

        event : str
            The event in a string name.
            It can have one of the following patterns:
            - "start xxx", which marks the start of event "xxx",
            - "finish xxx", which marks the finish of event "xxx",
            - "yyy", which marks the instant event "yyy".
            The "starts" and "finishes" will be automatically paired in the trace recorder.
        """
        return _ffi_api.EventTraceRecorderAddEvent(  # type: ignore  # pylint: disable=no-member
            self, request_id, event
        )

    def dump_json(self) -> str:
        """Dump the logged events in Chrome Trace Event Format in JSON string."""
        return _ffi_api.EventTraceRecorderDumpJSON(self)  # type: ignore  # pylint: disable=no-member
