from types import SimpleNamespace

import pytest

from mlc_llm.serve.engine_base import AsyncRequestStream, EngineState

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def test_engine_state_is_instance_local():
    first = EngineState(enable_tracing=True)
    second = EngineState(enable_tracing=False)
    request_id = "same-request-id"

    first.async_streamers[request_id] = (AsyncRequestStream(), [])
    second.async_streamers[request_id] = (AsyncRequestStream(), [])
    assert first.trace_recorder is not None
    assert second.trace_recorder is None
    assert first.async_event_loop is None
    assert second.async_event_loop is None
    assert first.async_streamers is not second.async_streamers
    assert first.sync_output_queue is not second.sync_output_queue
    assert first.sync_text_streamers is not second.sync_text_streamers
    assert second.sync_text_streamers == []

    class FinalOutput:
        @staticmethod
        def unpack():
            return request_id, [SimpleNamespace(request_final_usage_json_str="{}")]

    first._async_request_stream_callback_impl([FinalOutput()])

    assert request_id not in first.async_streamers
    assert request_id in second.async_streamers


if __name__ == "__main__":
    test_engine_state_is_instance_local()
