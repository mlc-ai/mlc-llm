# pylint: disable=missing-module-docstring,missing-function-docstring
import json

import pytest

from mlc_llm.serve.event_trace_recorder import EventTraceRecorder

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def test_event_trace_recorder():
    trace_recorder = EventTraceRecorder()
    request_ids = ["x", "y"]
    num_decode = 5

    for request_id in request_ids:
        trace_recorder.add_event(request_id, event="start tokenization")
        trace_recorder.add_event(request_id, event="finish tokenization")
        trace_recorder.add_event(request_id, event="add request")
        trace_recorder.add_event(request_id, event="start embed")
        trace_recorder.add_event(request_id, event="finish embed")
        trace_recorder.add_event(request_id, event="start prefill")
        trace_recorder.add_event(request_id, event="finish prefill")

    for _ in range(num_decode):
        for request_id in request_ids:
            trace_recorder.add_event(request_id, event="start decode")
            trace_recorder.add_event(request_id, event="finish decode")
    for request_id in request_ids:
        trace_recorder.add_event(request_id, event="start detokenization")
        trace_recorder.add_event(request_id, event="finish detokenization")

    events = json.loads(trace_recorder.dump_json())
    decode_count = {}
    for event in events:
        request_id = event["tid"]
        if event["name"].startswith("decode"):
            if request_id not in decode_count:
                decode_count[request_id] = 1
            else:
                decode_count[request_id] += 1

    for _, decode_cnt in decode_count.items():
        assert decode_cnt == num_decode * 2, decode_cnt


if __name__ == "__main__":
    test_event_trace_recorder()
