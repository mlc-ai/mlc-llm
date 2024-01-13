"""Benchmark latency offline."""
import argparse
import cuda
import cuda.cudart
import time, numpy as np
from mlc_serve.engine import (
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.utils import (
    get_default_mlc_serve_argparser,
    postproc_mlc_serve_args,
    create_mlc_engine,
)
from utils import add_sampling_flags, postproc_sampling_args


request_counter = 0


def create_request(request_id):
    global request_counter
    request_counter += 1
    return Request(
        request_id=str(request_counter),
        messages=None,  # Provide prompt as `DebugOption` to bypass the conv template
        sampling_params=SamplingParams(
            temperature=args.temperature,
            top_p=(1 if args.temperature == 0.0 else args.sampling_setting["top_p"]),
            top_k=(-1 if args.temperature == 0.0 else args.sampling_setting["top_k"]),
            repetition_penalty=args.sampling_setting["repetition_penalty"],
            frequency_penalty=args.sampling_setting["frequency_penalty"],
            presence_penalty=args.sampling_setting["presence_penalty"],
            logit_bias=args.sampling_setting["logit_bias"],
        ),
        stopping_criteria=StoppingCriteria(
            max_tokens=args.num_output_tokens, stop_sequences=None
        ),
        debug_options=DebugOptions(
            ignore_eos=True, prompt_token_ids=[3] * args.num_input_tokens
        ),
        num_sequences=args.num_sequences_to_sample,
    )


def main(args: argparse.Namespace):
    print(args)

    engine = create_mlc_engine(args)

    # warm up
    engine.add([create_request(args)])

    while engine.has_pending_requests():
        engine.step()

    latencies = []
    engine.add([create_request(args)])

    cuda.cudart.cudaProfilerStart()
    while engine.has_pending_requests():
        t0 = time.perf_counter()
        engine.step()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
    cuda.cudart.cudaProfilerStop()

    if args.use_staging_engine:
        engine.stop()

    assert len(latencies) == args.num_output_tokens
    ttft = latencies[0]  # time to first token
    itl = np.mean(latencies[1:])  # inter-token latency for subsequent tokens
    e2e = np.sum(latencies)

    print(
        f"User side metrics\n"
        f"* number of input tokens: {args.num_input_tokens}, number of output tokens: {args.num_output_tokens}\n"
        f"* Time To First Token (TTFT): {ttft*1000:.3f} ms\n"
        f"* Inter-Subsequent-Token-Latency (ISTL): {itl*1000:.3f} ms ({1/itl:.3f} tok/s)\n"
        f"* End-to-end latency: {e2e:.3f} s\n"
    )


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser(description="Benchmark the throughput.")
    parser.add_argument("--num-input-tokens", type=int, default=128)
    parser.add_argument("--num-output-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temparature. By default, random sampling has used.",
    )
    add_sampling_flags(parser)
    args = parser.parse_args()
    postproc_mlc_serve_args(args)
    postproc_sampling_args(args)

    main(args)
