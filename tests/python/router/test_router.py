import asyncio

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.router import Router

model_tp1 = "./dist/Llama-3.2-1B-Instruct-q0f16-MLC/"
model_lib_tp1 = "./dist/lib/Llama-3.2-1B-q0f16-cuda.so"
# model_lib_tp1 = None

model_tp2 = "./dist/Llama-3.2-1B-Instruct-q0f16-MLC-tp2/"
model_lib_tp2 = "./dist/lib/Llama-3.2-1B-q0f16-cuda-tp2.so"
# model_lib_tp2 = None


def get_router_1tp1():
    return (
        Router(
            model_tp1,
            model_lib=model_lib_tp1,
            hosts=["127.0.0.1"],
            ports=[8080],
        ),
        model_tp1,
    )


def get_router_2tp1():
    return (
        Router(
            model_tp1,
            model_lib=model_lib_tp1,
            hosts=["127.0.0.1", "127.0.0.1"],
            ports=[8080, 8081],
            device_id_starts=[0, 1],
            npes=2,
        ),
        model_tp1,
    )


def get_router_1tp2():
    return (
        Router(
            model_tp2,
            model_lib=model_lib_tp2,
            hosts=["127.0.0.1"],
            ports=[8080],
            npes=2,
        ),
        model_tp2,
    )


def get_router_2tp2():
    return (
        Router(
            model_tp2,
            model_lib=model_lib_tp2,
            hosts=["127.0.0.1", "127.0.0.1"],
            ports=[8080, 8081],
            device_id_starts=[0, 2],
            npes=4,
        ),
        model_tp2,
    )


CONFIG_TO_ROUTER = {
    "1tp1": get_router_1tp1,
    "2tp1": get_router_2tp1,
    "1tp2": get_router_1tp2,
    "2tp2": get_router_2tp2,
}


async def test_router(schedule: str = "round_robin", endpoints_config: str = "1tp1"):
    router, model_id = CONFIG_TO_ROUTER[endpoints_config]()

    request = openai_api_protocol.CompletionRequest(
        prompt="The meaning of life ",
        model=model_id,
        stream=True,
        max_tokens=64,
        stream_options=openai_api_protocol.StreamOptions(include_usage=True),
    )
    if schedule == "round_robin":
        async for chunk in router._handle_completion_round_robin(request, "1"):
            print(chunk)
    elif schedule == "disagg":
        async for chunk in router._handle_completion_disagg(request, "1"):
            print(chunk)
    else:
        raise ValueError(f"Unknown scheduling method: {schedule}")
    router.terminate()


if __name__ == "__main__":
    # asyncio.run(test_router("round_robin", endpoints_config="1tp1"))
    # asyncio.run(test_router("round_robin", endpoints_config="1tp2"))
    # asyncio.run(test_router("round_robin", endpoints_config="2tp1"))
    asyncio.run(test_router("round_robin", endpoints_config="2tp2"))
