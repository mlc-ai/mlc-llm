import argparse
import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import tvm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .chat_module import ChatModule, quantization_keys
from .interface.openai_api import *

session = {}


def _shared_lib_suffix():
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        return ".so"
    if sys.platform.startswith("win32"):
        return ".dll"
    if sys.platform.startswith("darwin"):
        cpu_brand_string = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8")
        if cpu_brand_string.startswith("Apple"):
            # Apple Silicon
            return ".so"
        else:
            # Intel (x86)
            return ".dylib"
    return ".so"


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_mod = ChatModule(ARGS.device_name, ARGS.device_id)
    model_path = os.path.join(ARGS.artifact_path, ARGS.model + "-" + ARGS.quantization)
    model_dir = ARGS.model + "-" + ARGS.quantization
    model_lib = model_dir + "-" + ARGS.device_name + _shared_lib_suffix()
    lib_dir = os.path.join(model_path, model_lib)
    prebuilt_lib_dir = os.path.join(ARGS.artifact_path, "prebuilt", "lib", model_lib)
    if os.path.exists(lib_dir):
        lib = tvm.runtime.load_module(lib_dir)
    elif os.path.exists(prebuilt_lib_dir):
        lib = tvm.runtime.load_module(prebuilt_lib_dir)
    else:
        raise ValueError(
            f"Unable to find {model_lib} at {lib_dir} or {prebuilt_lib_dir}."
        )

    local_model_path = os.path.join(model_path, "params")
    prebuilt_model_path = os.path.join(
        ARGS.artifact_path, "prebuilt", f"mlc-chat-{model_dir}"
    )
    if os.path.exists(local_model_path):
        chat_mod.reload(lib=lib, model_path=local_model_path)
    elif os.path.exists(prebuilt_model_path):
        chat_mod.reload(lib=lib, model_path=prebuilt_model_path)
    else:
        raise ValueError(
            f"Unable to find model params at {local_model_path} or {prebuilt_model_path}."
        )
    session["chat_mod"] = chat_mod

    yield

    session.clear()


app = FastAPI(lifespan=lifespan)


def _parse_args():
    args = argparse.ArgumentParser("MLC Chat REST API")
    args.add_argument("--model", type=str, default="vicuna-v1-7b")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--quantization",
        type=str,
        choices=quantization_keys(),
        default=quantization_keys()[0],
    )
    args.add_argument("--device-name", type=str, default="cuda")
    args.add_argument("--device-id", type=int, default=0)
    args.add_argument("--host", type=str, default="127.0.0.1")
    args.add_argument("--port", type=int, default=8000)

    parsed = args.parse_args()
    return parsed


class AsyncChatCompletionStream:
    def __aiter__(self):
        return self

    async def get_next_msg(self):
        if not session["chat_mod"].stopped():
            session["chat_mod"].decode()
            msg = session["chat_mod"].get_message()
            return msg
        else:
            raise StopAsyncIteration

    async def __anext__(self):
        if not session["chat_mod"].stopped():
            task = asyncio.create_task(self.get_next_msg())
            msg = await task
            return msg
        else:
            raise StopAsyncIteration


@app.post("/v1/chat/completions")
async def request_completion(request: ChatCompletionRequest):
    """
    Creates model response for the given chat conversation.
    """
    for message in request.messages:
        session["chat_mod"].prefill(input=message.content)
    if request.stream:

        async def iter_response():
            prev_txt = ""
            async for content in AsyncChatCompletionStream():
                if content:
                    chunk = ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role="assistant", content=content[len(prev_txt) :]
                                ),
                                finish_reason="stop",
                            )
                        ]
                    )
                    prev_txt = content
                    yield f"data: {chunk.json(exclude_unset=True)}\n\n"

        return StreamingResponse(iter_response(), media_type="text/event-stream")
    else:
        msg = None
        while not session["chat_mod"].stopped():
            session["chat_mod"].decode()
            msg = session["chat_mod"].get_message()
        return ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=msg),
                    finish_reason="stop",
                )
            ],
            # TODO: Fill in correct usage info
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


@app.post("/v1/completions")
async def request_completion(request: CompletionRequest):
    """
    Creates a completion for a given prompt.
    """
    session["chat_mod"].reset_chat()
    # Langchain's load_qa_chain.run expects the input to be a list with the query
    if isinstance(request.prompt, list):
        prompt = request.prompt[0]
    else:
        prompt = request.prompt
    session["chat_mod"].prefill(input=prompt)

    msg = None
    while not session["chat_mod"].stopped():
        session["chat_mod"].decode()
        msg = session["chat_mod"].get_message()
    return CompletionResponse(
        choices=[CompletionResponseChoice(index=0, text=msg)],
        # TODO: Fill in correct usage info
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


@app.post("/v1/embeddings")
async def request_embeddings(request: EmbeddingsRequest):
    """
    Gets embedding for some text.
    """
    assert "Endpoint not implemented."


@app.post("/chat/reset")
async def reset():
    """
    Reset the chat for the currently initialized model.
    """
    session["chat_mod"].reset_chat()


@app.get("/stats")
async def read_stats():
    """
    Get the runtime stats.
    """
    return session["chat_mod"].runtime_stats_text()


ARGS = _parse_args()
if __name__ == "__main__":
    uvicorn.run("mlc_chat.rest:app", host=ARGS.host, port=ARGS.port, reload=False, access_log=False)
