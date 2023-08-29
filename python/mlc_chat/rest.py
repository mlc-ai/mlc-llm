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
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass, field, fields
from typing import Optional

from .chat_module import ChatModule, PlaceInPrompt
from .interface.openai_api import *

import numpy as np

@dataclass
class RestAPIArgs:
    """RestAPIArgs is the dataclass that organizes the arguments used for starting a REST API server."""

    model: str = field(
        metadata={
            "help": (
                """
                The model folder after compiling with MLC-LLM build process. The parameter
                can either be the model name with its quantization scheme
                (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                folder. In the former case, we will use the provided name to search
                for the model folder over possible paths.
                """
            )
        }
    )
    lib_path: str = field(
        default=None,
        metadata={
            "help": (
                """
                The full path to the model library file to use (e.g. a ``.so`` file).
                """
            )
        }
    )
    device: str = field(
        default="auto",
        metadata={
            "help": (
                """
                The description of the device to run on. User should provide a string in the
                form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
                'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
                local device), and 'device_id' is the device id to run on. If no 'device_id'
                is provided, it will be set to 0 by default.
                """
            )
        }
    )
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": (
                """
                The host at which the server should be started, defaults to ``127.0.0.1``.
                """
            )
        }
    )
    port: int = field(
        default=8000,
        metadata={
            "help": (
                """
                The port on which the server should be started, defaults to ``8000``.
                """
            )
        }
    )

def convert_args_to_argparser() -> argparse.ArgumentParser:
    """Convert from RestAPIArgs to an equivalent ArgumentParser."""
    args = argparse.ArgumentParser("MLC Chat REST API")
    for field in fields(RestAPIArgs):
        name = field.name.replace("_", "-")
        field_name = f"--{name}"
        # `kwargs` contains `help`, `choices`, and `action`
        kwargs = field.metadata.copy()
        if field.type == bool:
            # boolean arguments do not need to specify `type`
            args.add_argument(field_name, default=field.default, **kwargs)
        else:
            args.add_argument(field_name, type=field.type, default=field.default, **kwargs)
    return args


session = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_mod = ChatModule(
        model=ARGS.model,
        device=ARGS.device,
        lib_path=ARGS.lib_path
    )
    session["chat_mod"] = chat_mod

    yield

    session.clear()


app = FastAPI(lifespan=lifespan)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AsyncChatCompletionStream:
    def __aiter__(self):
        return self

    async def get_next_msg(self):
        if not session["chat_mod"]._stopped():
            session["chat_mod"]._decode()
            msg = session["chat_mod"]._get_message()
            return msg
        else:
            raise StopAsyncIteration

    async def __anext__(self):
        if not session["chat_mod"]._stopped():
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
    if len(request.messages) > 1:
            raise ValueError(
                """
                The /v1/chat/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """)

    if request.stream:

        session["chat_mod"]._prefill(input=request.messages[0].content)

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
        msg = session["chat_mod"].generate(prompt=request.messages[0].content)
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
        if len(request.prompt) > 1:
            raise ValueError(
                """
                The /v1/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """)
        prompt = request.prompt[0]
    else:
        prompt = request.prompt

    msg = session["chat_mod"].generate(prompt=prompt)

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
    inps = []
    if type(request.input) == str:
        inps.append(request.input)
    elif type(request.input) == list:
        inps = request.input
    else:
        assert f"Invalid input type {type(request.input)}"
    
    data = []
    for i, inp in enumerate(inps):
        session["chat_mod"].reset_chat()
        emb = session["chat_mod"].embed_text(input=inp).numpy()
        mean_emb = np.squeeze(np.mean(emb, axis=1), axis=0)
        norm_emb = mean_emb / np.linalg.norm(mean_emb)
        data.append({"object": "embedding", "embedding": norm_emb.tolist(), "index": i})
    # TODO: Fill in correct usage info
    return EmbeddingsResponse(
        data=data,
        usage=UsageInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )


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
    return session["chat_mod"].stats()


ARGS = convert_args_to_argparser().parse_args()
if __name__ == "__main__":
    uvicorn.run("mlc_chat.rest:app", host=ARGS.host, port=ARGS.port, reload=False, access_log=False)
