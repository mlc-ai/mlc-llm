# pylint: disable=missing-docstring,fixme,import-error
import argparse
import ast
import asyncio
import dataclasses
import json
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from mlc_chat.chat_module import GenerationConfig
from mlc_chat.support.random import set_global_random_seed

from .chat_module import ChatModule
from .interface.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ToolCalls,
    ToolChoice,
    UsageInfo,
    VisualStudioCodeCompletionRequest,
    VisualStudioCodeCompletionResponse,
)


@dataclasses.dataclass
class RestAPIArgs:
    """RestAPIArgs is the dataclass that organizes the arguments used for starting a REST API
    server."""

    model: str = dataclasses.field(
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
    lib_path: str = dataclasses.field(
        default=None,
        metadata={
            "help": (
                """
                The full path to the model library file to use (e.g. a ``.so`` file).
                """
            )
        },
    )
    device: str = dataclasses.field(
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
        },
    )
    host: str = dataclasses.field(
        default="127.0.0.1",
        metadata={
            "help": (
                """
                The host at which the server should be started, defaults to ``127.0.0.1``.
                """
            )
        },
    )
    port: int = dataclasses.field(
        default=8000,
        metadata={
            "help": (
                """
                The port on which the server should be started, defaults to ``8000``.
                """
            )
        },
    )
    random_seed: int = dataclasses.field(
        default=None,
        metadata={
            "help": (
                """
                The random seed to initialize all the RNG used in mlc-chat. By default,
                no seed is set.
                """
            )
        },
    )


def convert_args_to_argparser() -> argparse.ArgumentParser:
    """Convert from RestAPIArgs to an equivalent ArgumentParser."""
    args = argparse.ArgumentParser("MLC Chat REST API")
    for field in dataclasses.fields(RestAPIArgs):
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


session: Dict[str, ChatModule] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if ARGS.random_seed is not None:
        set_global_random_seed(ARGS.random_seed)
    chat_mod = ChatModule(
        model=ARGS.model,
        device=ARGS.device,
        model_lib_path=ARGS.lib_path,
    )
    session["chat_mod"] = chat_mod
    yield
    session.clear()


origins = ["*"]

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AsyncCompletionStream:
    def __init__(self, generation_config: GenerationConfig):
        self.generation_config = generation_config

    def __aiter__(self):
        return self

    async def get_next_msg(self):
        # pylint: disable=protected-access
        if not session["chat_mod"]._stopped():
            session["chat_mod"]._decode(generation_config=self.generation_config)
            msg = session["chat_mod"]._get_message()
            return msg
        # pylint: enable=protected-access
        raise StopAsyncIteration

    async def __anext__(self):
        if not session["chat_mod"]._stopped():
            task = asyncio.create_task(self.get_next_msg())
            msg = await task
            return msg
        raise StopAsyncIteration


def add_function_call(prompt: List[ChatMessage], function_string: str):
    # update content of the last input message to include function string
    user_query = prompt[-1].content
    prompt[-1].content = f"<<question>> {user_query} <<function>> {function_string}\n"


def function_call_util(request: ChatCompletionRequest):
    """Performs the necessary actions to add function calls to the prompt
    returns True if function calls are added to the prompt else returns False
    TODO: Check function name in tools.function['name']
    TODO: Currently auto mode default to generating function calls instead of smartly
    checking weather to generate function calls or not
    """

    # return if no tools are provided
    if request.tools is None:
        return False

    # skip if tool_choice is set to none
    if isinstance(request.tool_choice, str) and request.tool_choice == "none":
        return False

    if isinstance(request.tool_choice, ToolChoice):
        # force the model to use a specific function provided by tool_choice
        if request.tool_choice.type != "function":
            raise ValueError("Only 'function' tool choice is supported")
        for tool in request.tools:
            if tool.function["name"] == request.tool_choice.function["name"]:
                add_function_call(request.messages, json.dumps(tool.function))
                return True
        raise ValueError("ToolChoice.function.name not found in tools")

    if isinstance(request.tool_choice, str):
        # Add all the functions to the input prompt
        function_list = []
        for tool in request.tools:
            if tool.type == "function":
                function_list.append(tool.function)
            else:
                raise ValueError("Only 'function' tool.type is supported")
        add_function_call(request.messages, json.dumps(function_list))
    else:
        raise ValueError("Invalid toolChoice instance type")
    return True


def convert_function_str_to_json(stringified_calls):
    def parse_function_call(call_str):
        node = ast.parse(call_str, mode="eval")
        call_node = node.body
        if isinstance(call_node, ast.Call):
            name = call_node.func.id
            arguments = {}
            for keyword in call_node.keywords:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            return {"name": name, "arguments": arguments}
        return None

    calls = ast.literal_eval(stringified_calls)
    result = [parse_function_call(call_str) for call_str in calls]
    return result


@app.post("/v1/chat/completions")
async def request_chat_completion(request: ChatCompletionRequest):
    """
    Creates model response for the given chat conversation.
    The messages field contains a list of messages (describing the conversation history). eg:
    ```"messages": [{"role": "user", "content": "What's my name?"},
                    {"role": "assistant", "content": "Your name is Llama."},
                    {"role": "user", "content": "No, that's your name. My name is X."},
                    {"role": "assistant", "content": "Ah, my apologies! Your name is X! "},
                    {"role": "user", "content": "What is the meaning of life?"},
                ]
    ```
    ]
    """
    generation_config = GenerationConfig(
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        top_p=request.top_p,
        mean_gen_len=request.mean_gen_len,
        max_gen_len=request.max_gen_len,
        n=request.n,
        stop=request.stop,
    )

    session["chat_mod"].reset_chat()  # Reset previous history, KV cache, etc.

    use_function_call = function_call_util(request)

    if request.stream:
        session["chat_mod"]._prefill(  # pylint: disable=protected-access
            input=request.messages,
            generation_config=generation_config,
        )

        async def iter_response():
            prev_txt = ""
            async for content in AsyncCompletionStream(generation_config=generation_config):
                if content:
                    # Remove the replacement character (U+FFFD) from the response
                    # This is to handle emojis. An emoji might be made up of multiple tokens.
                    # In the Rest streaming setting, if an emoji gets truncated in the middle of
                    # its encoded byte sequence, a replacement character will appear.
                    valid_content = content.replace("�", "")
                    chunk = ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role="assistant", content=valid_content[len(prev_txt) :]
                                ),
                                finish_reason="stop",
                            )
                        ]
                    )
                    prev_txt = valid_content
                    yield f"data: {chunk.json(exclude_unset=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(iter_response(), media_type="text/event-stream")
    msg = session["chat_mod"].generate(
        prompt=request.messages, generation_config=generation_config, stateless=True
    )
    if isinstance(msg, str):
        msg = [msg]

    choices = []
    for index, msg_i in enumerate(msg):
        if use_function_call:
            choices.append(
                ChatCompletionResponseChoice(
                    index=index,
                    message=ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCalls(
                                function=fn_json_obj,
                            )
                            for fn_json_obj in convert_function_str_to_json(msg_i)
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            )
        else:
            choices.append(
                ChatCompletionResponseChoice(
                    index=index,
                    message=ChatMessage(
                        role="assistant",
                        content=msg_i,
                    ),
                    finish_reason="stop",
                )
            )

    return ChatCompletionResponse(
        choices=choices,
        # TODO: Fill in correct usage info
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


@app.post("/v1/completions")
async def request_completion(request: CompletionRequest):
    """
    Creates a completion for a given prompt.
    """

    generation_config = GenerationConfig(
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        top_p=request.top_p,
        mean_gen_len=request.mean_gen_len,
        max_gen_len=request.max_gen_len,
        n=request.n,
        stop=request.stop,
    )

    session["chat_mod"].reset_chat()
    # Langchain's load_qa_chain.run expects the input to be a list with the query
    if isinstance(request.prompt, list):
        if len(request.prompt) > 1:
            raise ValueError(
                """
                The /v1/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """
            )
        prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if request.stream:
        session["chat_mod"]._prefill(  # pylint: disable=protected-access
            input=prompt,
            generation_config=generation_config,
        )

        async def iter_response():
            prev_txt = ""
            async for content in AsyncCompletionStream(generation_config=generation_config):
                if content:
                    chunk = CompletionStreamResponse(
                        choices=[
                            CompletionResponseStreamChoice(
                                index=0,
                                text=content[len(prev_txt) :],
                                finish_reason="stop",
                            )
                        ]
                    )
                    prev_txt = content
                    yield f"data: {chunk.json(exclude_unset=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(iter_response(), media_type="text/event-stream")
    msg = session["chat_mod"].generate(prompt=prompt, generation_config=generation_config)
    if isinstance(msg, str):
        msg = [msg]
    return CompletionResponse(
        choices=[
            CompletionResponseChoice(index=index, text=msg[index]) for index in range(len(msg))
        ],
        # TODO: Fill in correct usage info
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


@app.post("/v1/embeddings")
async def request_embeddings(request: EmbeddingsRequest):
    """
    Gets embedding for some text.
    """
    inps = []
    if isinstance(request.input, str):
        inps.append(request.input)
    elif isinstance(request.input, list):
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
        data=data, usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
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


@app.get("/verbose_stats")
async def read_stats_verbose():
    """
    Get the verbose runtime stats.
    """
    return session["chat_mod"].stats(verbose=True)


@app.post("/v1/llm-vscode/completions")
async def request_llm_vscode(request: VisualStudioCodeCompletionRequest):
    """
    Creates a vscode code completion for a given prompt.
    Follows huggingface LSP (https://github.com/huggingface/llm-ls)
    """
    generation_config = GenerationConfig(
        temperature=request.parameters.temperature,
        top_p=request.parameters.top_p,
        mean_gen_len=request.parameters.max_new_tokens,
        max_gen_len=request.parameters.max_new_tokens,
    )
    msg = session["chat_mod"].generate(prompt=request.inputs, generation_config=generation_config)

    return VisualStudioCodeCompletionResponse(generated_text=msg)


ARGS = convert_args_to_argparser().parse_args()
if __name__ == "__main__":
    uvicorn.run("mlc_chat.rest:app", host=ARGS.host, port=ARGS.port, reload=False, access_log=False)
