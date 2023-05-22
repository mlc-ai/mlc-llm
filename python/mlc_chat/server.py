from chat_module import LLMChatModule, supported_models
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import tvm
import os

app = FastAPI()
global chat_mod
chat_mod = None

"""
List the currently supported models and provides basic information about each of them.
"""
@app.get("/models")
async def read_models():
    return {
        "data": [{
            "id": model, 
            "object":"model"
        } for model in supported_models()]
    }

"""
Check whether a model has currently been initialized.
"""
@app.get("/models/init")
async def read_init_status():
    global chat_mod
    return chat_mod is not None

"""
Retrieve a model instance with basic information about the model.
"""
@app.get("/models/{model}")
async def read_model(model: str):
    if model not in supported_models():
        raise HTTPException(status_code=404, detail=f"Model {model} is not supported.")
    return {
        "id": model, 
        "object":"model"
    }

class ModelRequest(BaseModel):
    model: str
    quantization: str = "q3f16_0"
    mlc_path: str
    artifact_path: str = "dist"
    device_name: str = "cuda"
    device_id: int = 0

"""
Load and initialize a model from a specified path.
"""
@app.post("/models/init")
async def initialize_model(request: ModelRequest):
    if request.model not in supported_models():
        raise HTTPException(status_code=404, detail=f"Model {request.model} is not supported.")
    mlc_lib_path = os.path.join(request.mlc_path, "build/libmlc_llm_module.so")
    model_path = os.path.join(
        request.artifact_path, 
        request.model + "-" + request.quantization
    )
    global chat_mod
    chat_mod = LLMChatModule(
        mlc_lib_path,
        request.device_name, 
        request.device_id
    )
    model_dir = request.model + "-" + request.quantization
    model_lib = model_dir + "-" + request.device_name + ".so"
    lib = tvm.runtime.load_module(os.path.join(model_path, model_lib))
    chat_mod.reload(lib=lib, model_path=os.path.join(model_path, "params"))

class ChatRequest(BaseModel):
    prompt: str

"""
Creates model response for the given chat conversation.
"""
@app.post("/chat/completions")
def request_completion(request: ChatRequest):
    global chat_mod
    if not chat_mod:
        raise HTTPException(status_code=404, detail=f"A model has not been initialized. Please initialize a model using models/init")
    chat_mod.prefill(input=request.prompt)
    msg = None
    while not chat_mod.stopped():
        chat_mod.decode()
        msg = chat_mod.get_message()
    return {"message": msg}