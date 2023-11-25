"""RESTful HTTP request server in MLC LLM"""
import argparse

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from .server_variables import parse_args_and_initialize

if __name__ == "__main__":
    # Parse the arguments and initialize the asynchronous engine.
    args: argparse.Namespace = parse_args_and_initialize()
    app = fastapi.FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers from subdirectories.
    from .entrypoints import openai_entrypoints

    app.include_router(openai_entrypoints.app)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
