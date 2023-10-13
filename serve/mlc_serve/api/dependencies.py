from typing import Annotated

from fastapi import Depends, FastAPI, Request

from .. import engine


def get_async_engine_connector(request: Request) -> engine.AsyncEngineConnector:
    return request.app.state.async_engine_connector


AsyncEngineConnector = Annotated[engine.AsyncEngineConnector, Depends(get_async_engine_connector)]
