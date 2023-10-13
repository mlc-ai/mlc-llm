from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..engine import AsyncEngineConnector
from .handler import router


def create_app(async_engine_connector: AsyncEngineConnector) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await async_engine_connector.start()
        app.state.async_engine_connector = async_engine_connector
        yield
        await async_engine_connector.stop()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
