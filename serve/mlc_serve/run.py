import uvicorn

from .api import APIConfig, create_app
from .engine import AsyncEngineConnector
from .engine.dummy import DummyInferenceEngine


def run_server():
    engine = DummyInferenceEngine()
    connector = AsyncEngineConnector(engine)
    app = create_app(APIConfig(), connector)
    uvicorn.run(
        app,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    run_server()
