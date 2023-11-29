"""Server context that shared by multiple entrypoint files."""
from typing import Dict, List, Optional, Union

from . import async_engine

EngineClass = Union[async_engine.AsyncEngine, async_engine.AsyncThreadedEngine]


class ServerContext:
    """The global server context, including the running models
    and corresponding async engines.
    """

    _models: Dict[str, EngineClass] = {}

    @staticmethod
    def add_model(hosted_model: str, engine: EngineClass) -> None:
        """Add a new model to the server context together with the engine."""
        if hosted_model in ServerContext._models:
            raise RuntimeError(f"Model {hosted_model} already running.")
        ServerContext._models[hosted_model] = engine

    @staticmethod
    def get_engine(model: str) -> Optional[EngineClass]:
        """Get the async engine of the requested model."""
        return ServerContext._models.get(model, None)

    @staticmethod
    def get_model_list() -> List[str]:
        """Get the list of models on serve."""
        return list(ServerContext._models.keys())
