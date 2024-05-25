"""Server context that shared by multiple entrypoint files."""

from typing import Dict, List, Optional

from ..engine import AsyncMLCEngine


class ServerContext:
    """The global server context, including the running models
    and corresponding async engines.
    """

    server_context: Optional["ServerContext"] = None
    enable_debug: bool = False

    def __init__(self) -> None:
        self._models: Dict[str, AsyncMLCEngine] = {}

    def __enter__(self):
        if ServerContext.server_context is not None:
            raise RuntimeError("Server context already exists.")
        ServerContext.server_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for model_engine in self._models.values():
            model_engine.terminate()
        self._models.clear()

    @staticmethod
    def current():
        """Returns the current ServerContext."""
        return ServerContext.server_context

    def add_model(self, hosted_model: str, engine: AsyncMLCEngine) -> None:
        """Add a new model to the server context together with the engine."""
        if hosted_model in self._models:
            raise RuntimeError(f"Model {hosted_model} already running.")
        self._models[hosted_model] = engine

    def get_engine(self, model: Optional[str]) -> Optional[AsyncMLCEngine]:
        """Get the async engine of the requested model, or the unique async engine
        if only one engine is served."""
        if len(self._models) == 1:
            return next(iter(self._models.values()))
        return self._models.get(model, None)

    def get_model_list(self) -> List[str]:
        """Get the list of models on serve."""
        return list(self._models.keys())
