"""Server context that shared by multiple entrypoint files."""

import json
from typing import Dict, List, Optional

from ...chat_module import _get_model_path
from ...conversation_template import ConvTemplateRegistry
from ...protocol.conversation_protocol import Conversation
from .. import async_engine


class ServerContext:
    """The global server context, including the running models
    and corresponding async engines.
    """

    server_context: Optional["ServerContext"] = None

    def __init__(self):
        self._models: Dict[str, async_engine.AsyncThreadedEngine] = {}
        self._conv_templates: Dict[str, Conversation] = {}
        self._model_configs: Dict[str, Dict] = {}

    def __enter__(self):
        if ServerContext.server_context is not None:
            raise RuntimeError("Server context already exists.")
        ServerContext.server_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for model_engine in self._models.values():
            model_engine.terminate()
        self._models.clear()
        self._conv_templates.clear()
        self._model_configs.clear()

    @staticmethod
    def current():
        """Returns the current ServerContext."""
        return ServerContext.server_context

    def add_model(self, hosted_model: str, engine: async_engine.AsyncThreadedEngine) -> None:
        """Add a new model to the server context together with the engine."""
        if hosted_model in self._models:
            raise RuntimeError(f"Model {hosted_model} already running.")
        self._models[hosted_model] = engine

        # Get the conversation template.
        if engine.conv_template_name is not None:
            conv_template = ConvTemplateRegistry.get_conv_template(engine.conv_template_name)
            if conv_template is not None:
                self._conv_templates[hosted_model] = conv_template

        _, config_file_path = _get_model_path(hosted_model)
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        self._model_configs[hosted_model] = config

    def get_engine(self, model: str) -> Optional[async_engine.AsyncThreadedEngine]:
        """Get the async engine of the requested model."""
        return self._models.get(model, None)

    def get_conv_template(self, model: str) -> Optional[Conversation]:
        """Get the conversation template of the requested model."""
        conv_template = self._conv_templates.get(model, None)
        if conv_template is not None:
            return conv_template.model_copy(deep=True)
        return None

    def get_model_list(self) -> List[str]:
        """Get the list of models on serve."""
        return list(self._models.keys())

    def get_model_config(self, model: str) -> Optional[Dict]:
        """Get the model config path of the requested model."""
        return self._model_configs.get(model, None)
