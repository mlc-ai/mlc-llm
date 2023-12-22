from typing import Any

class TextGenerationError(Exception):
    def __init__(self, error: Any) -> None:
        self.error = error
        super().__init__(error)

class EngineException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(f"InferenceEngine: {msg}")
