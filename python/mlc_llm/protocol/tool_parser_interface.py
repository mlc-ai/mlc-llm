from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Literal
import uuid
from pydantic import BaseModel, Field

class ChatFunctionCall(BaseModel):
    name: str
    arguments: Union[None, Dict[str, Any]] = None

class ChatToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: Literal["function"]
    function: Any  # Placeholder for type safety

class BaseToolParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> tuple[str, List[Any]]:
        pass

    @abstractmethod
    def parse_streaming(self, token: str) -> tuple[Optional[str], List[Any]]:
        pass

_PARSER_REGISTRY: Dict[str, type[BaseToolParser]] = {}

def register_parser(name: str):
    def decorator(cls: type[BaseToolParser]):
        _PARSER_REGISTRY[name] = cls
        return cls
    return decorator

def get_parser_instance(name: str) -> BaseToolParser:
    if name not in _PARSER_REGISTRY:
        raise ValueError(f"No parser registered for '{name}'")
    return _PARSER_REGISTRY[name]()
