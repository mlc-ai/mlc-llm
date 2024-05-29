"""Low-level generation config class"""

# pylint: disable=missing-class-docstring, disable=too-many-instance-attributes
from typing import Dict, List, Optional

from pydantic import BaseModel

from .debug_protocol import DebugConfig
from .openai_api_protocol import RequestResponseFormat


class GenerationConfig(BaseModel):  # pylint:
    """The generation configuration dataclass.

    This is a config class used by Engine internally.
    """

    n: int = 1
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = None
    # internally we use -1 to represent infinite
    max_tokens: int = -1
    seed: Optional[int] = None
    stop_strs: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    response_format: Optional[RequestResponseFormat] = None
    debug_config: Optional[Optional[DebugConfig]] = None
