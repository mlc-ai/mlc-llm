"""The request class in MLC LLM serving"""

from typing import List

import tvm.ffi
from tvm.runtime import Object

from mlc_llm.protocol.generation_config import GenerationConfig

from . import _ffi_api
from .data import Data


@tvm.ffi.register_object("mlc.serve.Request")  # pylint: disable=protected-access
class Request(Object):
    """The user submitted text-generation request, which contains
    a unique request id, a list of multi-modal inputs, a set of
    generation configuration parameters.

    Note
    ----
    Do not explicitly construct this class.
    Construct this object via engine.create_request functions.
    """

    @property
    def inputs(self) -> List[Data]:
        """The inputs of the request."""
        return _ffi_api.RequestGetInputs(self)  # type: ignore  # pylint: disable=no-member

    @property
    def generation_config(self) -> GenerationConfig:
        """The generation config of the request."""
        return GenerationConfig.model_validate_json(
            _ffi_api.RequestGetGenerationConfigJSON(self)  # type: ignore  # pylint: disable=no-member
        )
