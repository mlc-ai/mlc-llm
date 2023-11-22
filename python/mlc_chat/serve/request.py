"""The request class in MLC LLM serving"""
from typing import Callable, List, Union

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api
from .config import GenerationConfig
from .data import Data, TokenData


@tvm._ffi.register_object("mlc.serve.Request")  # pylint: disable=protected-access
class Request(Object):
    """The user submitted text-generation request, which contains
    a list of multi-modal inputs, a set of generation configuration
    parameters and a callback function for request finish handling.

    Parameters
    ----------
    request_id : str
        The unique identifier of the request.
        Different requests should have different ids.

    inputs : List[Data]
        The user inputs of a request. Input may have multi-modality.

    generation_config : GenerationConfig
        The sampling configuration which may contain temperature,
        top_p, repetition_penalty, max_gen_len, etc.

    fcallback : Callable[[str, TokenData, bool], None]
        The provided callback function to handle the generation
        output. It has the signature of `(str, TokenData, bool) -> None`,
        where
        - the string is the request id,
        - the TokenData contains the generated **delta** token ids since
        the invocation of the callback on the specific request,
        - the boolean value denotes if the generation of the request is
        finished.
    """

    def __init__(
        self,
        request_id: str,
        inputs: Union[Data, List[Data]],
        generation_config: GenerationConfig,
        fcallback: Callable[[str, TokenData, bool], None],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.__init_handle_by_constructor__(
            _ffi_api.Request,  # type: ignore  # pylint: disable=no-member
            request_id,
            inputs,
            generation_config.asjson(),
            fcallback,
        )

    @property
    def inputs(self) -> List[Data]:
        """The inputs of the request."""
        return _ffi_api.RequestGetInputs(self)  # type: ignore  # pylint: disable=no-member

    @property
    def generation_config(self) -> GenerationConfig:
        """The generation config of the request."""
        return GenerationConfig.from_json(
            _ffi_api.RequestGetGenerationConfigJSON(self)  # type: ignore  # pylint: disable=no-member
        )
