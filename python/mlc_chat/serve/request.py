"""The request class in MLC LLM serving"""
from typing import List, Optional, Tuple, Union

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api
from .config import GenerationConfig
from .data import Data, TokenData


@tvm._ffi.register_object("mlc.serve.Request")  # pylint: disable=protected-access
class Request(Object):
    """The user submitted text-generation request, which contains
    a unique request id, a list of multi-modal inputs, a set of
    generation configuration parameters.

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
    """

    def __init__(
        self,
        request_id: str,
        inputs: Union[Data, List[Data]],
        generation_config: GenerationConfig,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.__init_handle_by_constructor__(
            _ffi_api.Request,  # type: ignore  # pylint: disable=no-member
            request_id,
            inputs,
            generation_config.asjson(),
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


@tvm._ffi.register_object("mlc.serve.RequestStreamOutput")  # pylint: disable=protected-access
class RequestStreamOutput(Object):
    """The generated delta request output that is streamed back
    through callback stream function.
    It contains three fields (in order):

    request_id : str
        The id of the request that the function is invoked for.

    delta_tokens : data.TokenData
        The new generated tokens since the last callback invocation
        for the input request.

    finish_reason : Optional[str]
        The finish reason of the request when it is finished,
        of None if the request has not finished yet.

    Note
    ----
    We do not provide constructor, since in practice only C++ side
    instantiates this class.
    """

    def unpack(self) -> Tuple[str, TokenData, Optional[str]]:
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        delta_tokens : data.TokenData
            The new generated tokens since the last callback invocation
            for the input request.

        finish_reason : Optional[str]
            The finish reason of the request when it is finished,
            of None if the request has not finished yet.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)  # type: ignore  # pylint: disable=no-member
        return str(fields[0]), fields[1], str(fields[2]) if fields[2] is not None else None
