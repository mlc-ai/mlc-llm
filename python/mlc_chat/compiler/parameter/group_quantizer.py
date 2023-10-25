"""A group quantizer for on the fly parameter quantization"""
# pylint: disable=too-few-public-methods
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.runtime import NDArray

    from ..parameter import QuantizeMapping


class GroupQuantizer:
    """A group quantizer that quantizes given mlc-llm parameters"""

    quantize_map: "QuantizeMapping"

    def __init__(self, quantize_map: "QuantizeMapping") -> None:
        self.quantize_map = quantize_map

    def quantize(self, name: str, param: "NDArray"):
        """Apply group quantization to the given paramete

        Parameters
        ----------
        name : str
            The name of the parameter
        param : NDArray
            The parameter to be quantized

        Returns
        -------
        List[Tuple[str, NDArray]]
            The quantized parameters, each with its name
        """

        assert name in self.quantize_map.param_map
        quantized_names = self.quantize_map.param_map[name]
        quantized_params = self.quantize_map.map_func[name](param)
        return zip(quantized_names, quantized_params)
