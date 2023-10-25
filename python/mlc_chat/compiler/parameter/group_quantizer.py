from tvm.runtime import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..parameter import QuantizeMapping


class GroupQuantizer:
    quantize_map: "QuantizeMapping"

    def __init__(self, quantize_map: "QuantizeMapping") -> None:
        self.quantize_map = quantize_map

    def quantize(self, name: str, param: NDArray):
        assert name in self.quantize_map.param_map
        quantized_names = self.quantize_map.param_map[name]
        quantized_params = self.quantize_map.map_func[name](param)
        return zip(quantized_names, quantized_params)
