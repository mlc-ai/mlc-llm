"""Parameter mapping for converting different LLM implementations to MLC LLM."""

import dataclasses
from typing import Callable, Dict, List, Set, Union

import numpy as np
from tvm.runtime import NDArray

MapFuncVariadic = Union[
    Callable[[], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
]


@dataclasses.dataclass
class ExternMapping:
    """Mapping from a parameter name in MLC LLM's model definition to its potential source,
    for example, from MLC parameter "model.layers.2.post_attention_layernorm.weight" to PyTorch's
    parameter correspondingly.

    Parameters
    ----------
    param_map : Dict[str, List[str]]
        A dictionary that maps the name of a parameter to its source. For example,
        in Llama2, the source of MLC parameter "model.layers.0.self_attn.qkv_proj.weight" from
        huggingface torch are:

        - "model.layers.0.self_attn.q_proj.weight"
        - "model.layers.0.self_attn.k_proj.weight"
        - "model.layers.0.self_attn.v_proj.weight"

    map_func : Dict[str, Callable[[np.ndarray, ...], np.ndarray]]
        A dictionary that maps the name of a parameter to a function that combines the source
        parameters into the MLC parameter. For example, for the above example, the function
        would be: `lambda q, k, v: np.concatenate([q, k, v], axis=0)`.

    unused_params : Set[str]
        Parameter names in the source weights that are not used in the MLC LLM model definition.
    """

    param_map: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    map_func: Dict[str, MapFuncVariadic] = dataclasses.field(default_factory=dict)
    unused_params: Set[str] = dataclasses.field(default_factory=set)

    def add_mapping(
        self,
        map_from: str,
        map_to: List[str],
        func: MapFuncVariadic,
    ) -> None:
        """Add a mapping from MLC parameters to source parametes as well as a mapping function."""
        self.param_map[map_from] = map_to
        self.map_func[map_from] = func

    def add_unused(self, name: str):
        """Add a parameter name in the source parameters to the set of unused parameters."""
        self.unused_params.add(name)


@dataclasses.dataclass
class QuantizeMapping:
    """Mapping from a parameter in MLC LLM's model definition to its eventual names and values after
    quantization. In certain group quantization, for example, `qkv_proj.weight` is mapped to
    `qkv_proj.weight_quantized` and `qkv_proj.weight_scale` respectively. If a parameter's name is
    not in the mapping, it is assumed to be unchanged, i.e. not quantized.

    Parameters
    ----------
    param_map : Dict[str, List[str]]
        A dictionary that maps the name of a parameter to its destination. For example,
        in certain group quantization, the destinations of MLC parameter "qkv_proj.weight` are:

        - "qkv_proj.weight_quantized"
        - "qkv_proj.weight_scale"

    map_func : Dict[str, Callable[NDArray, List[NDArray]]]
        A dictionary that maps the name of a parameter to a function that splits the MLC parameter
        into the destination parameters.

    Notes
    -----
    There are two forms of weight conversion in MLC LLM, one is A) on-the-fly quantization to the
    raw fp16/bf16/fp32 weights from HuggingFace, and the other is B) loading pre-quantized weights
    from an external framework, e.g. AutoGPTQ, AutoAWQ. From the perspective of parameter
    correspondence.

    - In case A), it is recommended that the weight loader take both `ExternMapping` and
    `QuantizeMapping` as input, and do quantiaztion on the fly as a raw parameter being
    loaded into RAM;
    - In case B), a pass over `nn.Module` is recommended to take place first to converts parameters
    from its non-quantized form to the quantized one, and then only `ExternMapping` is
    used to convert the quantized parameters into the desired form.
    """

    param_map: Dict[str, List[str]]
    map_func: Dict[str, Callable[[NDArray], List[NDArray]]]


__all__ = ["ExternMapping", "QuantizeMapping"]
