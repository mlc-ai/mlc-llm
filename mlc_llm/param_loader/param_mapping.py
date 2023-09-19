"""Parameter mapping for converting different LLM implementations to MLC LLM."""
import dataclasses
from typing import Callable, Dict, Set, Tuple

import numpy as np


@dataclasses.dataclass
class ParameterMapping:
    """Mapping from a parameter name in MLC LLM's model definition to its potential source,
    for example, from MLC parameter "model.layers.2.post_attention_layernorm.weight" to PyTorch's
    parameter correspondingly.

    Parameters
    ----------
    name_map : Dict[str, Tuple[str, ...]]
        A dictionary that maps the name of a parameter to its source. For example,
        in Llama2, the source of MLC parameter "model.layers.0.self_attn.qkv_proj.weight" from
        huggingface torch are:

        - "model.layers.0.self_attn.q_proj.weight"
        - "model.layers.0.self_attn.k_proj.weight"
        - "model.layers.0.self_attn.v_proj.weight"

    map_func: Dict[str, Callable[[np.ndarray, ...], np.ndarray]]
        A dictionary that maps the name of a parameter to a function that combines the source
        parameters into the MLC parameter. For example, for the above example, the function
        would be: `lambda q, k, v: np.concatenate([q, k, v], axis=0)`.

    unused_params : Set[str]
        Parameter names in the source weights that are not used in the MLC LLM model definition.
    """

    name_map: Dict[str, Tuple[str, ...]]
    map_func: Dict[str, Callable[[np.ndarray, ...], np.ndarray]]
    unused_params: Set[str] = dataclasses.field(default_factory=dict)
