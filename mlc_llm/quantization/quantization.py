import enum
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Type, Union

import tvm
from tvm import relax, te
from tvm.relax.expr_functor import PyExprVisitor, visitor

FQuantize = Callable[[relax.BlockBuilder, List[relax.Expr]], relax.Var]
FTEQuantize = Callable[[te.Tensor], List[te.Tensor]]
FTEDequantize = Callable[[List[te.Tensor]], te.Tensor]


@dataclass
class QuantizationSpec:
    """The base dataclass of quantization specification.
    A specification describes how a parameter is quantized and dequantized.

    A subclass of QuantizationSpec
      - contains more data fields (e.g., the "group size" in group quantization)
      which instruct the quantization/dequantization,
      - defines the `get_quantize_func` method, which returns a function
      (`Callable[[relax.BlockBuilder, List[relax.Expr]], relax.Var]`) that takes a
      Relax BlockBuilder and the weight relax Var to be quantized, computes
      the quantization and returns the relax Var of quantized results.
      algorithm of the quantization.
      - defines the `get_dequantize_func` method, which returns function
      (`Callable[[relax.BlockBuilder, List[relax.Expr]], relax.Var]`) that takes
      the quantized results, computes and returns the dequantization result.
      - optionally overloads the `get_loaded_tensor_info` when the parameter is
      pre-quantized, in which case `get_loaded_tensor_info` needs to be overloaded
      so that we know how many quantized data tensors there are, and the dtype
      and shape of each quantized data tensor.
    """

    dtype: str

    def get_loaded_tensor_info(
        self, param_info: relax.TensorStructInfo
    ) -> List[relax.TensorStructInfo]:
        """Returns the shape and dtype of the tensors that need to be loaded
        from the disk.

        It is useful when the parameter is pre-quantized. In such cases, we need
        to know how many tensors the parameter is quantized into, and together
        with the dtype and shape of each tensor, so that we can load the
        pre-quantized tensors in.
        """
        return [param_info]

    def get_quantize_func(self, param_info: relax.TensorStructInfo) -> Optional[FQuantize]:
        """Returns the function which computes quantization.
        Returning `None` means the parameter does not need quantization or is
        pre-quantized.

        The returned function takes a Relax BlockBuilder and a (list of) weight
        relax Var to be quantized, computes the quantization and returns the
        quantization result Relax Var(s).

        You can use `convert_TE_func` to convert a TE function to the function
        of the desired return format. See `group_quantization.py` for examples.
        """
        return NotImplementedError()

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FQuantize]:
        """Returns the function which computes dequantization.
        Returning `None` means the parameter does not need dequantization.

        The returned function takes a Relax BlockBuilder and a (list of)
        quantized weight relax Var, computes the dequantization and returns the
        result Relax Var(s).

        You can use `convert_TE_func` to convert a TE function to the function
        of the desired return format. See `group_quantization.py` for examples.
        """
        return NotImplementedError()


@dataclass
class NoQuantizationSpec(QuantizationSpec):
    """The quantization specification that describes doing no quantization."""

    def get_quantize_func(self, param_info: relax.TensorStructInfo) -> Optional[FQuantize]:
        return None

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FQuantize]:
        return None


class ParamQuantKind(enum.IntEnum):
    """The parameter quantization kind class.

    We categorized all the parameters in a model into four kinds:
    - the weights of the internal linear layers, which are the main targets of quantization,
    - the embedding table of every token,
    - the weight of the fully-connected layer at the end of the model, which is
    used for computes the logits of each input token,
    - other parameters (e.g., the weight of layer normalization, etc.).
    """

    linear_weight = 0
    embedding_table = 1
    final_fc_weight = 2
    others = 3


class QuantizationScheme:
    """The quantization scheme class describes how an entire model is quantized.
    It contains the quantization specification for each parameter quantization kind.

    Besides, it has an optional field for a visitor class which has the ability to
    take the constructed model (in format of IRModule) as input, go through the
    model and update the QuantizationSpec for certain parameters.
    """

    name: str
    linear_weight: QuantizationSpec
    embedding_table: QuantizationSpec
    final_fc_weight: QuantizationSpec
    others: QuantizationSpec
    pre_quantized: bool

    qspec_updater_class: Optional[Type["QuantSpecUpdater"]]

    """_base_model_prefix is used to match the parameter names when loading from pre-quantized pytorch checkpoints.
    Its value can vary depending on the transformer models used. For example, it can be "model" for Llama, "levit" for LeViT.
    see more candidates in huggingchat [modeling](https://github.com/huggingface/transformers/blob/src/transformers/models/llama/modeling_llama.py#L344) 
    """
    _base_model_prefix: str

    """_layers_block_name is used to match the parameter names when loading from pre-quantized pytorch checkpoints.
    the parameter names usually to be {_base_model_prefix}.{_layers_block_name}.{parameter_name}
    
    For example, the parameter names of the linear layers in Llama are "model.model.encoder.layers.0.linear1.weight", 
    "model.model.encoder.layers.0.linear1.bias", etc.
    """
    _layers_block_name: str

    """_inside_layer_modules defines the names of the layers that are inside the quantize process.
    
    For example, the autogptq scheme:
        ```python
            _inside_layer_modules=[
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"],
                ["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"],
            ]
        ```
    represents the q, k, v projection layers, the output projection layer, and the mlp layers are quantized.
    """
    _inside_layer_modules: List[List[str]]

    """This optional callable function is used to load quantized parameters from the checkpoints.
    If the scheme is used for pre-quantized, this function must be provided.
    """
    _load_quantized_params_func: Optional[Callable]

    def __init__(
        self,
        name: str,
        linear_weight: QuantizationSpec,
        *,
        embedding_table: Optional[Union[QuantizationSpec, Literal["same_as_linear_weight"]]] = None,
        final_fc_weight: Optional[Union[QuantizationSpec, Literal["same_as_linear_weight"]]] = None,
        others: Optional[QuantizationSpec] = None,
        qspec_updater_class: Optional[Type["QuantSpecUpdater"]] = None,
        pre_quantized: bool = False,
        _base_model_prefix: str = "",
        _layers_block_name: str = "",
        _inside_layer_modules: List[List[str]] = [],
        _load_quantized_params_func: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.linear_weight = linear_weight
        self.others = others if others is not None else NoQuantizationSpec(self.model_dtype)

        if embedding_table is None:
            self.embedding_table = self.others
        elif embedding_table == "same_as_linear_weight":
            self.embedding_table = self.linear_weight
        else:
            self.embedding_table = embedding_table

        if final_fc_weight is None:
            self.final_fc_weight = self.others
        elif final_fc_weight == "same_as_linear_weight":
            self.final_fc_weight = self.linear_weight
        else:
            self.final_fc_weight = final_fc_weight

        self.qspec_updater_class = qspec_updater_class

        self.pre_quantized = pre_quantized
        self._base_model_prefix = _base_model_prefix
        self._layers_block_name = _layers_block_name
        self._inside_layer_modules = _inside_layer_modules
        self._load_quantized_params_func = _load_quantized_params_func

    @property
    def model_dtype(self) -> str:
        """Returns the overall model dtype, which is defined as the dtype of
        the linear layers.
        """
        return self.linear_weight.dtype

    def is_inside_layer_modules(self, name: str) -> bool:
        return any(module in name for module in sum(self._inside_layer_modules, []))

    def load_quantized_params(self, *args, **kwargs) -> Optional[List[relax.Var]]:
        if self.pre_quantized:
            return self._load_quantized_params_func(*args, **kwargs)
        else:
            raise RuntimeError("The model is not pre-quantized.")

    def get_layers_block_name(self) -> str:
        return self._layers_block_name

    def get_base_model_prefix(self) -> str:
        return self._base_model_prefix


def convert_TE_func(te_func: Union[FTEQuantize, FTEDequantize], func_name: str) -> FQuantize:
    def func(bb: relax.BlockBuilder, inputs: List[relax.Expr]) -> relax.Var:
        return bb.call_te(te_func, *inputs, primfunc_name_hint=func_name)

    return func


@visitor
class QuantSpecUpdater(PyExprVisitor):
    def __init__(self, param_manager) -> None:
        super().__init__()
        self.param_manager = param_manager
        self.param_map = None
        self.builder = relax.BlockBuilder()

    def lookup_binding(self, var: relax.Var):
        return self.builder.lookup_binding(var)

    def visit_module(self, mod: tvm.IRModule):
        for gv, func in mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is None or not "num_input" in func.attrs:
                continue

            self.param_map = dict()
            num_input = int(func.attrs["num_input"])
            params_in_func = self.param_manager.params_in_func[gv.name_hint]
            assert len(func.params) - num_input == len(params_in_func)
            for i, relax_param in enumerate(func.params[num_input:]):
                self.param_map[relax_param] = params_in_func[i]

            self.builder.normalize(func)
            self.visit_expr(func)
