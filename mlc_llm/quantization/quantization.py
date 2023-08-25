import enum
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple, Type, Union

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
        self, pname: str, param_info: relax.TensorStructInfo
    ) -> Tuple[List[str], List[relax.TensorStructInfo]]:
        """Returns the names and shapes and dtypes of the tensors that need to
        be loaded from the disk.

        It is useful when the parameter is pre-quantized. In such cases, we need
        to know how many tensors the parameter is quantized into, and together
        with the dtype and shape of each tensor, so that we can load the
        pre-quantized tensors in.
        """
        return [pname], [param_info]

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

    qspec_updater_class: Optional[Type["QuantSpecUpdater"]]
    f_convert_param_bkwd: Optional[Callable[[str, Any], Optional[List[Tuple[str, Any]]]]]
    f_compute_relax_param: Optional[Callable[[str, List[Any]], Any]]
    f_run_prequantize: Optional[Callable[[str], str]]

    def __init__(
        self,
        name: str,
        linear_weight: QuantizationSpec,
        *,
        embedding_table: Optional[Union[QuantizationSpec, Literal["same_as_linear_weight"]]] = None,
        final_fc_weight: Optional[Union[QuantizationSpec, Literal["same_as_linear_weight"]]] = None,
        others: Optional[QuantizationSpec] = None,
        qspec_updater_class: Optional[Type["QuantSpecUpdater"]] = None,
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
        self.f_convert_param_bkwd = None
        self.f_compute_relax_param = None
        self.f_run_prequantize = None

        for spec in [self.linear_weight, self.embedding_table, self.final_fc_weight, self.others]:
            if hasattr(spec, "convert_param_bkwd"):
                self.f_convert_param_bkwd = spec.convert_param_bkwd
            if hasattr(spec, "compute_relax_param"):
                self.f_compute_relax_param = spec.compute_relax_param
            if hasattr(spec, "run_prequantize"):
                self.f_run_prequantize = spec.run_prequantize

    @property
    def model_dtype(self) -> str:
        """Returns the overall model dtype, which is defined as the dtype of
        the linear layers.
        """
        return self.linear_weight.dtype


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
