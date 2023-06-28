import enum
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Union

from tvm import relax, te

FQuantize = Callable[[te.Tensor], List[te.Tensor]]
FDequantize = Callable[[List[te.Tensor]], te.Tensor]


@dataclass
class QuantizationSpec:
    """The base dataclass of quantization specification.
    A specification describes how a parameter is quantized and dequantized.

    A subclass of QuantizationSpec
      - contains more data fields (e.g., the "group size" in group quantization)
      which instruct the quantization/dequantization,
      - defines the `get_quantize_func` method, which returns a TE function
      (`Callable[[te.Tensor], List[te.Tensor]]`) that describes the computation
      algorithm of the quantization.
      - defines the `get_dequantize_func` method, which returns a TE function
      (`Callable[[List[te.Tensor]], te.Tensor]`) that describes the computation
      algorithm of the dequantization.
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

    def get_quantize_func(
        self, param_info: relax.TensorStructInfo
    ) -> Optional[FQuantize]:
        """Returns the TE function which computes quantization.
        Returning `None` means the parameter does not need quantization or is
        pre-quantized.
        """
        return NotImplementedError()

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FDequantize]:
        """Returns the TE function which computes dequantization.
        Returning `None` means the parameter does not need dequantization.
        """
        return NotImplementedError()


@dataclass
class NoQuantizationSpec(QuantizationSpec):
    """The quantization specification that describes doing no quantization."""

    def get_quantize_func(
        self, param_info: relax.TensorStructInfo
    ) -> Optional[FQuantize]:
        return None

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FDequantize]:
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
    """

    name: str
    linear_weight: QuantizationSpec
    embedding_table: QuantizationSpec
    final_fc_weight: QuantizationSpec
    others: QuantizationSpec

    def __init__(
        self,
        name: str,
        linear_weight: QuantizationSpec,
        *,
        embedding_table: Optional[
            Union[QuantizationSpec, Literal["same_as_linear_weight"]]
        ] = None,
        final_fc_weight: Optional[
            Union[QuantizationSpec, Literal["same_as_linear_weight"]]
        ] = None,
        others: Optional[QuantizationSpec] = None
    ) -> None:
        self.name = name
        self.linear_weight = linear_weight
        self.others = (
            others if others is not None else NoQuantizationSpec(self.model_dtype)
        )

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

    @property
    def model_dtype(self) -> str:
        """Returns the overall model dtype, which is defined as the dtype of
        the linear layers.
        """
        return self.linear_weight.dtype
