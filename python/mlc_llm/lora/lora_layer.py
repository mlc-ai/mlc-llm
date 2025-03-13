from typing import (
    Any,
    Callable,
    Dict,
    Union,
    Optional,
    List,
)
from tvm import relax as rx
from tvm import tir
from tvm.script import tir as T
from tvm.relax.frontend.nn import Object, Module, ModuleList, Linear, Tensor, Parameter, op
from mlc_llm.lora.backend.base_backend import BaseLoraBackend
import tvm
from tvm.runtime import NDArray


def get_lora_batch_info(max_batch_size: int):
    return op.wrap_nested(
        rx.call_pure_packed(
            "vm.builtin.kv_state_get_lora_batch_info",
            [],
            sinfo_args=[
                rx.TupleStructInfo([
                    rx.TensorStructInfo((max_batch_size,), "int32"),  # seq_lens
                    rx.TensorStructInfo((max_batch_size + 1,), "int32"),  # seq_indptr
                    rx.TensorStructInfo((max_batch_size,), "int32"),  # weight_indices
                ])
            ],
        ),
        name="get_lora_batch_info",
    )

class BaseLora(Module):
    batch_info: List[Tensor] = None
    current_layer_index: int = -1

    def __init__(self):
        super().__init__()


class LinearLoraA(BaseLora):
    """
    Module for Linear Lora A Layer.
    """
    def __init__(
        self,
        in_features: Union[int, str, tir.PrimExpr],
        out_features: Union[int, str, tir.PrimExpr],
        max_loras_per_batch: tir.Var,
        lora_backend: BaseLoraBackend,
        layer_index: int,
        max_batch_size: int,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.weight = Parameter((max_loras_per_batch, out_features, in_features), dtype)
        self.lora_backend = lora_backend
        self.max_batch_size = max_batch_size
        self.out_dtype = out_dtype
        self.to(out_dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for Linear Lora A Layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the Lora Layer.
        """
        # get_lora_batch_info need to be invoked only once for each function,
        # as batch_info is shared for all Lora models in a function. Layer_index of current layer is bigger than that of
        # previous layer or not as a flag to decide whether current Lora layer is the first layer or not.
        if BaseLora.current_layer_index >= self.layer_index:
            BaseLora.batch_info = None
            BaseLora.current_layer_index = -1
        if BaseLora.batch_info is None:
            BaseLora.batch_info = get_lora_batch_info(self.max_batch_size)
            BaseLora.current_layer_index = self.layer_index
        return self.lora_backend.sgemm_lora_a_fwd(
            x,
            self.weight,
            BaseLora.batch_info[0],
            BaseLora.batch_info[1],
            BaseLora.batch_info[2],
        )

    def to(self, dtype: Optional[str] = None) -> None:
        self.weight.to(dtype=dtype)
        self.out_dtype = dtype


class LinearLoraB(BaseLora):
    """
    Module for Linear Lora B layer.
    """

    def __init__(
        self,
        in_features: Union[int, str, tir.PrimExpr],
        out_features: Union[int, str, tir.PrimExpr],
        stacked_num: int,
        lora_module_name: str,
        lora_backend: BaseLoraBackend,
        max_loras_per_batch: T.Var,
        q_output_dim: int,
        kv_output_dim: int,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.max_loras_per_batch = max_loras_per_batch
        self.weight = Parameter((max_loras_per_batch, out_features, in_features), dtype)
        self.lora_module_name = lora_module_name
        self.lora_stacked_num = stacked_num
        self.lora_backend = lora_backend
        self.q_output_dim = q_output_dim
        self.kv_output_dim = kv_output_dim
        self.out_dtype = out_dtype
        self.to(out_dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for Linear Lora B layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the Lora layer.
        """
        assert BaseLora.batch_info
        if self.lora_stacked_num == 1:
            return self.lora_backend.sgemm_lora_a_fwd(
                x,
                self.weight,
                BaseLora.batch_info[0],
                BaseLora.batch_info[1],
                BaseLora.batch_info[2],
            )
        else:
            if self.lora_module_name == "qkv_proj":
                return self.lora_backend.qkv_lora_b_fwd(
                    x,
                    self.weight,
                    self.q_output_dim,
                    self.kv_output_dim,
                    BaseLora.batch_info[0],
                    BaseLora.batch_info[1],
                    BaseLora.batch_info[2],
                )
            elif self.lora_module_name == "gate_up_proj":
                return self.lora_backend.gate_up_lora_b_fwd(
                    x,
                    self.weight,
                    BaseLora.batch_info[0],
                    BaseLora.batch_info[1],
                    BaseLora.batch_info[2],
                )
            else:
                raise NotImplementedError(f"Unsupport lora module {self.lora_module_name}")
    
    def to(self, dtype: Optional[str]=None) -> None:
        self.weight.to(dtype=dtype)
        self.out_dtype = dtype