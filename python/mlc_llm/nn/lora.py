"""LoRA (Low-Rank Adaptation) implementation for MLC LLM."""
import math
from typing import Optional, Union

from tvm import relax, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.support import logging
from mlc_llm.lora.lora_config import LoRAConfig  # Use shared config implementation

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation) support.
    
    This implementation follows the paper: https://arxiv.org/abs/2106.09685
    
    LoRA decomposes the weight update into two low-rank matrices:
    h = Wx + BAx where B ∈ R^{d×r}, A ∈ R^{r×k}
    
    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : Union[int, tir.Var]
        Size of each output sample
    r : int
        LoRA rank (typically 4, 8, 16, or 32)
    lora_alpha : float
        LoRA scaling factor
    lora_dropout : float
        Dropout probability for LoRA layers
    fan_in_fan_out : bool
        Whether the layer uses fan_in_fan_out convention
    merge_weights : bool
        Whether to merge LoRA weights during inference
    bias : bool
        Whether to use bias in the base linear layer
    dtype : Optional[str]
        Data type of the layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: Union[int, tir.Var],
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        bias: bool = True,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        
        # Base linear layer
        self.weight = nn.Parameter((out_features, in_features), dtype=dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype)
        else:
            self.bias = None
            
        # LoRA layers
        if r > 0:
            self.lora_A = nn.Parameter((r, in_features), dtype=dtype)
            self.lora_B = nn.Parameter((out_features, r), dtype=dtype)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            logger.info(
                f"Created LoRA layer: in_features={in_features}, "
                f"out_features={out_features}, r={r}, alpha={lora_alpha}"
            )
        else:
            self.lora_A = None
            self.lora_B = None
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        if self.r > 0:
            # Initialize A with Kaiming uniform and B with zeros
            # This ensures LoRA starts from zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional LoRA adaptation."""
        if self.r > 0 and not self.merged:
            # Use the fused helper so we have identical code-path everywhere.
            from mlc_llm.op.lora import lora_dense  # local import to avoid cycle

            # Compose delta = BA (shape: out_features × in_features)
            if self.lora_A is None or self.lora_B is None:  # pragma: no cover
                raise RuntimeError("LoRA parameters not initialised properly")

            delta_w = op.matmul(self.lora_B, self.lora_A)
            result = lora_dense(x, self.weight, delta_w, self.scaling)

            if self.bias is not None:
                result = result + self.bias

            return result
        else:
            # Use merged weights or no LoRA
            result = op.matmul(x, op.permute_dims(self.weight))
            if self.bias is not None:
                result = result + self.bias
            return result
    
    def merge_weights(self):
        """Merge LoRA weights into the base weights for efficient inference."""
        if self.r > 0 and not self.merged:
            # Merge: W' = W + BA * scaling
            delta_w = op.matmul(self.lora_B, self.lora_A) * self.scaling
            self.weight.data += delta_w
            self.merged = True
            logger.info("Merged LoRA weights into base weights")
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the base weights."""
        if self.r > 0 and self.merged:
            # Unmerge: W = W' - BA * scaling
            delta_w = op.matmul(self.lora_B, self.lora_A) * self.scaling
            self.weight.data -= delta_w
            self.merged = False
            logger.info("Unmerged LoRA weights from base weights")
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
    ) -> "LoRALinear":
        """
        Convert a standard nn.Linear layer to LoRALinear.
        
        Parameters
        ----------
        linear : nn.Linear
            The linear layer to convert
        r : int
            LoRA rank
        lora_alpha : float
            LoRA scaling factor
        lora_dropout : float
            Dropout probability
        fan_in_fan_out : bool
            Whether to use fan_in_fan_out convention
        merge_weights : bool
            Whether to merge weights during inference
            
        Returns
        -------
        LoRALinear
            The converted LoRA linear layer
        """
        out_features, in_features = linear.weight.shape
        lora_linear = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            merge_weights=merge_weights,
            bias=getattr(linear, "bias", None) is not None,
            dtype=linear.weight.dtype,
        )
        
        # Copy weights from original linear layer
        lora_linear.weight.data = linear.weight.data
        if hasattr(linear, "bias") and linear.bias is not None:
            lora_linear.bias.data = linear.bias.data
            
        # Initialize LoRA parameters
        lora_linear.reset_parameters()
        
        # Copy attributes
        if hasattr(linear.weight, "attrs"):
            lora_linear.weight.attrs = linear.weight.attrs
        if hasattr(linear, "bias") and linear.bias is not None and hasattr(linear.bias, "attrs"):
            lora_linear.bias.attrs = linear.bias.attrs
            
        return lora_linear


# NOTE: The original LoRAConfig implementation previously lived in this file
# but has been promoted to ``mlc_llm.lora.lora_config`` so it can be reused by
# the new unified LoRA pipeline.  To preserve backward-compatibility we import
# the canonical definition above and simply re-export it here.

# Re-export for ``from mlc_llm.nn import LoRAConfig`` users
__all__ = [
    "LoRALinear",
    "LoRAConfig",
] 