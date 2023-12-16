"""FlashInfer library."""
import dataclasses
from typing import List, Optional

from tvm.relax.frontend import nn


@dataclasses.dataclass
class FlashInfer:
    """A fast kernel library for LLM inference.

    --- Variables ---
    s: sequence length of the current query
    t: total sequence length
    d: head dimension
    h_q: number of heads in query
    h_kv: number of heads in key and value

    --- Shapes ---
    q: [s, h_q, d]
    k: [t, h_kv, d]
    v: [t, h_kv, d]
    o: [1, s, hidden = h_q * d]
    """

    rope_scale: float = 1.0
    rope_theta: float = 10000.0
    mod: Optional[nn.SourceModule] = None

    def configure(
        self,
        arch_list: List[int],
        rope_scale: float,
        rope_theta: float,
    ):
        """Configure FlashInfer as an nn.SourceModule.

        Parameters
        ----------
        arch_list : List[int]
            List of GPU architectures, e.g. [80, 96, 90]

        rope_scale : float
            Scaling factor for the RoPE embedding.

        rope_theta : float
            The base period of the RoPE embedding.
        """

        # pylint: disable=no-member,unexpected-keyword-arg,no-value-for-parameter
        def _infer(q: nn.Tensor, *_args):  # pylint: disable=invalid-name
            _, s, h_q, d = q.shape  # pylint: disable=invalid-name
            return nn.Tensor.placeholder((1, s, h_q * d), dtype="float16")

        assert self.mod is None

        compile_options = nn.SourceModule.get_compile_options(
            source_format="cu",
            tvm_pkg=["flashinfer/include"],
        )
        for arch in arch_list:
            compile_options += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]

        self.rope_scale = rope_scale
        self.rope_theta = rope_theta
        self.mod = nn.SourceModule(
            symbols={
                "FlashInferSinglePrefillWithKVCache": _infer,
                "FlashInferSingleDecodeWithKVCache": _infer,
            },
            source_code=nn.SourceModule.tvm_home() / "3rdparty/flashinfer/src/tvm_wrapper.cu",
            source_format="cu",
            compile_options=compile_options,
            compiler="nvcc",
        )
        nn.add_extern(self.mod)
        # pylint: enable=no-member,unexpected-keyword-arg,no-value-for-parameter

    def single_batch(  # pylint: disable=invalid-name
        self,
        q: nn.Tensor,
        k: nn.Tensor,
        v: nn.Tensor,
    ):
        """Single batch inference with FlashInfer"""
        assert self.mod is not None, "FlashInfer module does not exist"
        assert q.dtype == "float16" and q.ndim == 4
        assert k.dtype == "float16" and k.ndim == 3
        assert v.dtype == "float16" and v.ndim == 3
        _, s, _, _ = q.shape
        casual = 1  # True
        qkv_layout = 0  # "NHD", N for seq_len, H for num_heads, D for head_dim
        rotary_mode = 1  # "kLlama"
        allow_fp16_qk_reduction = 1  # True
        # Decoding
        if isinstance(s, int) and s == 1:
            return self.mod["FlashInferSingleDecodeWithKVCache"](
                q,
                k,
                v,
                qkv_layout,
                rotary_mode,
                self.rope_scale,
                self.rope_theta,
            )
        # Prefilling
        return self.mod["FlashInferSinglePrefillWithKVCache"](
            q,
            k,
            v,
            casual,
            qkv_layout,
            rotary_mode,
            allow_fp16_qk_reduction,
            self.rope_scale,
            self.rope_theta,
        )
