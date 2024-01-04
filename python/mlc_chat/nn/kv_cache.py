"""Attention KV cache modeling."""
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor

from ..op.kv_cache import kv_cache_debug_get_kv, kv_cache_transpose_append


class PagedKVCache(Object):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    def attention(self, layer_id: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute attention with the given q/k/v data and in-cache k/v data
        on the specified layer. Rotary position embeddings are applied to k/v
        within this function.

        - For prefill, the input q and output tensor have shape
        (1, total_seq_len, num_attention_heads, head_dim), and the
        k/v tensors have shape (1, total_seq_len, num_key_value_heads, head_dim).
        - For decode, the input q and output tensor have shape
        (batch_size, 1, num_attention_heads, head_dim), and the
        k/v tensors have shape (batch_size, 1, num_key_value_heads, head_dim).
        """
        # pylint: disable=protected-access
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.paged_attention_kv_cache_attention",
                    [self._expr, rx.PrimValue(layer_id), q._expr, k._expr, v._expr],
                    out_sinfo=q._expr.struct_info,
                )
            )
        )
        # pylint: enable=protected-access


class FlashInferPagedKVCache(PagedKVCache):
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    @staticmethod
    def create(  # pylint: disable=too-many-arguments
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        page_size: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        name: str = "paged_kv_cache",
    ) -> "FlashInferPagedKVCache":
        """Create a paged KV cache object with FlashInfer kernels.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tir.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        """

        bb = rx.BlockBuilder.current()
        return PagedKVCache(
            _expr=rx.Call(
                rx.extern("vm.builtin.paged_attention_kv_cache_create"),
                args=[
                    rx.ShapeExpr([max_batch_size, max_total_seq_len, page_size]),
                    rx.PrimValue(num_hidden_layers),
                    rx.PrimValue(num_attention_heads),
                    rx.PrimValue(num_key_value_heads),
                    rx.PrimValue(head_dim),
                    rx.PrimValue(rope_scale),
                    rx.PrimValue(rope_theta),
                    rx.op.zeros((), dtype),
                    bb.add_func(
                        kv_cache_transpose_append(num_key_value_heads, head_dim, dtype),
                        "kv_cache_transpose_append",
                    ),
                    rx.extern("paged_kv_cache.attention_kernel_prefill"),
                    rx.extern("paged_kv_cache.attention_kernel_decode"),
                    rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"),
                    rx.extern(
                        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"
                    ),
                    rx.extern(
                        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"
                    ),
                    rx.extern("paged_kv_cache.attention_kernel_prefill_begin_forward"),
                    rx.extern("paged_kv_cache.attention_kernel_prefill_end_forward"),
                    rx.extern("paged_kv_cache.attention_kernel_decode_begin_forward"),
                    rx.extern("paged_kv_cache.attention_kernel_decode_end_forward"),
                    rx.extern("flashinfer.batch_qk_apply_rotary_in_place"),
                    rx.extern("flashinfer.merge_state_in_place"),
                    bb.add_func(
                        kv_cache_debug_get_kv(
                            num_hidden_layers, num_key_value_heads, head_dim, dtype
                        ),
                        "kv_cache_debug_get_kv",
                    ),
                ],
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )
