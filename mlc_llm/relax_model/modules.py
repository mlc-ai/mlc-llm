# pylint: disable=missing-docstring,invalid-name
from typing import Dict, List, Tuple

import numpy as np
from tvm import relax, te, tir
from tvm.relax.op import reshape, take
from tvm.relax.op.nn import layer_norm
from tvm.relax.testing import nn
from tvm.runtime.ndarray import array as tvm_array


class ModuleList(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features), dtype=dtype, name="linear_weight"
        )
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(x, self.weight, self.bias))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        x_shape = x.struct_info.shape.values
        emb_size = self.weight.struct_info.shape.values[-1]
        x = nn.emit(reshape(x, shape=[-1]))
        embedding = nn.emit(take(self.weight, x, axis=0))
        return nn.emit(reshape(embedding, [*x_shape, emb_size]))


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        dtype,
        eps=1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,), dtype="float32", name="weight")
        self.bias = nn.Parameter((hidden_size,), dtype="float32", name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        if x.struct_info.dtype != "float32":
            x = nn.emit(relax.op.astype(x, "float32"))
        x = nn.emit(
            layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        position_embedding_base: int,
        max_sequence_length: int,
        rotary_pct: float,
        swizzle_style: str = "neox",
        dtype: str = "float32",
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        rotary_ndim = int(head_dim * rotary_pct)
        inv_freq = 1.0 / (
            position_embedding_base
            ** (np.arange(0, rotary_ndim, 2).astype("float32") / rotary_ndim)
        )
        t = np.arange(max_sequence_length, dtype=inv_freq.dtype)
        freq = np.einsum("i,j->ij", t, inv_freq)
        if swizzle_style == "neox":
            emb = np.concatenate((freq, freq), axis=-1)
        elif swizzle_style == "gptj":
            emb = np.repeat(freq, repeats=2, axis=-1)
        else:
            raise KeyError("Unrecognized swizzle style {}".format(swizzle_style))
        self.swizzle_style = swizzle_style
        self.rotary_ndim = rotary_ndim
        self.cos_cached = relax.const(tvm_array(np.cos(emb).astype(dtype)))
        self.sin_cached = relax.const(tvm_array(np.sin(emb).astype(dtype)))

    def get_x_swizzle(self, x, i_batch_size, i_seq_len, i_num_heads, i_head_dim):
        if self.swizzle_style == "neox":
            n_feat_half = self.rotary_ndim
            return tir.Select(
                i_head_dim < n_feat_half,
                -x[
                    i_batch_size,
                    i_seq_len,
                    i_num_heads,
                    i_head_dim + n_feat_half,
                ],
                x[
                    i_batch_size,
                    i_seq_len,
                    i_num_heads,
                    i_head_dim - n_feat_half,
                ],
            )
        elif self.swizzle_style == "gptj":
            return tir.Select(
                i_head_dim % 2 == 0,
                -x[i_batch_size, i_seq_len, i_num_heads, i_head_dim + 1],
                x[i_batch_size, i_seq_len, i_num_heads, i_head_dim - 1],
            )
        else:
            raise KeyError("Unrecognized swizzle style: {}.".format(self.swizzle_style))

    def forward(
        self,
        q: relax.Expr,
        k: relax.Expr,
        offset: relax.Expr,
    ) -> Tuple[relax.Expr, relax.Expr]:
        def rotary_embedding(x, cos, sin, offset):
            def compute(
                i_batch_size,
                i_seq_len,
                i_num_heads,
                i_head_dim,
            ):
                return tir.Select(
                    i_head_dim < self.rotary_ndim,
                    cos[
                        offset + i_seq_len,
                        i_head_dim,
                    ]
                    * x(i_batch_size, i_seq_len, i_num_heads, i_head_dim)
                    + sin[
                        offset + i_seq_len,
                        i_head_dim,
                    ]
                    * self.get_x_swizzle(
                        x, i_batch_size, i_seq_len, i_num_heads, i_head_dim
                    ),
                    x(i_batch_size, i_seq_len, i_num_heads, i_head_dim),
                )

            return te.compute(x.shape, compute, name="rotary")

        cos, sin = self.cos_cached, self.sin_cached
        q_embed = nn.emit_te(
            rotary_embedding,
            q,
            cos,
            sin,
            offset,
            primfunc_name_hint="rotary_embedding",
        )
        k_embed = nn.emit_te(
            rotary_embedding,
            k,
            cos,
            sin,
            offset,
            primfunc_name_hint="rotary_embedding",
        )
        return q_embed, k_embed


def named_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    params: Dict[str, nn.Parameter] = {}
    for name, module in model.__dict__.items():
        if isinstance(module, nn.Parameter):
            params[name] = module
        elif isinstance(module, ModuleList):
            for i, m in enumerate(module):
                for param_name, param in named_parameters(m).items():
                    params[f"{name}.{i}.{param_name}"] = param
        elif isinstance(module, nn.Module):
            for param_name, param in named_parameters(module).items():
                params[f"{name}.{param_name}"] = param
    return params
