# pylint: disable=missing-docstring,invalid-name
from typing import Dict, List, Tuple, Optional

import numpy as np
from tvm import relax, te, tir
from tvm.relax.op import matmul, permute_dims, reshape, take
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
    def __init__(
        self,
        in_features,
        out_features,
        dtype,
        bias=True,
        out_dtype=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features),
            dtype=dtype,
            name="linear_weight",
        )
        if bias:
            self.bias = nn.Parameter(
                (out_features,),
                dtype=dtype if out_dtype is None else out_dtype,
                name="linear_bias",
            )
        else:
            self.bias = None
        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(x)
        weight = permute_dims(self.weight, axes=None)
        x = nn.emit(matmul(x, weight, out_dtype=self.out_dtype))
        if self.bias is not None:
            x = nn.emit(x + self.bias)
        return x


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
        rotary_pct: Optional[float] = None,
        rotary_dim: Optional[int] = None,
        swizzle_style: str = "neox",
        dtype: str = "float32",
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        if rotary_dim is not None:
            rotary_ndim = rotary_dim
        else:
            rotary_ndim = int(head_dim * rotary_pct)
        inv_freq = 1.0 / (
            position_embedding_base
            ** (np.arange(0, rotary_ndim, 2).astype("float32") / rotary_ndim)
        )
        t = np.arange(max_sequence_length, dtype=inv_freq.dtype)
        freq = np.einsum("i,j->ij", t, inv_freq)
        if swizzle_style == "neox":
            emb = np.concatenate((freq, freq), axis=-1)
        elif swizzle_style in ("gptj", "glm"):
            emb = np.repeat(freq, repeats=2, axis=-1)
        else:
            raise KeyError("Unrecognized swizzle style {}".format(swizzle_style))
        self.swizzle_style = swizzle_style
        self.rotary_ndim = rotary_ndim
        self.cos_cached = relax.const(tvm_array(np.cos(emb).astype(dtype)))
        self.sin_cached = relax.const(tvm_array(np.sin(emb).astype(dtype)))

    def get_x_swizzle(self, x, i_batch_size, i_seq_len, i_num_heads, i_head_dim):
        if self.swizzle_style == "neox":
            n_feat_half = self.rotary_ndim // 2
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
        elif self.swizzle_style in ("gptj", "glm"):
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


class TransformImage(nn.Module):
    def __init__(self, dtype: str, in_chans: int = 4):
        self.in_chans = in_chans
        self.dtype = dtype

        # used in normalization, assume channels are RGB
        self.r_mean = relax.const(0.48145466, "float32")
        self.g_mean = relax.const(0.4578275, "float32")
        self.b_mean = relax.const(0.40821073, "float32")
        self.r_std = relax.const(0.26862954, "float32")
        self.g_std = relax.const(0.26130258, "float32")
        self.b_std = relax.const(0.27577711, "float32")

    def forward(self, input: relax.Expr) -> relax.Expr:
        from tvm.relax.op import astype, concat, permute_dims, strided_slice

        assert input.struct_info.ndim == 4
        # perform torch.ToTensor on input of shape (bs, height, width, in_chans)
        input = permute_dims(input, [0, 3, 1, 2])
        x = astype(input, "float32") / relax.const(255.0, "float32")
        r = strided_slice(x, axes=[1], begin=[0], end=[1])
        g = strided_slice(x, axes=[1], begin=[1], end=[2])
        b = strided_slice(x, axes=[1], begin=[2], end=[3])

        # normalize rgba to rgb
        if self.in_chans == 4:
            a = strided_slice(x, axes=[1], begin=[3], end=[4])
            r /= a
            g /= a
            b /= a

        # perform torch.Normalize
        r = (r - self.r_mean) / self.r_std
        g = (g - self.g_mean) / self.g_std
        b = (b - self.b_mean) / self.b_std
        res = concat([r, g, b], axis=1)
        res = astype(res, self.dtype)

        return res


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
