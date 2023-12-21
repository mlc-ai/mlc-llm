"""A compiler pass that fuses split + rotary embedding."""
import tvm
from tvm import relax, tir
from tvm.relax.dpl import (
    GlobalVarPattern,
    PatternContext,
    TuplePattern,
    is_op,
    is_tuple_get_item,
    rewrite_bindings,
    wildcard,
)
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.stmt_functor import post_order_visit

# pylint: disable=too-many-arguments,too-many-locals,protected-access,too-many-statements


def get_dynamic_split_rotary(dtype):
    """Implementation of R.split(rotary_embedding(fused_qkv))

    Implementation is generic over the number of query heads,
    key/value heads, sequence length, head dimension, and position
    embedding base.  These parameters can be replaced with static
    values using `PrimFunc.specialize`.
    """

    @T.prim_func(private=True)
    def split_rotary(
        fused_qkv_handle: T.handle,
        embedded_query_handle: T.handle,
        embedded_key_handle: T.handle,
        value_handle: T.handle,
        rotary_offset: T.int64,
        batch_size: T.int64,
        seq_len: T.int64,
        num_query_heads: T.int64,
        num_kv_heads: T.int64,
        head_dim: T.int64,
        position_embedding_base: T.float32,
    ):
        fused_qkv = T.match_buffer(
            fused_qkv_handle,
            [batch_size, seq_len, num_query_heads + num_kv_heads * 2, head_dim],
            dtype=dtype,
        )
        embedded_query = T.match_buffer(
            embedded_query_handle,
            [batch_size, seq_len, num_query_heads, head_dim],
            dtype=dtype,
        )
        embedded_key = T.match_buffer(
            embedded_key_handle,
            [batch_size, seq_len, num_kv_heads, head_dim],
            dtype=dtype,
        )
        value = T.match_buffer(
            value_handle,
            [batch_size, seq_len, num_kv_heads, head_dim],
            dtype=dtype,
        )

        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})

        for iters in T.grid(batch_size, seq_len, num_query_heads + num_kv_heads * 2, head_dim):
            with T.block("FusedRotaryEmbeddingAndSplitQKV"):
                batch_i, seq_i, head_num, head_i = T.axis.remap("SSSS", iters)
                pos: T.float32 = T.Cast("float32", rotary_offset + seq_i - seq_len)

                inv_freq: T.float32 = T.float32(1) / T.pow(
                    position_embedding_base,
                    T.Cast("float32", (head_i * 2) % head_dim) / T.float32(head_dim),
                )
                freq: T.float32 = pos * inv_freq
                cos_value = T.Cast("float16", T.cos(freq)) if dtype == "float16" else T.cos(freq)
                sin_value = T.Cast("float16", T.sin(freq)) if dtype == "float16" else T.sin(freq)

                input_value = fused_qkv[batch_i, seq_i, head_num, head_i]
                embedded_value = cos_value * input_value + sin_value * T.Select(
                    head_i < T.int64(head_dim // 2),
                    fused_qkv[batch_i, seq_i, head_num, head_i + T.int64(head_dim // 2)]
                    * T.FloatImm(dtype, -1),
                    fused_qkv[batch_i, seq_i, head_num, head_i - T.int64(head_dim // 2)],
                )
                if head_num < num_query_heads:
                    embedded_query[batch_i, seq_i, head_num, head_i] = embedded_value
                elif head_num < num_query_heads + num_kv_heads:
                    embedded_key[
                        batch_i, seq_i, head_num - num_query_heads, head_i
                    ] = embedded_value
                else:
                    value[
                        batch_i, seq_i, head_num - num_query_heads - num_kv_heads, head_i
                    ] = input_value

    param_sinfo = []
    for param in split_rotary.params:
        if param in split_rotary.buffer_map:
            buf = split_rotary.buffer_map[param]
            sinfo = relax.TensorStructInfo(shape=buf.shape, dtype=buf.dtype)
        else:
            sinfo = relax.PrimStructInfo(param.dtype)
        param_sinfo.append(sinfo)

    return split_rotary


def collect_position_embedding_base(func: tir.PrimFunc):
    """Collect position embedding base from rotary embedding function"""
    position_embedding_base = None

    def visit(node):
        nonlocal position_embedding_base
        if isinstance(node, tir.Call) and node.op.name == "tir.pow":
            if isinstance(node.args[0], tir.FloatImm):
                position_embedding_base = node.args[0].value
            else:
                raise ValueError("position_embedding_base is not a constant")

    post_order_visit(func.body, visit)
    return position_embedding_base


@tvm.transform.module_pass(opt_level=0, name="FuseSplitRotaryEmbedding")
class FuseSplitRotaryEmbedding:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses split and rotary embedding"""

    def transform_module(
        self,
        mod: tvm.IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> tvm.IRModule:
        """IRModule-level transformation"""
        with PatternContext() as ctx:
            pat_rotary_embedding_gvar = GlobalVarPattern()

            pat_offset = wildcard()

            pat_fused_qkv = wildcard()

            pat_qkv_tuple = is_op("relax.split")(pat_fused_qkv)
            pat_fused_qkv.used_by(pat_qkv_tuple)
            pat_query = is_tuple_get_item(pat_qkv_tuple, 0)
            pat_qkv_tuple.used_by(pat_query)
            pat_key = is_tuple_get_item(pat_qkv_tuple, 1)
            pat_qkv_tuple.used_by(pat_key)
            pat_value = is_tuple_get_item(pat_qkv_tuple, 2)
            pat_qkv_tuple.used_by(pat_value)

            pat_embedded_query = is_op("relax.call_tir")(
                pat_rotary_embedding_gvar,
                TuplePattern([pat_query]),
                pat_offset,
                add_constraint=False,
            )
            pat_embedded_key = is_op("relax.call_tir")(
                pat_rotary_embedding_gvar,
                TuplePattern([pat_key]),
                pat_offset,
                add_constraint=False,
            )

            pat_query.used_by(pat_embedded_query)
            pat_key.used_by(pat_embedded_key)

        def rewriter(matchings, bindings):
            # Extracting all the relax and TIR variables that we'll need
            fused_qkv = matchings[pat_fused_qkv]

            query = matchings[pat_query]
            key = matchings[pat_key]
            value = matchings[pat_value]

            embedded_query = matchings[pat_embedded_query]
            embedded_key = matchings[pat_embedded_key]

            rotary_embedding_gvar = bindings[embedded_query].args[0]
            rotary_embedding_func = mod[rotary_embedding_gvar]
            if (
                "mlc.rotary_embedding_to_all_dims" not in rotary_embedding_func.attrs
                or not rotary_embedding_func.attrs["mlc.rotary_embedding_to_all_dims"]
            ):
                # manually skip split rotary fuse
                return {}
            position_embedding_base = collect_position_embedding_base(rotary_embedding_func)

            rotary_embedding_offset = bindings[embedded_query].args[-1][0]

            batch_size, seq_len, num_query_heads, head_dim = query.struct_info.shape
            _, _, num_kv_heads, _ = key.struct_info.shape
            dtype = query.struct_info.dtype
            assert dtype in ["float16", "float32"]
            # Rewriting along the new path
            split_rotary_sinfo = [
                R.Tensor((batch_size, seq_len, num_query_heads, head_dim), dtype=dtype),
                R.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype),
                R.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype),
            ]
            if not any(gvar.name_hint == "split_rotary" for gvar in mod.functions):
                split_rotary = get_dynamic_split_rotary(dtype)

                (
                    dyn_batch_size,
                    dyn_seq_len,
                    dyn_num_query_heads,
                    dyn_num_kv_heads,
                    dyn_head_dim,
                    dyn_position_embedding_base,
                ) = split_rotary.params[-6:]

                split_rotary = split_rotary.specialize(
                    {
                        # Static model parameters
                        dyn_batch_size: T.int64(batch_size),
                        dyn_num_query_heads: T.int64(num_query_heads),
                        dyn_num_kv_heads: T.int64(num_kv_heads),
                        dyn_head_dim: T.int64(head_dim),
                        dyn_position_embedding_base: T.float32(position_embedding_base),
                        # Dynamic parameters, to be inferred from TIR Buffer shapes
                        dyn_seq_len: tvm.tir.Var("query_sequence_length", "int64"),
                    }
                )
                mod["split_rotary"] = split_rotary
                split_rotary_gvar = mod.get_global_var("split_rotary")
                relax.expr._update_struct_info(split_rotary_gvar, relax.ObjectStructInfo())
            else:
                split_rotary_gvar = mod.get_global_var("split_rotary")

            qkv_tuple_new = R.call_tir(
                split_rotary_gvar,
                (fused_qkv,),
                out_sinfo=split_rotary_sinfo,
                tir_vars=[rotary_embedding_offset],
            )

            embedded_query_new = qkv_tuple_new[0]
            embedded_key_new = qkv_tuple_new[1]
            value_new = qkv_tuple_new[2]

            return {
                value: value_new,
                embedded_query: embedded_query_new,
                embedded_key: embedded_key_new,
            }

        new_mod = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                func = rewrite_bindings(ctx, rewriter, func)
            new_mod[gvar] = func
        for gvar, func in mod.functions.items():
            if isinstance(func, tir.PrimFunc) and gvar not in new_mod:
                new_mod[gvar] = func
        new_mod = tvm.IRModule(new_mod, mod.type_definitions, mod.attrs, mod.global_infos)
        return new_mod
