import tvm
from tvm import relax
from tvm.relax.dpl import (
    PatternContext,
    is_op,
    rewrite_bindings,
    wildcard,
    is_tuple_get_item,
    GlobalVarPattern,
    TuplePattern,
    is_shape,
)
from tvm.script import relax as R, tir as T


def get_dynamic_split_rotary():
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
        Fused_QKV = T.match_buffer(
            fused_qkv_handle,
            [batch_size, seq_len, num_query_heads + num_kv_heads * 2, head_dim],
            dtype="float16",
        )
        EmbeddedQuery = T.match_buffer(
            embedded_query_handle,
            [batch_size, seq_len, num_query_heads, head_dim],
            dtype="float16",
        )
        EmbeddedKey = T.match_buffer(
            embedded_key_handle,
            [batch_size, seq_len, num_kv_heads, head_dim],
            dtype="float16",
        )
        Value = T.match_buffer(
            value_handle,
            [batch_size, seq_len, num_kv_heads, head_dim],
            dtype="float16",
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
                cos_value: T.float16 = T.Cast("float16", T.cos(freq))
                sin_value: T.float16 = T.Cast("float16", T.sin(freq))

                input_value = Fused_QKV[batch_i, seq_i, head_num, head_i]
                embedded_value = cos_value * input_value + sin_value * T.Select(
                    head_i < T.int64(head_dim // 2),
                    Fused_QKV[batch_i, seq_i, head_num, head_i + T.int64(head_dim // 2)]
                    * T.float16(-1),
                    Fused_QKV[batch_i, seq_i, head_num, head_i - T.int64(head_dim // 2)],
                )
                if head_num < num_query_heads:
                    EmbeddedQuery[batch_i, seq_i, head_num, head_i] = embedded_value
                elif head_num < num_query_heads + num_kv_heads:
                    EmbeddedKey[batch_i, seq_i, head_num - num_query_heads, head_i] = embedded_value
                else:
                    Value[
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

    relax.expr._update_struct_info(
        split_rotary,
        tvm.relax.FuncStructInfo(
            params=param_sinfo,
            ret=relax.TupleStructInfo([]),
            purity=False,
        ),
    )

    return split_rotary


def fuse_split_rotary_embedding(
    num_query_heads, num_kv_heads, hidden_size, position_embedding_base
):
    @tvm.ir.transform.module_pass(opt_level=0, name="fuse_split_rotary_embedding")
    def ir_module_pass(mod: tvm.IRModule, _pass_context) -> tvm.IRModule:
        head_dim = hidden_size // num_query_heads
        split_rotary = get_dynamic_split_rotary()

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
                dyn_batch_size: T.int64(1),
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
        relax.expr._update_struct_info(split_rotary_gvar, mod["split_rotary"].struct_info)

        with PatternContext() as ctx:
            # flat_qkv_tuple: R.Tuple(
            #     R.Tensor((batch_size, seq_len, 4096), dtype="float16"),
            #     R.Tensor((batch_size, seq_len, 4096), dtype="float16"),
            #     R.Tensor((batch_size, seq_len, 4096), dtype="float16"),
            # ) = R.split(flat_fused_qkv, indices_or_sections=[4096, 8192], axis=2)
            #
            # flat_query: R.Tensor((batch_size, seq_len, 4096), dtype="float16") = flat_qkv_tuple[0]
            # query: R.Tensor((batch_size, seq_len, 32, 128), dtype="float16") = R.reshape(
            #     flat_query, R.shape([batch_size, seq_len, 32, 128])
            # )
            # flat_key: R.Tensor((batch_size, seq_len, 4096), dtype="float16") = flat_qkv_tuple[1]
            # key: R.Tensor((batch_size, seq_len, 32, 128), dtype="float16") = R.reshape(
            #     flat_key, R.shape([batch_size, seq_len, 32, 128])
            # )
            # flat_value: R.Tensor((batch_size, seq_len, 4096), dtype="float16") = flat_qkv_tuple[2]
            # value: R.Tensor((batch_size, seq_len, 32, 128), dtype="float16") = R.reshape(
            #     flat_value, R.shape([batch_size, seq_len, 32, 128])
            # )
            # embedded_query = R.call_tir(
            #     cls.rotary_embedding1,
            #     [query],
            #     out_sinfo=R.Tensor((batch_size, seq_len, 32, 128), dtype="float16"),
            #     tir_vars=R.shape([n]),
            # )
            # embedded_key = R.call_tir(
            #     cls.rotary_embedding1,
            #     [key],
            #     out_sinfo=R.Tensor((batch_size, seq_len, 32, 128), dtype="float16"),
            #     tir_vars=R.shape([n]),
            # )

            pat_rotary_embedding_gvar = GlobalVarPattern()

            pat_flat_fused_qkv = wildcard()
            pat_offset = wildcard()

            # query_shape = is_shape([1, seq_len, num_query_heads, head_dim])
            pat_query_shape = wildcard()
            # value_shape = is_shape([1, seq_len, num_kv_heads, head_dim])
            pat_key_shape = wildcard()
            # value_shape = is_shape([1, seq_len, num_kv_heads, head_dim])
            pat_value_shape = wildcard()

            pat_flat_qkv_tuple = is_op("relax.split")(pat_flat_fused_qkv)
            pat_flat_query = is_tuple_get_item(pat_flat_qkv_tuple, 0)
            pat_query = is_op("relax.reshape")(
                pat_flat_query, pat_query_shape, add_constraint=False
            )
            pat_flat_query.used_by(pat_query)
            pat_flat_key = is_tuple_get_item(pat_flat_qkv_tuple, 1)
            pat_key = is_op("relax.reshape")(pat_flat_key, pat_key_shape, add_constraint=False)
            pat_flat_key.used_by(pat_key)
            pat_flat_value = is_tuple_get_item(pat_flat_qkv_tuple, 2)
            pat_value = is_op("relax.reshape")(
                pat_flat_value, pat_value_shape, add_constraint=False
            )
            pat_flat_value.used_by(pat_value)

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

            pat_flat_qkv_tuple.used_by(pat_flat_query)
            pat_flat_qkv_tuple.used_by(pat_flat_key)
            pat_flat_qkv_tuple.used_by(pat_flat_value)
            pat_query.used_by(pat_embedded_query)
            pat_key.used_by(pat_embedded_key)

        def rewriter(matchings, bindings):
            # Extracting all the relax and TIR variables that we'll need
            flat_fused_qkv = matchings[pat_flat_fused_qkv]
            flat_qkv_tuple = matchings[pat_flat_qkv_tuple]

            flat_query = matchings[pat_flat_query]
            flat_key = matchings[pat_flat_key]
            flat_value = matchings[pat_flat_value]

            query = matchings[pat_query]
            key = matchings[pat_key]
            value = matchings[pat_value]

            embedded_query = matchings[pat_embedded_query]
            embedded_key = matchings[pat_embedded_key]

            # rotary_embedding_offset = bindings[query].args[-1][1]
            rotary_embedding_offset = bindings[embedded_query].args[-1][0]

            batch_size, seq_len, num_query_heads, head_dim = query.struct_info.shape
            _batch_size, _seq_len, num_kv_heads, _head_dim = key.struct_info.shape

            # Rewriting along the new path

            fused_qkv = relax.op.reshape(
                flat_fused_qkv, [batch_size, seq_len, num_query_heads + 2 * num_kv_heads, head_dim]
            )

            split_rotary_sinfo = [
                R.Tensor((batch_size, seq_len, num_query_heads, head_dim), dtype="float16"),
                R.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype="float16"),
                R.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype="float16"),
            ]
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

        new_mod = tvm.IRModule(new_mod, mod.type_definitions, mod.attrs, mod.global_infos)
        return new_mod

    return ir_module_pass
