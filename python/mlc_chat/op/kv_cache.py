"""Operators for KV cache manipulations."""
# pylint: disable=too-many-locals
from tvm.script import tir as T


def kv_cache_transpose_append(num_key_value_heads, head_dim, dtype):
    """Return the TIR function that appends new k/v data to PagedKVCache."""

    @T.prim_func
    def tir_kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        ntoken = T.SizeVar("ntoken", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()

        pages = T.match_buffer(
            var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype
        )
        k_data = T.match_buffer(var_k_data, (ntoken, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_key_value_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32")

        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(
                    pages[
                        position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf
                    ]
                )
                position: T.int32 = position_map[vgpos]  # type: ignore
                pages[
                    T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf
                ] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(
                    pages[
                        position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf
                    ]
                )
                position: T.int32 = position_map[vgpos]  # type: ignore
                pages[
                    T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf
                ] = v_data[vgpos, vh, vf]

    return tir_kv_cache_transpose_append


def kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    @T.prim_func
    def tir_kv_cache_debug_get_kv(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        seqlen = T.SizeVar("seqlen", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()

        pages = T.match_buffer(
            var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype
        )
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32")
        k_data = T.match_buffer(
            var_k_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype
        )
        v_data = T.match_buffer(
            var_v_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype
        )

        for p, h, d in T.grid(seqlen, num_key_value_heads, head_dim):
            with T.block("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(
                    position_map[vp],
                    pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd],
                )
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp]
                k_data[layer_id, vp, vh, vd] = pages[
                    T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd
                ]
                v_data[layer_id, vp, vh, vd] = pages[
                    T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd
                ]

    return tir_kv_cache_debug_get_kv
