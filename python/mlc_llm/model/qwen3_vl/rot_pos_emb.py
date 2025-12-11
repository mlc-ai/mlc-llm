from tvm.relax.frontend.nn.op import wrap_nested
from tvm.relax.op import strided_slice as relax_strided_slice
from tvm.relax.frontend.nn import Tensor

def op_strided_slice(x, axes, begin, end):
    return wrap_nested(relax_strided_slice(x._expr, axes, begin, end), name="strided_slice")


from tvm.script import tir as T

from tvm import relax as rx

def _wrap_op(f, *args):
    args = [x._expr if isinstance(x, Tensor) else x for x in args]
    return wrap_nested(f(*args), name=f.__name__)

def op_power(a, b): return _wrap_op(rx.op.power, a, b)


@T.prim_func
def populate_pos_ids_tir(
    var_grid_thw: T.handle,
    var_merge_size: T.handle,
    var_pos_ids: T.handle,
):
    n = T.int64()
    grid_thw = T.match_buffer(var_grid_thw, (n, 3), "int64")
    merge_size_buf = T.match_buffer(var_merge_size, (), "int64")
    
    # We match output buffer. The shape is (m, 2) where m is total tokens.
    # Since m is dynamic and not passed as explicit arg, we declare it.
    m = T.int64()
    pos_ids = T.match_buffer(var_pos_ids, (m, 2), "int64")
    
    merge_size = merge_size_buf[()]
    
    with T.block("root"):
        offset = T.alloc_buffer((1,), "int64")
        offset[0] = 0
        for i in range(n):
            t = grid_thw[i, 0]
            h = grid_thw[i, 1]
            w = grid_thw[i, 2]
            
            merged_h = h // merge_size
            merged_w = w // merge_size
            
            # total tokens for this image
            current_tokens = t * h * w
            
            for f in range(t):
                for bh in range(merged_h):
                    for bw in range(merged_w):
                        for mh in range(merge_size):
                            for mw in range(merge_size):
                                
                                internal_idx = f * (h * w) + bh * (merged_w * merge_size * merge_size) + bw * (merge_size * merge_size) + mh * merge_size + mw
                                output_idx = offset[0] + internal_idx
                                
                                # Safety check could be added, but T.Buffer access usually assumes in-bound
                                pos_ids[output_idx, 0] = bh * merge_size + mh
                                pos_ids[output_idx, 1] = bw * merge_size + mw
            
            offset[0] = offset[0] + current_tokens



@T.prim_func
def compute_freq_table_tir(
    var_max_hw: T.handle,
    var_inv_freq: T.handle,
    var_freq_table: T.handle,
):
    max_hw_buf = T.match_buffer(var_max_hw, (), "int64")
    d = T.int64()
    inv_freq = T.match_buffer(var_inv_freq, (d,), "float32")
    m = T.int64() # dynamic row
    # We match output with dynamic m and d
    freq_table = T.match_buffer(var_freq_table, (m, d), "float32")
    
    max_hw = max_hw_buf[()]
    
    for i in range(max_hw):
        for j in range(d):
            freq_table[i, j] = T.cast(i, "float32") * inv_freq[j]
