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

@T.prim_func
def fast_pos_embed_interpolate_tir(
    var_grid_thw: T.handle,
    var_pos_embed: T.handle,
    var_num_grid_per_side: T.handle,
    var_spatial_merge_size: T.handle,
    var_output: T.handle,
):
    n = T.int64()
    grid_thw = T.match_buffer(var_grid_thw, (n, 3), "int64")
    
    num_pos_emb = T.int64()
    hidden_size = T.int64()
    pos_embed = T.match_buffer(var_pos_embed, (num_pos_emb, hidden_size), "float32")
    
    num_grid_buf = T.match_buffer(var_num_grid_per_side, (), "int64")
    merge_size_buf = T.match_buffer(var_spatial_merge_size, (), "int64")
    
    total_tokens = T.int64()
    output = T.match_buffer(var_output, (total_tokens, hidden_size), "float32")
    
    num_grid_per_side = num_grid_buf[()]
    merge_size = merge_size_buf[()]
    
    with T.block("root"):
        offset = T.alloc_buffer((1,), "int64")
        offset[0] = 0
        
        for i in range(n):
            t = grid_thw[i, 0]
            h = grid_thw[i, 1]
            w = grid_thw[i, 2]
            
            # Derived dimensions
            merged_h = h // merge_size
            merged_w = w // merge_size
            
            current_tokens = t * h * w
            
            # --- Bilinear Interpolation Logic ---
            # We iterate over the *output* structure because that's what we need to fill.
            # But the output structure is permuted: (t, merged_h, merged_w, merge_size, merge_size)
            # flattened into (total_tokens, hidden_size)
            
            for f in range(t):
                for bh in range(merged_h):
                    for bw in range(merged_w):
                        for mh in range(merge_size):
                            for mw in range(merge_size):
                                # Logic to map back to original h, w indices for interpolation
                                # The output follows the structure:
                                # [t, h, w] implicitly but reordered.
                                # The original pixel index (oh, ow) inside the image (h, w) for this specific token:
                                oh = bh * merge_size + mh
                                ow = bw * merge_size + mw
                                
                                # Now compute bilinear interpolation for (oh, ow)
                                # h_idxs calculation:
                                # h_idx = 0 + oh * (num_grid_per_side - 1) / (h - 1)
                                
                                # Use float calculations for interpolation
                                h_frac = T.cast(oh, "float32") * T.cast(num_grid_per_side - 1, "float32") / T.max(T.cast(h - 1, "float32"), 1.0)
                                w_frac = T.cast(ow, "float32") * T.cast(num_grid_per_side - 1, "float32") / T.max(T.cast(w - 1, "float32"), 1.0)
                                
                                h_floor = T.floor(h_frac)
                                w_floor = T.floor(w_frac)
                                
                                # Clip ceil to max(num_grid - 1), though math says it shouldn't exceed much
                                # In Python: (int(h) + 1).clip(max=num_grid - 1)
                                h_ceil = T.min(h_floor + 1.0, T.cast(num_grid_per_side - 1, "float32"))
                                w_ceil = T.min(w_floor + 1.0, T.cast(num_grid_per_side - 1, "float32"))
                                
                                dh = h_frac - h_floor
                                dw = w_frac - w_floor
                                
                                # Convert to int indices for lookup
                                h0 = T.cast(h_floor, "int64")
                                h1 = T.cast(h_ceil, "int64")
                                w0 = T.cast(w_floor, "int64")
                                w1 = T.cast(w_ceil, "int64")
                                
                                # Base indices in pos_embed grid (flattened)
                                # pos_embed is (num_grid_per_side * num_grid_per_side, hidden_size)
                                # logical grid is (num_grid, num_grid)
                                # idx = r * num_grid + c
                                
                                idx00 = h0 * num_grid_per_side + w0
                                idx01 = h0 * num_grid_per_side + w1
                                idx10 = h1 * num_grid_per_side + w0
                                idx11 = h1 * num_grid_per_side + w1
                                
                                # Calculate output index
                                # Output is flattened tensor. 
                                # We fill it sequentially in the order of loops: f, bh, bw, mh, mw
                                # This order matches exactly the permute logic in PyTorch:
                                # (t, h//ms, ms, w//ms, ms) -> (t, h//ms, w//ms, ms, ms)
                                
                                flattened_idx = (
                                    f * (merged_h * merged_w * merge_size * merge_size) + 
                                    bh * (merged_w * merge_size * merge_size) + 
                                    bw * (merge_size * merge_size) + 
                                    mh * merge_size + 
                                    mw
                                )
                                out_idx = offset[0] + flattened_idx

                                # Perform interpolation for each hidden dim
                                for d_idx in range(hidden_size):
                                    v00 = pos_embed[idx00, d_idx]
                                    v01 = pos_embed[idx01, d_idx]
                                    v10 = pos_embed[idx10, d_idx]
                                    v11 = pos_embed[idx11, d_idx]
                                    
                                    val = (
                                        v00 * (1.0 - dh) * (1.0 - dw) +
                                        v01 * (1.0 - dh) * dw +
                                        v10 * dh * (1.0 - dw) +
                                        v11 * dh * dw
                                    )
                                    
                                    output[out_idx, d_idx] = val

            offset[0] = offset[0] + current_tokens
