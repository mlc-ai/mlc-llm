import json
from typing import List

from tvm import relax, te, topi


def create_metadata_func(
    bb: relax.BlockBuilder,
    model_name: str,
    max_window_size: int,
    stop_tokens: List[int],
    add_prefix_space: bool,
):
    metadata = json.dumps(
        {
            "model_name": model_name,
            "max_window_size": max_window_size,
            "stop_tokens": stop_tokens,
            "add_prefix_space": add_prefix_space,
        }
    )
    with bb.function("get_metadata", params=[]):
        bb.emit_func_output(relax.StringImm(metadata))


def create_shard_info_func(mod, param_manager, args, model_config):
    use_ft_quant = args.quantization.name in ["q4f16_ft", "q8f16_ft"]
    num_shards = args.num_shards
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    q_heads = model_config.num_attention_heads
    kv_heads = model_config.get_num_key_value_heads()

    # pylint: disable=invalid-name
    def shard_qkv_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial) * num_shards, int(red)
        a = te.placeholder((spatial, red), dtype=dtype)
        w = topi.reshape(a, (spatial // head_dim, head_dim, red))
        q = te.compute((q_heads, head_dim, red), lambda i, j, k: w[i, j, k])
        k = te.compute((kv_heads, head_dim, red), lambda i, j, k: w[q_heads + i, j, k])
        v = te.compute((kv_heads, head_dim, red), lambda i, j, k: w[q_heads + kv_heads + i, j, k])
        q = topi.reshape(q, (num_shards, q_heads // num_shards, head_dim, red))
        k = topi.reshape(k, (num_shards, kv_heads // num_shards, head_dim, red))
        v = topi.reshape(v, (num_shards, kv_heads // num_shards, head_dim, red))
        w = topi.concatenate((q, k, v), axis=1)
        w = topi.reshape(w, (num_shards, (q_heads + kv_heads * 2) // num_shards * head_dim, red))
        func = te.create_prim_func([a, w])
        return func
    
    def shard_ft_qkv_weight_scale(x: relax.TensorStructInfo):
        if x.ndim == 2:
            return shard_ft_qkv_weight(x)
        else:
            return shard_ft_qkv_scale(x)
    
    def shard_ft_qkv_weight(weight: relax.TensorStructInfo):
        (red, spatial), dtype = weight.shape, weight.dtype
        red, spatial = int(red), int(spatial) * num_shards
        head_dim = spatial // (q_heads + 2 * kv_heads)
        a = te.placeholder((red, spatial), dtype=dtype)
        w = topi.reshape(a, (red, spatial // head_dim, head_dim))
        q = te.compute((red, q_heads, head_dim), lambda i, j, k: w[i, j, k])
        k = te.compute((red, kv_heads, head_dim), lambda i, j, k: w[i, q_heads + j, k])
        v = te.compute((red, kv_heads, head_dim), lambda i, j, k: w[i, q_heads + kv_heads + j, k])
        q = topi.reshape(q, (red, num_shards, q_heads // num_shards, head_dim))
        k = topi.reshape(k, (red, num_shards, kv_heads // num_shards, head_dim))
        v = topi.reshape(v, (red, num_shards, kv_heads // num_shards, head_dim))
        w = topi.concatenate((q, k, v), axis=2)
        w = topi.reshape(w, (red, num_shards, (q_heads + kv_heads * 2) // num_shards * head_dim))
        w = topi.transpose(w, (1, 0, 2))
        func = te.create_prim_func([a, w])
        return func
    
    def shard_ft_qkv_scale(scale: relax.TensorStructInfo):
        (spatial,), dtype = scale.shape, scale.dtype
        spatial = int(spatial) * num_shards
        head_dim = spatial // (q_heads + 2 * kv_heads)
        a = te.placeholder((spatial,), dtype=dtype)
        w = topi.reshape(a, (spatial // head_dim, head_dim))
        q = te.compute((q_heads, head_dim), lambda i, j: w[i, j])
        k = te.compute((kv_heads, head_dim), lambda i, j: w[q_heads + i, j])
        v = te.compute((kv_heads, head_dim), lambda i, j: w[q_heads + kv_heads + i, j])
        q = topi.reshape(q, (num_shards, q_heads // num_shards, head_dim))
        k = topi.reshape(k, (num_shards, kv_heads // num_shards, head_dim))
        v = topi.reshape(v, (num_shards, kv_heads // num_shards, head_dim))
        w = topi.concatenate((q, k, v), axis=1)
        w = topi.reshape(w, (num_shards, (q_heads + kv_heads * 2) // num_shards * head_dim))
        func = te.create_prim_func([a, w])
        return func

    def shard_k_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red) * num_shards
        a = te.placeholder((spatial, red), dtype=dtype)
        w = topi.reshape(a, (spatial, num_shards, red // num_shards))
        w = topi.transpose(w, (1, 0, 2))
        func = te.create_prim_func([a, w])
        return func
    
    def shard_ft_k_weight(weight: relax.TensorStructInfo):
        (red, spatial), dtype = weight.shape, weight.dtype
        red, spatial = int(red) * num_shards, int(spatial)
        a = te.placeholder((red, spatial), dtype=dtype)
        w = topi.reshape(a, (num_shards, red // num_shards, spatial))
        func = te.create_prim_func([a, w])
        return func

    def shard_gate_up_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial) * num_shards, int(red)
        a = te.placeholder((spatial, red), dtype=dtype)
        g = te.compute((spatial // 2, red), lambda i, j: a[i, j])
        u = te.compute((spatial // 2, red), lambda i, j: a[spatial // 2 + i, j])
        g = topi.reshape(g, (num_shards, spatial // 2 // num_shards, red))
        u = topi.reshape(u, (num_shards, spatial // 2 // num_shards, red))
        w = topi.concatenate((g, u), axis=1)
        w = topi.reshape(w, (num_shards, spatial // num_shards, red))
        func = te.create_prim_func([a, w])
        return func
    
    def shard_ft_gate_up_weight_scale(x: relax.TensorStructInfo):
        if x.ndim == 2:
            return shard_ft_gate_up_weight(x)
        else:
            return shard_ft_gate_up_scale(x)
    
    def shard_ft_gate_up_weight(weight: relax.TensorStructInfo):
        (red, spatial), dtype = weight.shape, weight.dtype
        red, spatial = int(red), int(spatial) * num_shards
        a = te.placeholder((red, spatial), dtype=dtype)
        g = te.compute((red, spatial // 2), lambda i, j: a[i, j])
        u = te.compute((red, spatial // 2), lambda i, j: a[i, spatial // 2 + j])
        g = topi.reshape(g, (red, num_shards, spatial // 2 // num_shards))
        u = topi.reshape(u, (red, num_shards, spatial // 2 // num_shards))
        w = topi.concatenate((g, u), axis=2)
        w = topi.reshape(w, (red, num_shards, spatial // num_shards))
        w = topi.transpose(w, (1, 0, 2))
        func = te.create_prim_func([a, w])
        return func
    
    def shard_ft_gate_up_scale(weight: relax.TensorStructInfo):
        (spatial,), dtype = weight.shape, weight.dtype
        spatial = int(spatial) * num_shards
        a = te.placeholder((spatial,), dtype=dtype)
        g = te.compute((spatial // 2,), lambda i: a[i])
        u = te.compute((spatial // 2,), lambda i: a[spatial // 2 + i])
        g = topi.reshape(g, (num_shards, spatial // 2 // num_shards))
        u = topi.reshape(u, (num_shards, spatial // 2 // num_shards))
        w = topi.concatenate((g, u), axis=1)
        w = topi.reshape(w, (num_shards, spatial // num_shards))
        func = te.create_prim_func([a, w])
        return func

    # pylint: enable=invalid-name

    shard_info_dict = {}
    shard_funcs = {}

    def add_to_shard_info(param_name: str, func_name: str):
        func = shard_funcs[func_name]
        buffer = func.buffer_map[func.params[-1]]
        shape = [int(i) for i in buffer.shape]
        dtype = str(buffer.dtype)
        shard_info_dict[param_name] = [(func_name, [shape, dtype])]

    q_params = param_manager.get_quantized_param_info("prefill").fields
    for _, param in param_manager.params.items():
        if param.shard_strategy is None:
            pass
        elif param.shard_strategy == "shard_qkv":
            for i, weight in enumerate(param_manager.param2qrange[param]):
                name = f"shard_qkv_{i}"
                if name not in shard_funcs:
                    if use_ft_quant:
                        shard_funcs[name] = shard_ft_qkv_weight_scale(q_params[weight])
                    else:
                        shard_funcs[name] = shard_qkv_weight_scale(q_params[weight])                  
                add_to_shard_info(f"param_{weight}", name)
        elif param.shard_strategy == "shard_mlp_k":
            for i, weight in enumerate(param_manager.param2qrange[param]):
                name = f"shard_mlp_k_{i}"
                if name not in shard_funcs:
                    if use_ft_quant:    
                        if q_params[weight].ndim == 1:
                            # replicate
                            continue
                        else:
                            shard_funcs[name] = shard_ft_k_weight(q_params[weight])
                    else:
                        shard_funcs[name] = shard_k_weight_scale(q_params[weight])
                add_to_shard_info(f"param_{weight}", name)
        elif param.shard_strategy == "shard_o_proj_k":
            for i, weight in enumerate(param_manager.param2qrange[param]):
                name = f"shard_o_proj_k_{i}"
                if name not in shard_funcs:
                    if use_ft_quant:    
                        if q_params[weight].ndim == 1:
                            # replicate
                            continue
                        else:
                            shard_funcs[name] = shard_ft_k_weight(q_params[weight])
                    else:
                        shard_funcs[name] = shard_k_weight_scale(q_params[weight])
                add_to_shard_info(f"param_{weight}", name)
        elif param.shard_strategy == "shard_gate_up":
            for i, weight in enumerate(param_manager.param2qrange[param]):
                name = f"shard_gate_up_{i}"
                if name not in shard_funcs:
                    if use_ft_quant:
                        shard_funcs[name] = shard_ft_gate_up_weight_scale(q_params[weight])
                    else:
                        shard_funcs[name] = shard_gate_up_weight_scale(q_params[weight])
                add_to_shard_info(f"param_{weight}", name)
        else:
            raise NotImplementedError(f"Shard strategy not implemented: {param.shard_strategy}")
    for name, func in shard_funcs.items():
        func = func.with_attr({"global_symbol": name})
        mod[name] = func
    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
    with bb.function("get_shard_info", params=[]):
        bb.emit_func_output(relax.StringImm(json.dumps(shard_info_dict)))
    mod["get_shard_info"] = bb.get()["get_shard_info"]
