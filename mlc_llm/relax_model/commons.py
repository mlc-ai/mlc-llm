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
    hidden_size = model_config.hidden_size
    num_shards = args.num_shards
    head_dim = hidden_size // model_config.num_attention_heads
    q_heads = model_config.num_attention_heads
    kv_heads = model_config.num_key_value_heads

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

    def shard_k_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red) * num_shards
        a = te.placeholder((spatial, red), dtype=dtype)
        w = topi.reshape(a, (spatial, num_shards, red // num_shards))
        w = topi.transpose(w, (1, 0, 2))
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

    # pylint: enable=invalid-name

    shard_info_dict = {}
    shard_funcs = {}

    def add_to_shard_info(i: int, func_name: str):
        func = shard_funcs[func_name]
        buffer = func.buffer_map[func.params[-1]]
        shape = [int(i) for i in buffer.shape]
        dtype = str(buffer.dtype)
        shard_info_dict[f"param_{i}"] = [(func_name, [shape, dtype])]

    q_params = param_manager.get_quantized_param_info("prefill").fields
    for _, param in param_manager.params.items():
        if param.shard_strategy is None:
            pass
        elif param.shard_strategy == "shard_qkv":
            weight, scale = param_manager.param2qrange[param]
            if "shard_qkv_weight" not in shard_funcs:
                shard_funcs["shard_qkv_weight"] = shard_qkv_weight_scale(q_params[weight])
            if "shard_qkv_scale" not in shard_funcs:
                shard_funcs["shard_qkv_scale"] = shard_qkv_weight_scale(q_params[scale])
            add_to_shard_info(weight, "shard_qkv_weight")
            add_to_shard_info(scale, "shard_qkv_scale")
        elif param.shard_strategy == "shard_mlp_k":
            weight, scale = param_manager.param2qrange[param]
            if "shard_mlp_k_weight" not in shard_funcs:
                shard_funcs["shard_mlp_k_weight"] = shard_k_weight_scale(q_params[weight])
            if "shard_mlp_k_scale" not in shard_funcs:
                shard_funcs["shard_mlp_k_scale"] = shard_k_weight_scale(q_params[scale])
            add_to_shard_info(weight, "shard_mlp_k_weight")
            add_to_shard_info(scale, "shard_mlp_k_scale")
        elif param.shard_strategy == "shard_o_proj_k":
            weight, scale = param_manager.param2qrange[param]
            if "shard_o_proj_k_weight" not in shard_funcs:
                shard_funcs["shard_o_proj_k_weight"] = shard_k_weight_scale(q_params[weight])
            if "shard_o_proj_k_scale" not in shard_funcs:
                shard_funcs["shard_o_proj_k_scale"] = shard_k_weight_scale(q_params[scale])
            add_to_shard_info(weight, "shard_o_proj_k_weight")
            add_to_shard_info(scale, "shard_o_proj_k_scale")
        elif param.shard_strategy == "shard_gate_up":
            weight, scale = param_manager.param2qrange[param]
            if "shard_gate_up_weight" not in shard_funcs:
                shard_funcs["shard_gate_up_weight"] = shard_gate_up_weight_scale(q_params[weight])
            if "shard_gate_up_scale" not in shard_funcs:
                shard_funcs["shard_gate_up_scale"] = shard_gate_up_weight_scale(q_params[scale])
            add_to_shard_info(weight, "shard_gate_up_weight")
            add_to_shard_info(scale, "shard_gate_up_scale")
        else:
            raise NotImplementedError(f"Shard strategy not implemented: {param.shard_strategy}")
    for name, func in shard_funcs.items():
        func = func.with_attr({"global_symbol": name})
        mod[name] = func
    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
    with bb.function("get_shard_info", params=[]):
        bb.emit_func_output(relax.StringImm(json.dumps(shard_info_dict)))
    mod["get_shard_info"] = bb.get()["get_shard_info"]
