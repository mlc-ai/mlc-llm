import json
from typing import Dict, List, Optional

import mlc_llm
import tvm
from tvm import relax, te, tir, topi


def create_metadata_func(
    bb: relax.BlockBuilder,
    model_name: str,
    max_window_size: int,
    stop_tokens: List[int],
    add_prefix_space: bool,
    prefill_chunk_size: int = -1,
    sliding_window: int = -1,
):
    metadata = json.dumps(
        {
            "model_name": model_name,
            "max_window_size": max_window_size,
            "stop_tokens": stop_tokens,
            "add_prefix_space": add_prefix_space,
            "prefill_chunk_size": prefill_chunk_size,
            "sliding_window": sliding_window,
        }
    )
    with bb.function("get_metadata", params=[]):
        bb.emit_func_output(relax.StringImm(metadata))


def _get_shard_strategies(
    model_config, num_shards: int, param_shape_is_already_sharded: bool
) -> Dict[str, tvm.tir.PrimFunc]:
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    q_heads = model_config.num_attention_heads
    kv_heads = model_config.get_num_key_value_heads()

    # pylint: disable=invalid-name
    def shard_qkv_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red)
        if param_shape_is_already_sharded:
            spatial *= num_shards
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
        spatial, red = int(spatial), int(red)
        if param_shape_is_already_sharded:
            red *= num_shards
        a = te.placeholder((spatial, red), dtype=dtype)
        w = topi.reshape(a, (spatial, num_shards, red // num_shards))
        w = topi.transpose(w, (1, 0, 2))
        func = te.create_prim_func([a, w])
        return func

    def shard_gate_up_weight_scale(weight: relax.TensorStructInfo):
        (spatial, red), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red)
        if param_shape_is_already_sharded:
            spatial *= num_shards
        a = te.placeholder((spatial, red), dtype=dtype)
        g = te.compute((spatial // 2, red), lambda i, j: a[i, j])
        u = te.compute((spatial // 2, red), lambda i, j: a[spatial // 2 + i, j])
        g = topi.reshape(g, (num_shards, spatial // 2 // num_shards, red))
        u = topi.reshape(u, (num_shards, spatial // 2 // num_shards, red))
        w = topi.concatenate((g, u), axis=1)
        w = topi.reshape(w, (num_shards, spatial // num_shards, red))
        func = te.create_prim_func([a, w])
        return func

    def moe_shard_k_weight_scale(weight: relax.TensorStructInfo):
        (num_experts, red, spatial), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red)
        if param_shape_is_already_sharded:
            red *= num_shards
        a = te.placeholder((num_experts, red, spatial), dtype=dtype)
        w = topi.reshape(a, (num_experts, num_shards, red // num_shards, spatial))
        w = topi.transpose(w, (1, 0, 2, 3))
        func = te.create_prim_func([a, w])
        return func

    def moe_shard_gate_up_weight_scale(weight: relax.TensorStructInfo):
        (num_experts, red, spatial), dtype = weight.shape, weight.dtype
        spatial, red = int(spatial), int(red)
        if param_shape_is_already_sharded:
            spatial *= num_shards
        a = te.placeholder((num_experts, red, spatial), dtype=dtype)
        g = te.compute((num_experts, red, spatial // 2), lambda e, i, j: a[e, i, j])
        u = te.compute((num_experts, red, spatial // 2), lambda e, i, j: a[e, i, spatial // 2 + j])
        g = topi.reshape(g, (num_experts, red, num_shards, spatial // 2 // num_shards))
        u = topi.reshape(u, (num_experts, red, num_shards, spatial // 2 // num_shards))
        w = topi.concatenate((g, u), axis=3)
        w = topi.reshape(w, (num_experts, red, num_shards, spatial // num_shards))
        w = topi.transpose(w, (2, 0, 1, 3))
        func = te.create_prim_func([a, w])
        return func


    # pylint: enable=invalid-name

    return {
        "shard_qkv": shard_qkv_weight_scale,
        "shard_mlp_k": shard_k_weight_scale,
        "shard_o_proj_k": shard_k_weight_scale,
        "shard_gate_up": shard_gate_up_weight_scale,
        "moe_shard_mlp_k": moe_shard_k_weight_scale,
        "moe_shard_gate_up": moe_shard_gate_up_weight_scale,
    }


def _get_shard_strategies_ft(
    model_config, num_shards: int, param_shape_is_already_sharded: bool
) -> Dict[str, tvm.tir.PrimFunc]:
    q_heads = model_config.num_attention_heads
    kv_heads = model_config.get_num_key_value_heads()

    def shard_qkv_weight_scale(x: relax.TensorStructInfo):
        (red, spatial), dtype = x.shape, x.dtype
        red, spatial = int(red), int(spatial)
        if param_shape_is_already_sharded:
            spatial *= num_shards
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

    def shard_k_weight(weight: relax.TensorStructInfo):
        (red, spatial), dtype = weight.shape, weight.dtype
        red, spatial = int(red), int(spatial)
        if param_shape_is_already_sharded:
            red *= num_shards
        a = te.placeholder((red, spatial), dtype=dtype)
        w = topi.reshape(a, (num_shards, red // num_shards, spatial))
        func = te.create_prim_func([a, w])
        return func

    def shard_gate_up_weight_scale(x: relax.TensorStructInfo):
        (red, spatial), dtype = x.shape, x.dtype
        red, spatial = int(red), int(spatial)
        if param_shape_is_already_sharded:
            spatial *= num_shards
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

    return {
        "shard_qkv": shard_qkv_weight_scale,
        "shard_mlp_k": shard_k_weight,
        "shard_o_proj_k": shard_k_weight,
        "shard_gate_up": shard_gate_up_weight_scale,
    }


def create_shard_info_func(param_manager, args, model_config) -> tvm.IRModule:
    shard_strategy_to_func = _get_shard_strategies(
        model_config,
        num_shards=args.num_shards,
        param_shape_is_already_sharded=args.build_model_only,
    )

    shard_info_dict = {}
    shard_funcs = {}

    def add_to_shard_info(param_name: str, func_name: Optional[str]):
        shard_info = []
        if func_name is not None:
            func = shard_funcs[func_name]
            buffer = func.buffer_map[func.params[-1]]
            shape = [int(i) for i in buffer.shape]
            dtype = str(buffer.dtype)
            shard_info.append((func_name, [shape, dtype]))

        shard_info_dict[param_name] = shard_info

    q_params = param_manager.get_quantized_param_info("prefill").fields
    for _, param in param_manager.params.items():
        if param.shard_strategy is None:
            pass
        elif param.shard_strategy in shard_strategy_to_func:
            for i, weight in enumerate(param_manager.param2qrange[param]):
                if args.use_presharded_weights:
                    sharding_func_name = None
                else:
                    sharding_func_name = f"{param.shard_strategy}_{i}"
                    if sharding_func_name not in shard_funcs:
                        shard_funcs[sharding_func_name] = shard_strategy_to_func[
                            param.shard_strategy
                        ](q_params[weight])
                add_to_shard_info(f"param_{weight}", sharding_func_name)
        else:
            raise NotImplementedError(f"Shard strategy not implemented: {param.shard_strategy}")

    bb = relax.BlockBuilder()  # pylint: disable=invalid-name

    for name, func in shard_funcs.items():
        func = func.with_attr({"global_symbol": name})
        bb.add_func(func, name)

    with bb.function("get_shard_info", params=[]):
        bb.emit_func_output(relax.StringImm(json.dumps(shard_info_dict)))

    return bb.get()


def create_shard_transformation_func(param_manager, args, model_config) -> tvm.IRModule:
    use_ft_quant = args.quantization.name in ["q4f16_ft", "q8f16_ft", "q4f16_ft_group", "q8f16_ft_group"]

    if use_ft_quant:
        shard_strategy_to_func = _get_shard_strategies_ft(
            model_config,
            num_shards=args.num_shards,
            param_shape_is_already_sharded=args.build_model_only,
        )
    else:
        shard_strategy_to_func = _get_shard_strategies(
            model_config,
            num_shards=args.num_shards,
            param_shape_is_already_sharded=args.build_model_only,
        )

    q_params = param_manager.get_quantized_param_info("prefill").fields

    # The order of the quantized parameters must be preserved.
    # Therefore, we need to loop over q_params and look up information
    # as needed, rather than looping over original parameters and
    # looking up the quantized parameters as needed.
    orig_param_lookup = {}
    for param in param_manager.params_in_func["prefill"]:
        qrange = param_manager.param2qrange[param]
        for i_orig_part, i_qparam in enumerate(qrange):
            orig_param_lookup[i_qparam] = (
                param,
                i_orig_part,
                len(qrange),
            )

    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
    with bb.function("transform_params"):
        rank = tir.SizeVar("rank", "int64")
        # TODO(Lunderberg): Support primitive inputs to relax
        # functions.  Currently, using a PrimStructInfo as the
        # argument results in an error thrown during
        # `vm_shape_lower.cc`, due to BindParams failing to replace
        # the symbolic variable "rank" when defined in a R.PrimValue.
        #
        # rank_arg = relax.Var("rank", relax.PrimStructInfo(value=rank))
        rank_arg = relax.Var("rank_arg", relax.ShapeStructInfo([rank]))

        args = [rank_arg]
        output = []

        for i_qparam, qparam_sinfo in enumerate(q_params):
            param, i_orig_part, num_orig_parts = orig_param_lookup[i_qparam]

            if isinstance(param.quant_spec, mlc_llm.quantization.NoQuantizationSpec):
                arg_name = param.name
            elif num_orig_parts == 1:
                arg_name = f"{param.name}.quantized"
            else:
                arg_name = f"{param.name}.quantized_{i_orig_part}"

            arg = relax.Var(arg_name, qparam_sinfo)

            if param.shard_strategy is None or (
                use_ft_quant
                and param.shard_strategy in ["shard_mlp_k", "shard_o_proj_k"]
                and qparam_sinfo.shape[0] == 1
            ):
                sharded = arg
            else:
                strategy_func = shard_strategy_to_func[param.shard_strategy](
                    qparam_sinfo
                ).without_attr("global_symbol")

                strategy_gvar = bb.add_func(
                    strategy_func,
                    func_name=f"{arg_name}.sharding_func",
                )

                # TODO(Lunderberg): Write the strategies as relax
                # functions, so the sharded shapes can be inferred.
                reordered_buffer = strategy_func.buffer_map[strategy_func.params[-1]]
                reordered_sinfo = relax.TensorStructInfo(
                    reordered_buffer.shape, reordered_buffer.dtype
                )
                reordered = relax.op.call_tir(
                    strategy_gvar, relax.Tuple([arg]), out_sinfo=reordered_sinfo
                )

                # TODO(Lunderberg): Allow relax.PrimValue as the index
                # in a TupleGetItem.  This would allow all of the
                # splits to be generated at once in the merged
                # function, and could be optimized to an in-place view.
                #
                # split = relax.op.split(reordered, indices_or_sections=num_shards, axis=0)[rank]
                split = relax.op.strided_slice(
                    reordered,
                    axes=[0],
                    begin=[rank],
                    end=[rank + 1],
                    assume_inbound=True,
                )

                sharded = relax.op.squeeze(split, axis=0)

            args.append(arg)
            output.append(sharded)

        with bb.dataflow():
            gv = bb.emit_output(output)
        bb.emit_func_output(output=gv, params=args)

    return bb.get()
