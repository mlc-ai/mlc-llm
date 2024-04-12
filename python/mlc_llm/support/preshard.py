"""Functions for pre-sharding weights"""
import logging
from typing import Any, Callable, Dict, Sequence, Tuple, List

import tvm
from tvm import IRModule
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn
from tvm.runtime import Device, NDArray
from tvm.target import Target
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger("preshard")


def _sharded_param_name(param_name, worker_id):
    return f"{param_name}_shard-{worker_id}"


def _update_quantize_map(
    quantize_map: Any,
    named_params: Dict[str, nn.Parameter],
    mlc_name: str,
    tensor_parallel_shards: int,
):
    param_names: List[str] = [mlc_name]

    if mlc_name in quantize_map.param_map:
        # the parameter is quantized
        quantized_params = quantize_map.param_map[mlc_name]
        param_names = quantized_params
        quantize_func = quantize_map.map_func[mlc_name]

        for worker_id in range(tensor_parallel_shards):
            sharded_mlc_name = _sharded_param_name(mlc_name, worker_id)
            quantize_map.param_map[sharded_mlc_name] = [
                _sharded_param_name(param_name, worker_id) for param_name in quantized_params
            ]
            quantize_map.map_func[sharded_mlc_name] = quantize_func

    for param_name in param_names:
        param = named_params.pop(param_name)
        for worker_id in range(tensor_parallel_shards):
            named_params[_sharded_param_name(param_name, worker_id)] = param

def create_tir_shard_func(
    param: nn.Parameter,
    tensor_parallel_shards: int,
) -> Tuple[tvm.tir.PrimFunc, List[tvm.tir.PrimExpr], List[tvm.tir.PrimExpr]]:
    shard_strategy = param.attrs.get("shard_strategy", None)
    tir_func = shard_strategy.gen_tir(shards=tensor_parallel_shards, weight=param)
    tir_func = tir_func.without_attr("global_symbol")
    weight_shape = list(param.shape)
    weight_shape[shard_strategy.dim] = weight_shape[shard_strategy.dim] * tensor_parallel_shards
    sharded_weight_shape = [tensor_parallel_shards, *param.shape]

    return tir_func, weight_shape, sharded_weight_shape

def create_shard_func(
    bb: relax.BlockBuilder,
    param: nn.Parameter,
    tensor_parallel_shards: int,
    do_split: bool = True,
):  # pylint: disable=too-many-locals

    # generate tir shard function
    tir_func, weight_shape, sharded_weight_shape = create_tir_shard_func(param, tensor_parallel_shards)
    shard_strategy = param.attrs.get("shard_strategy", None)
    # add tir shard function to the IRModule
    tir_gvar = bb.add_func(tir_func, func_name=f"{shard_strategy.name}_tir")
    # create relax function that
    #     1. shard weight with tir shard function, result: [num_shards, *sharded_weight_shape]
    #     2. split the sharded weight along dim 0, result: num_shards * [1, *sharded_weight_shape]
    #     3. squeeze the 0th-dim of all shards, result: num_shards * [*sharded_weight_shape]
    weight_var = relax.Var("weight", relax.TensorStructInfo(weight_shape, param.dtype))

    with bb.function(name=shard_strategy.name, params=[weight_var]):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_tir(
                    tir_gvar,
                    weight_var,
                    out_sinfo=relax.TensorStructInfo(sharded_weight_shape, param.dtype),
                )
            )
            if do_split:
                lv1 = bb.emit(
                    relax.op.split(lv0, indices_or_sections=tensor_parallel_shards, axis=0)
                )
                output_vars = []
                for i in range(tensor_parallel_shards):
                    lvi = bb.emit(relax.TupleGetItem(lv1, i))
                    squeezed_lvi = bb.emit(relax.op.squeeze(lvi, 0))
                    output_vars.append(squeezed_lvi)
                gv = bb.emit_output(output_vars)
            else:
                gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    return tir_gvar, weight_shape, sharded_weight_shape


def _compile_shard_funcs(mod: IRModule, device: Device):
    target = Target.from_device(device)
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)
    return vm


def apply_preshard(
    quantize_map: Any,
    named_params: Dict[str, nn.Parameter],
    tensor_parallel_shards: int,
    args: Any,
) -> Tuple[Dict[str, nn.Parameter], Dict[str, Callable[[NDArray], Sequence[NDArray]]]]:
    """Apply pre-sharding to the named parameters.

    Parameters
    ----------
    named_params : Dict[str, nn.Parameter]
        The named parameters of the model. If the model is quantized, the named parameters should
        the state dictionary of the quantized model.
    tensor_parallel_shards : int
        The number of tensor parallel shards.
    args : Any
        The parsed arguments of weight conversion.

    Returns
    -------
    Tuple[Dict[str, nn.Parameter], Dict[str, Callable[[NDArray], Sequence[NDArray]]]
        The updated named parameters and the mapping from parameter name to the shard function.
    """

    # Update quantize_map and named_params, create shard functions based on shard strategies.
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)

    bb = relax.BlockBuilder()
    param_to_shard_func = {}
    shard_func_names = set()
    has_shard_strategy = False
    for name, param in model.state_dict().items():
        shard_strategy = param.attrs.get("shard_strategy", None)
        if shard_strategy is not None:
            has_shard_strategy = True
            _update_quantize_map(quantize_map, named_params, name, tensor_parallel_shards)

            # create shard functions
            param_to_shard_func[name] = shard_strategy.name
            if shard_strategy.name not in shard_func_names:
                if not isinstance(shard_strategy, tp.ShardScalar):
                    create_shard_func(bb, param, tensor_parallel_shards)
                    shard_func_names.add(shard_strategy.name)

    if not has_shard_strategy:
        logger.warning(
            "No parameters with 'shard_strategy' found."
            "At least one parameter must have a 'shard_strategy' for presharding. "
            "The model will continue to convert weights in a non-presharded manner."
        )

    mod = bb.finalize()
    vm = _compile_shard_funcs(mod, args.device)

    for name in param_to_shard_func:
        param_to_shard_func[name] = vm[param_to_shard_func[name]]
    return param_to_shard_func
