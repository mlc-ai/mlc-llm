# pylint: disable=missing-docstring,invalid-name
import argparse
from typing import List, Tuple

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax


def argparse_add_common(args: argparse.ArgumentParser) -> None:
    args.add_argument(
        "--model",
        type=str,
        default="vicuna-v1-7b",
        choices=[
            "vicuna-v1-7b",
            "dolly-v2-3b",
            "dolly-v2-7b",
            "dolly-v2-12b",
            "stablelm-tuned-alpha-3b",
            "stablelm-tuned-alpha-7b",
            "moss-moon-003-sft",
        ],
    )
    args.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
    )


def argparse_postproc_common(args: argparse.Namespace) -> None:
    if hasattr(args, "device_name"):
        if args.device_name == "auto":
            if tvm.cuda().exist:
                args.device_name = "cuda"
            elif tvm.metal().exist:
                args.device_name = "metal"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")
    if args.model.startswith("vicuna-") or args.model.startswith("llama-"):
        from mlc_llm.relax_model import llama  # pylint: disable=import-outside-toplevel

        args.conv_template = "vicuna_v1.1"

    elif args.model.startswith("dolly-") or args.model.startswith("stablelm-"):
        from mlc_llm.relax_model import (  # pylint: disable=import-outside-toplevel
            gpt_neox,
        )

        if args.model.startswith("dolly-"):
            args.conv_template = "dolly"
        elif args.model.startswith("stablelm-"):
            args.conv_template = "stablelm"
    elif args.model.startswith("moss-"):
        from mlc_llm.relax_model import moss  # pylint: disable=import-outside-toplevel

        args.conv_template = "moss"
    else:
        raise ValueError(f"Model {args.model} not supportqed")


def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_name = None
    gv_names = [gv.name_hint for gv in mod.get_global_vars()]
    for name in model_names:
        if name + "_transform_params" in gv_names:
            transform_func_name = name + "_transform_params"
    assert transform_func_name is not None

    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint == transform_func_name:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination([transform_func_name])(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(model_names)(mod_deploy)

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule, model_params: List[tvm.nd.NDArray]
) -> List[tvm.nd.NDArray]:
    transform_func_name = None
    for gv, func in mod_transform.functions.items():
        if isinstance(func, relax.Function):
            transform_func_name = gv.name_hint
    assert transform_func_name is not None

    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())
    res = vm[transform_func_name](model_params)
    return res


def save_params(params: List[tvm.nd.NDArray], artifact_path: str) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    for i, nd in enumerate(params):
        param_dict[f"param_{i}"] = nd
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def build_model_from_log(relax_mod, target, log_dir):
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)
    return relax_mod
