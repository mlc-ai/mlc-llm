import argparse
import os
import pickle
import json
from typing import List
import json

import tvm
import tvm.testing
from tvm import relax

import mlc_llm
from mlc_llm import utils
from mlc_llm.relax_model import gpt_neox, llama, moss


def _parse_args():
    args = argparse.ArgumentParser()
    utils.argparse_add_common(args)
    args.add_argument("--max-seq-len", type=int, default=-1)
    args.add_argument("--target", type=str, default="auto")
    args.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to log database. Default: ./log_db/{model}",
    )
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--debug-load-script", action="store_true", default=False)

    args.add_argument(
        "--llvm-mingw",
        type=str,
        default="",
        help="/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows",
    )
    args.add_argument("--system-lib", action="store_true", default=False)

    parsed = args.parse_args()
    assert parsed.max_seq_len == -1 or parsed.max_seq_len > 0

    parsed.export_kwargs = {}
    parsed.lib_format = "so"

    parsed = _setup_model_path(parsed)

    parsed.db_path = parsed.db_path or os.path.join("log_db", parsed.model)

    utils.parse_target(parsed)
    utils.argparse_postproc_common(parsed)

    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )

    return parsed

def _setup_model_path(args):
    if args.model_path and args.hf_path:
        assert (args.model_path and not args.hf_path) or (args.hf_path and not args.model_path), "You cannot specify both a model path and a HF path. Please select one to specify."
    if args.model_path:
        validate_config(args)
        with open(os.path.join(args.model_path, "config.json")) as f:
            config = json.load(f)
            args.model = config["_name_or_path"].split("/")[-1]
    elif args.hf_path:
        args.model = args.hf_path.split("/")[-1]
        args.model_path = os.path.join(args.artifact_path, "models", args.model)
        if os.path.exists(args.model_path):
            print(f"Weights exist at {args.model_path}, skipping download.")
        else:
            os.makedirs(args.model_path, exist_ok=True)
            os.system("git lfs install")
            os.system(f"git clone https://huggingface.co/{args.hf_path} {args.model_path}")
            print(f"Downloaded weights to {args.model_path}")
        validate_config(args)
    else:
        raise ValueError(f"Please specify either the model_path or the hf_path.")
    print(f"Using model path {args.model_path}")
    return args

def validate_config(args):
    assert os.path.exists(os.path.join(args.model_path, "config.json")), "Model path must contain valid config file."
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
        assert ("model_type" in config) and ("_name_or_path" in config), "Invalid config format."
        assert config["model_type"] in utils.supported_model_types, f"Model type {config['model_type']} not supported."

def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_load_script(name, args):
    input_path = os.path.join(args.artifact_path, "debug", name)
    lib = {"__file__": input_path}
    exec(compile(open(input_path, "rb").read(), input_path, "exec"), lib, lib)
    return lib["Module"]


def debug_dump_shader(ex, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
        "opencl": ".cl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def mod_transform_before_build(
    mod: tvm.IRModule,
    model_params: List[tvm.nd.NDArray],
    args: argparse.Namespace,
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    model_names = [
        "encoding",
        "decoding",
        "create_kv_cache",
        "softmax_with_temperature",
    ]

    if args.quantization.mode != "no":
        mod = mlc_llm.transform.GroupQuantize(
            group_size=40 if args.quantization.mode.endswith("3") else 32,
            sym=args.quantization.sym,
            mode=args.quantization.mode,
            storage_nbit=args.quantization.storage_nbit,
            dtype=args.quantization.model_dtype,
        )(mod)
    mod = mlc_llm.transform.FuseTransposeMatmul()(mod)

    mod = relax.pipeline.get_pipeline()(mod)
    mod = mlc_llm.transform.FuseDecodeMatmulEwise(
        args.quantization.model_dtype, args.target_kind
    )(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)

    debug_dump_script(mod_transform, "mod_lift_params.py", args)

    new_params = utils.transform_params(mod_transform, model_params)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy


def dump_default_mlc_llm_config(args):
    config = dict()
    config["model_lib"] = f"{args.model}-{args.quantization.name}"
    config["local_id"] = f"{args.model}-{args.quantization.name}"
    config["conv_template"] = args.conv_template
    config["temperature"] = 0.7
    config["top_p"] = 0.95
    config["stream_interval"] = 2
    config["mean_gen_len"] = 128
    config["shift_fill_factor"] = 0.3
    dump_path = os.path.join(args.artifact_path, "mlc_llm_config.json")
    with open(dump_path, "w") as outfile:
        json.dump(config, outfile, indent=4)
    print(f"Finish exporting mlc_llm_config to {dump_path}")


def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    target_kind = args.target_kind
    debug_dump_script(mod_deploy, "mod_before_build.py", args)
    if target_kind != "cpu":
        from tvm import meta_schedule as ms

        if os.path.exists(args.db_path):
            db = ms.database.create(work_dir=args.db_path)
        else:
            db = ms.database.MemoryDatabase()
        with db, tvm.target.Target("apple/m1-gpu-restricted"):
            mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
            if args.target_kind == "android":
                mod_deploy = mlc_llm.dispatch.DispatchTIROperatorAdreno()(mod_deploy)
            mod_deploy = mlc_llm.dispatch.DispatchTIROperator(args.model_category)(
                mod_deploy
            )
            mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    if args.debug_load_script:
        mod_deploy = debug_load_script("mod_build_stage_debug.py", args)

    debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target, system_lib=args.system_lib)

    output_filename = (
        f"{args.model}-{args.quantization.name}-{target_kind}.{args.lib_format}"
    )

    debug_dump_shader(ex, f"{args.model}_{args.quantization.name}_{target_kind}", args)
    lib_path = os.path.join(args.artifact_path, output_filename)
    ex.export_library(lib_path, **args.export_kwargs)
    print(f"Finish exporting to {lib_path}")


def dump_split_tir(mod: tvm.IRModule):
    template = """
from tvm.script import ir as I
from tvm.script import tir as T

# fmt: off
{content}
# fmt: on
"""
    mod_static, mod_dynamic = utils.split_static_dynamic_tir(mod)
    static_path = os.path.join(ARGS.artifact_path, "debug", "mod_tir_static.py")
    dynamic_path = os.path.join(ARGS.artifact_path, "debug", "mod_tir_dynamic.py")
    print(f"Dump static shape TIR to {static_path}")
    with open(static_path, "w") as o_f:
        o_f.write(template.format(content=mod_static.script()))
    print(f"Dump dynamic shape TIR to {dynamic_path}")
    with open(dynamic_path, "w") as o_f:
        o_f.write(template.format(content=mod_dynamic.script()))


if __name__ == "__main__":
    ARGS = _parse_args()
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(
        ARGS.artifact_path, f"mod_cache_before_build_{ARGS.target_kind}.pkl"
    )
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    with open(os.path.join(ARGS.model_path, "config.json")) as f:
        config = json.load(f)
        if not use_cache:
            if ARGS.model_category == "llama":
                mod, params = llama.get_model(ARGS, config)
            elif ARGS.model_category == "gpt_neox":
                mod, params = gpt_neox.get_model(
                    ARGS.model,
                    ARGS.model_path,
                    ARGS.quantization.model_dtype,
                    config
                )
            elif ARGS.model_category == "moss":
                mod, params = moss.get_model(ARGS, config)
            else:
                raise ValueError(f"Model {ARGS.model} not supported")
            mod = mod_transform_before_build(mod, params, ARGS)
            with open(cache_path, "wb") as outfile:
                pickle.dump(mod, outfile)
            print(f"Save a cached module to {cache_path}.")
            utils.copy_tokenizer(ARGS)
        else:
            print(
                f"Load cached module from {cache_path} and skip tracing. "
                "You can use --use-cache=0 to retrace"
            )
            mod = pickle.load(open(cache_path, "rb"))
        dump_split_tir(mod)
        build(mod, ARGS)
        dump_default_mlc_llm_config(ARGS)
