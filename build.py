# pylint: disable=missing-docstring
import argparse
import json
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import tvm
from tvm import meta_schedule as ms
from tvm import relax

import mlc_llm
from mlc_llm import utils
from mlc_llm.relax_model import gpt_neox, llama, minigpt, moss, rwkv


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="auto",
        help='The name of the model to build. If it is "auto", we will automatically set the '
        'model name according to "--model-path", "hf-path" or the model folders under '
        '"--artifact-path/models"',
    )
    args.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Hugging Face path from which to download params, tokenizer, and config from",
    )
    args.add_argument(
        "--quantization",
        type=str,
        choices=[*utils.quantization_dict.keys()],
        default=list(utils.quantization_dict.keys())[0],
    )
    args.add_argument("--max-seq-len", type=int, default=-1)
    args.add_argument("--target", type=str, default="auto")
    args.add_argument(
        "--db-path",
        type=str,
        default="log_db",
        help="Path to log database for all models. Default: ./log_db/",
    )
    args.add_argument(
        "--reuse-lib",
        type=str,
        default=None,
        help="Whether to reuse a previously generated lib.",
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
    parsed.system_lib_prefix = None
    parsed = _setup_model_path(parsed)

    if os.path.exists(parsed.db_path):
        filenames = os.listdir(parsed.db_path)
        if (
            len(filenames) == 2
            and "database_workload.json" in filenames
            and "database_tuning_record.json" in filenames
        ):
            ms.database.create(work_dir=parsed.db_path)
            parsed.db_path = [parsed.db_path]
        else:
            db_paths = []
            for filename in filenames:
                db_path = os.path.join(parsed.db_path, filename)
                if os.path.isdir(db_path):
                    try:
                        ms.database.create(work_dir=db_path)
                    except Exception:
                        continue
                    else:
                        db_paths.append(db_path)
            parsed.db_path = db_paths
    else:
        parsed.db_path = []

    if len(parsed.db_path) == 0:
        print(
            f"WARNING: --db-path does not point to a valid database: {parsed.db_path}"
        )
    else:
        print(f"Database paths: {parsed.db_path}")

    utils.parse_target(parsed)
    utils.argparse_postproc_common(parsed)

    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )

    # These dictionaries and functions are used for model weight loading.
    #  - "p" here stands for "parameter".
    #  - The first "Any" here stands for torch.Tensor, and the second stands for numpy.ndarray.
    parsed.pidx2pname: Dict[int, str] = dict()
    parsed.pname2binname: Dict[str, str] = dict()
    parsed.f_convert_pname_fwd: Callable[[str], str] = None
    parsed.f_convert_param_bkwd: Callable[[str, Any], List[Tuple[str, Any]]] = None

    return parsed


def _setup_model_path(args):  # pylint: disable=too-many-branches
    if args.hf_path:
        if args.model != "auto":
            assert args.model == os.path.basename(args.hf_path), (
                'When both "--model" and "--hf-path" is specified, the '
                'value of "--model" is required to match the basename of "--hf-path". '
                f'Got "--model {args.model}" and "--hf-path {args.hf_path}"'
            )
        else:
            args.model = os.path.basename(args.hf_path)
        args.model_path = os.path.join(args.artifact_path, "models", args.model)
        if os.path.exists(args.model_path):
            print(f"Weights exist at {args.model_path}, skipping download.")
        else:
            os.makedirs(args.model_path, exist_ok=True)
            os.system("git lfs install")
            os.system(
                f"git clone https://huggingface.co/{args.hf_path} {args.model_path}"
            )
            print(f"Downloaded weights to {args.model_path}")
        validate_config(args.model_path)
    elif args.model != "auto":
        if os.path.isdir(args.model):
            args.model_path = args.model
            args.model = os.path.basename(args.model)
        else:
            args.model_path = os.path.join(args.artifact_path, "models", args.model)
        validate_config(args.model_path)
    else:
        lookup_path = os.path.join(args.artifact_path, "models")
        print(
            f'"--model" is set to "auto". Searching in {lookup_path} for existing models.'
        )
        for dirname in os.listdir(lookup_path):
            if os.path.isdir(os.path.join(lookup_path, dirname)) and os.path.isfile(
                os.path.join(lookup_path, dirname, "config.json")
            ):
                try:
                    validate_config(os.path.join(lookup_path, dirname))
                except:  # pylint: disable=bare-except
                    pass
                else:
                    args.model_path = os.path.join(lookup_path, dirname)
                    args.model = dirname
                    break
        if args.model == "auto":
            raise ValueError("Please specify either the model_path or the hf_path.")

    print(f'Using path "{args.model_path}" for model "{args.model}"')
    return args


def validate_config(model_path: str):
    if os.path.exists(os.path.join(model_path, "mlc-chat-config.json")):
        raise KeyError(
            "The model located in the directory {} has already been compiled by MLC-LLM. There is"
            " no need to compile it again. If you wish to compile a new model, please provide a"
            " directory (or hf-path) that contains the pre-compiled model in raw HuggingFace"
            " format instead.".format(model_path)
        )
    config_path = os.path.join(model_path, "config.json")
    assert os.path.exists(
        config_path
    ), f"Expecting HuggingFace config, but file not found: {config_path}."
    with open(config_path, encoding="utf-8") as i_f:
        config = json.load(i_f)
        assert (
            "model_type" in config
        ), f"Invalid config format. Expecting HuggingFace config format in: {config_path}"
        assert (
            config["model_type"] in utils.supported_model_types
        ), f"Model type {config['model_type']} not supported."


def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_load_script(name, args):
    input_path = os.path.join(args.artifact_path, "debug", name)
    lib = {"__file__": input_path}
    with open(input_path, "rb") as i_f:
        exec(  # pylint: disable=exec-used
            compile(i_f.read(), input_path, "exec"), lib, lib
        )
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
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def dump_split_tir(mod: tvm.IRModule, args):
    if not args.debug_dump:
        return

    template = """
from tvm.script import ir as I
from tvm.script import tir as T

# fmt: off
{content}
# fmt: on
"""
    mod_static, mod_dynamic = utils.split_static_dynamic_tir(mod)
    static_path = os.path.join(args.artifact_path, "debug", "mod_tir_static.py")
    dynamic_path = os.path.join(args.artifact_path, "debug", "mod_tir_dynamic.py")
    print(f"Dump static shape TIR to {static_path}")
    with open(static_path, "w", encoding="utf-8") as o_f:
        o_f.write(template.format(content=mod_static.script()))
    print(f"Dump dynamic shape TIR to {dynamic_path}")
    with open(dynamic_path, "w", encoding="utf-8") as o_f:
        o_f.write(template.format(content=mod_dynamic.script()))


def mod_transform_before_build(
    mod: tvm.IRModule,
    model_params: List[Optional[tvm.nd.NDArray]],
    args: argparse.Namespace,
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    if args.model.startswith("rwkv-"):
        model_names = [
            "decode",
            "create_kv_cache",
            "softmax_with_temperature",
            "get_metadata",
            "reset_kv_cache",
        ]
    elif args.model.startswith("minigpt4-"):
        model_names = ["embed"]
    elif args.model_category == "llama":
        model_names = [
            "embed",
            "prefill",
            "decode",
            "create_kv_cache",
            "softmax_with_temperature",
            "get_metadata",
        ]
    else:
        model_names = [
            "prefill",
            "decode",
            "create_kv_cache",
            "softmax_with_temperature",
            "get_metadata",
        ]

    if args.quantization.mode != "no":
        if ARGS.model.startswith("rwkv-"):
            mod = mlc_llm.transform.RWKVQuantize(  # pylint: disable=not-callable
                mode=args.quantization.mode,
                dtype=args.quantization.model_dtype,
            )(mod)
        else:
            mod = mlc_llm.transform.GroupQuantize(  # pylint: disable=not-callable
                group_size=40 if args.quantization.mode.endswith("3") else 32,
                sym=args.quantization.sym,
                mode=args.quantization.mode,
                storage_nbit=args.quantization.storage_nbit,
                dtype=args.quantization.model_dtype,
            )(mod)
    mod = mlc_llm.transform.FuseTransposeMatmul()(mod)  # pylint: disable=not-callable
    mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter
    mod = mlc_llm.transform.FuseDecodeMatmulEwise(  # pylint: disable=not-callable
        args.quantization.model_dtype, args.target_kind
    )(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)

    debug_dump_script(mod_transform, "mod_lift_params.py", args)
    debug_dump_script(mod_deploy, "mod_deploy.py", args)

    new_params = utils.transform_params(mod_transform, model_params, args)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy


def dump_default_mlc_chat_config(args):
    params_path = os.path.join(args.artifact_path, "params")
    config: Dict[str, Any] = {}

    if args.reuse_lib:
        config["model_lib"] = f"{args.reuse_lib}"
        if not args.reuse_lib.endswith(args.quantization.name):
            raise RuntimeError(
                f"Trying to reuse lib without suffix {args.quantization.name}"
            )
    else:
        config["model_lib"] = f"{args.model}-{args.quantization.name}"

    config["local_id"] = f"{args.model}-{args.quantization.name}"
    config["conv_template"] = args.conv_template
    config["temperature"] = 0.7
    config["repetition_penalty"] = 1.0
    config["top_p"] = 0.95
    config["mean_gen_len"] = 128
    config["max_gen_len"] = 512
    config["shift_fill_factor"] = 0.3
    config["tokenizer_files"] = utils.get_tokenizer_files(params_path)

    dump_path = os.path.join(params_path, "mlc-chat-config.json")
    with open(dump_path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)
    print(f"Finish exporting chat config to {dump_path}")


def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    target_kind = args.target_kind
    if args.system_lib_prefix:
        mod_deploy = mod_deploy.with_attrs(
            {"system_lib_prefix": args.system_lib_prefix}
        )

    debug_dump_script(mod_deploy, "mod_before_build.py", args)
    if target_kind != "cpu":
        db = utils.get_database(args.db_path)  # pylint: disable=invalid-name
        with db, tvm.target.Target("apple/m1-gpu-restricted"):
            if args.target_kind == "android":
                mod_deploy = mlc_llm.dispatch.DispatchTIROperatorAdreno()(  # pylint: disable=not-callable
                    mod_deploy
                )
            mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
            mod_deploy = (
                mlc_llm.dispatch.DispatchTIROperator(  # pylint: disable=not-callable
                    args.model_category
                )(mod_deploy)
            )
            mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)
            mod_deploy = mlc_llm.transform.LiftTIRGlobalBufferAlloc()(mod_deploy)
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


def main():
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    if ARGS.debug_dump:
        os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(
        ARGS.artifact_path, f"mod_cache_before_build_{ARGS.target_kind}.pkl"
    )
    ARGS.raw_params_path = os.path.join(ARGS.artifact_path, "raw_params")
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    with open(os.path.join(ARGS.model_path, "config.json"), encoding="utf-8") as i_f:
        config = json.load(i_f)
        if not use_cache:
            if ARGS.model_category == "llama":
                mod, params = llama.get_model(ARGS, config)
            elif ARGS.model_category == "gpt_neox":
                mod, params = gpt_neox.get_model(ARGS, config)
            elif ARGS.model_category == "moss":
                mod, params = moss.get_model(ARGS, config)
            elif ARGS.model_category == "rwkv":
                mod, params = rwkv.get_model(ARGS, config)
            elif ARGS.model_category == "minigpt":
                mod, params = minigpt.get_model(ARGS)
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
            with open(cache_path, "rb") as pkl:
                mod = pickle.load(pkl)
        dump_split_tir(mod, ARGS)
        if not ARGS.reuse_lib:
            build(mod, ARGS)
        else:
            print("Reuse existing prebuilt lib {ARGS.reuse_lib}...")
        dump_default_mlc_chat_config(ARGS)


if __name__ == "__main__":
    ARGS = _parse_args()
    main()
