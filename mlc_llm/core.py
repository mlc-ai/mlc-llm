# pylint: disable=missing-docstring
import argparse
from dataclasses import dataclass, field, fields, asdict
import json
import os
import pickle
from typing import Any, Dict

import tvm
from tvm import dlight as dl
from tvm import meta_schedule as ms
from tvm import relax

import mlc_llm
from mlc_llm import utils
from mlc_llm.relax_model import (
    gpt_bigcode,
    gpt_neox,
    gptj,
    llama,
    minigpt,
    param_manager,
    rwkv,
)


@dataclass
class BuildArgs:
    """BuildArgs is the dataclass that organizes the arguments we use in
    building a model."""

    model: str = field(
        default="auto",
        metadata={
            "help": (
                'The name of the model to build. If it is "auto", we will '
                'automatically set the model name according to "--model-path", '
                '"hf-path" or the model folders under "--artifact-path/models"'
            )
        },
    )
    hf_path: str = field(
        default=None,
        metadata={
            "help": "Hugging Face path from which to download params, tokenizer, and config"
        },
    )
    quantization: str = field(
        default=list(utils.quantization_schemes.keys())[0],
        metadata={
            "help": "The quantization mode we use to compile.",
            "choices": [*utils.quantization_schemes.keys()],
        },
    )
    max_seq_len: int = field(
        default=-1,
        metadata={"help": "The maximum allowed sequence length for the model."},
    )
    target: str = field(
        default="auto",
        metadata={"help": "The target platform to compile the model for."},
    )
    db_path: str = field(
        default="log_db",
        metadata={"help": "Path to log database for all models. Default: ./log_db/"},
    )
    reuse_lib: str = field(
        default=None, metadata={"help": "Whether to reuse a previously generated lib."}
    )
    artifact_path: str = field(
        default="dist", metadata={"help": "Where to store the output."}
    )
    use_cache: int = field(
        default=1,
        metadata={"help": "Whether to use previously pickled IRModule and skip trace."},
    )
    convert_weight_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only convert model weights and not build the model.",
            "action": "store_true",
        },
    )
    build_model_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only build model and do not convert model weights.",
            "action": "store_true",
        },
    )
    debug_dump: bool = field(
        default=False,
        metadata={
            "help": "Whether to dump debugging files during compilation.",
            "action": "store_true",
        },
    )
    debug_load_script: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the script for debugging.",
            "action": "store_true",
        },
    )
    llvm_mingw: str = field(
        default="",
        metadata={
            "help": "/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows."
        },
    )
    system_lib: bool = field(
        default=False,
        metadata={"help": "A parameter to `relax.build`.", "action": "store_true"},
    )
    sep_embed: bool = field(
        default=False,
        metadata={
            "help": (
                "Build with separated embedding layer, only applicable to LlaMa. "
                "This feature is in testing stage, and will be formally replaced after "
                "massive overhaul of embedding feature for all models and use cases"
            ),
            "action": "store_true",
        },
    )


def convert_build_args_to_argparser() -> argparse.ArgumentParser:
    """Convert from BuildArgs to an equivalent ArgumentParser."""
    args = argparse.ArgumentParser()
    for field in fields(BuildArgs):
        name = field.name.replace("_", "-")
        field_name = f"--{name}"
        # `kwargs` contains `help`, `choices`, and `action`
        kwargs = field.metadata.copy()
        if field.type == bool:
            # boolean arguments do not need to specify `type`
            args.add_argument(field_name, default=field.default, **kwargs)
        else:
            args.add_argument(
                field_name, type=field.type, default=field.default, **kwargs
            )
    return args


def _parse_args(parsed) -> argparse.Namespace:
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

    return parsed


def _setup_model_path(args: argparse.Namespace):  # pylint: disable=too-many-branches
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
    if model_path.split("/")[-1].startswith("minigpt"):
        # minigpt does not contain a config.json file so we skip the check
        return
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


def mod_transform_before_build(
    mod: tvm.IRModule,
    param_manager: param_manager.ParamManager,
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
    elif args.model.startswith("minigpt"):
        model_names = ["embed"]
    else:
        model_names = [
            "prefill",
            "decode",
            "create_kv_cache",
            "softmax_with_temperature",
            "get_metadata",
        ]
        if args.sep_embed:
            model_names = ["embed", "prefill_with_embed"] + model_names[1:]

    mod = param_manager.transform_dequantize(mod)
    mod = mlc_llm.transform.FuseDecodeTranspose()(mod)  # pylint: disable=not-callable
    mod = mlc_llm.transform.FuseTransposeMatmul()(mod)  # pylint: disable=not-callable
    mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter
    mod = mlc_llm.transform.FuseDecodeMatmulEwise(  # pylint: disable=not-callable
        args.quantization.name, args.target_kind
    )(mod)
    mod = mlc_llm.transform.FuseDecodeTake()(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = mlc_llm.transform.CleanUpTIRAttrs()(mod)
    mod_deploy = mod

    utils.debug_dump_script(mod_deploy, "mod_deploy.py", args)

    return mod_deploy


def dump_default_mlc_chat_config(args: argparse.Namespace):
    args.params_path = os.path.join(args.artifact_path, "params")
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
    config["tokenizer_files"] = utils.get_tokenizer_files(args.params_path)
    config["model_category"] = args.model_category
    config["model_name"] = args.model

    args.chat_config_path = os.path.join(args.params_path, "mlc-chat-config.json")
    with open(args.chat_config_path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)
    print(f"Finish exporting chat config to {args.chat_config_path}")


def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    target_kind = args.target_kind
    if args.system_lib_prefix:
        mod_deploy = mod_deploy.with_attrs(
            {"system_lib_prefix": args.system_lib_prefix}
        )

    utils.debug_dump_script(mod_deploy, "mod_before_build.py", args)
    if target_kind != "cpu":
        db = utils.get_database(args.db_path)  # pylint: disable=invalid-name
        dispatch_target = (
            args.target
            if args.target_kind != "webgpu"
            else tvm.target.Target("apple/m1-gpu-restricted")
        )
        with db, dispatch_target:
            if args.target_kind == "android":
                mod_deploy = mlc_llm.dispatch.DispatchTIROperatorAdreno()(  # pylint: disable=not-callable
                    mod_deploy
                )
            mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
            mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod_deploy)
            mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod_deploy)
            mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(mod_deploy)
            mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(mod_deploy)
            mod_deploy = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod_deploy)
            mod_deploy = mlc_llm.transform.LiftTIRGlobalBufferAlloc()(mod_deploy)
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    if args.debug_load_script:
        mod_deploy = utils.debug_load_script("mod_build_stage_debug.py", args)

    utils.debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target, system_lib=args.system_lib)

    output_filename = (
        f"{args.model}-{args.quantization.name}-{target_kind}.{args.lib_format}"
    )

    utils.debug_dump_shader(
        ex, f"{args.model}_{args.quantization.name}_{target_kind}", args
    )
    args.lib_path = os.path.join(args.artifact_path, output_filename)
    ex.export_library(args.lib_path, **args.export_kwargs)
    print(f"Finish exporting to {args.lib_path}")


def build_model_from_args(args: argparse.Namespace):
    os.makedirs(args.artifact_path, exist_ok=True)
    if args.debug_dump:
        os.makedirs(os.path.join(args.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(args.artifact_path, "mod_cache_before_build.pkl")
    args.raw_params_path = os.path.join(args.artifact_path, "raw_params")
    use_cache = args.use_cache and os.path.isfile(cache_path)
    if args.sep_embed and args.model_category != "llama":
        raise ValueError(f"separate embedding not supported on {args.model}")
    if args.model_category != "minigpt":
        with open(
            os.path.join(args.model_path, "config.json"), encoding="utf-8"
        ) as i_f:
            config = json.load(i_f)
    if not use_cache or args.convert_weight_only:
        if args.model_category == "llama":
            mod, param_manager, params = llama.get_model(args, config)
        elif args.model_category == "gpt_neox":
            mod, param_manager, params = gpt_neox.get_model(args, config)
        elif args.model_category == "gpt_bigcode":
            mod, param_manager, params = gpt_bigcode.get_model(args, config)
        elif args.model_category == "minigpt":
            mod, param_manager, params = minigpt.get_model(args)
        elif args.model_category == "gptj":
            mod, param_manager, params = gptj.get_model(args, config)
        elif args.model_category == "rwkv":
            mod, param_manager, params = rwkv.get_model(args, config)
        else:
            raise ValueError(f"Model {args.model} not supported")

        if not args.build_model_only:
            new_params = utils.convert_weights(param_manager, params, args)
            utils.save_params(new_params, args.artifact_path)
            dump_default_mlc_chat_config(args)

        if args.convert_weight_only:
            exit(0)

        mod = mod_transform_before_build(mod, param_manager, args)
        with open(cache_path, "wb") as outfile:
            pickle.dump(mod, outfile)
        print(f"Save a cached module to {cache_path}.")
        if args.model_category != "minigpt":
            utils.copy_tokenizer(args)
    else:
        print(
            f"Load cached module from {cache_path} and skip tracing. "
            "You can use --use-cache=0 to retrace"
        )
        with open(cache_path, "rb") as pkl:
            mod = pickle.load(pkl)
    if not args.reuse_lib:
        build(mod, args)
    else:
        print("Reuse existing prebuilt lib {ARGS.reuse_lib}...")


def build_model(args: BuildArgs):
    r"""Builds/compiles a model.

    Parameters
    ----------
    args : mlc_llm.BuildArgs
        A dataclass of arguments for building models.

    Returns
    ----------
    lib_path: str
        The path to the `.so` library file.
    model_path: str
        The path to the folder of the model's parameters.
    chat_config_path: str
        The path to the chat config `.json` file.
    """
    # Convert BuildArgs to argparse.Namespace so that we can share the rest
    # of the code with the command line workflow
    build_args_as_dict = asdict(args)
    build_args_namespace = argparse.Namespace(**build_args_as_dict)
    args = _parse_args(build_args_namespace)
    build_model_from_args(args)

    return args.lib_path, args.params_path, args.chat_config_path
