# pylint: disable=missing-docstring, redefined-outer-name, not-callable
import argparse
import functools
import json
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

import mlc_llm
import tvm
import tvm.relax.backend.contrib.cublas as _
from mlc_llm import utils
from mlc_llm.relax_model import (
    chatglm,
    gpt_bigcode,
    gpt_neox,
    gptj,
    llama,
    llama_batched_vllm,
    minigpt,
    mistral,
    param_manager,
    rwkv,
    stablelm_3b,
)
from mlc_llm.relax_model.commons import (
    create_shard_info_func,
    create_shard_transformation_func,
)
from mlc_llm.relax_model.param_manager import (
    chain_parameter_transforms,
    transform_params_for_each_rank,
)
from mlc_llm.transform import fuse_split_rotary_embedding, rewrite_attention
from tvm import dlight as dl
from tvm import relax
from tvm.contrib.nvcc import parse_compute_version
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import annotate_workspace


@dataclass
class BuildArgs:
    r"""BuildArgs is the dataclass that organizes the arguments we use in
    building a model.

    To use :meth:`mlc_llm.build_model`, users pass in an instance of :class:`BuildArgs`; for
    CLI entry points, an equivalent :class:`ArgumentParser` instance is generated based
    on the definition of this class using :meth:`mlc_llm.convert_build_args_to_argparser`.

    Parameters
    ----------
    model: str
        The name of the model to build. If it is ``auto``, we will automatically
        set the model name according to ``--model-path``, ``hf-path``, or the model
        folders under ``--artifact-path/models``.

    hf_path: str
        Hugging Face path from which to download params, tokenizer, and config.

    quantization: str
        The quantization mode we use to compile.

    max_seq_len: int
        The maximum allowed sequence length for the model.

    target: str
        The target platform to compile the model for.

    db_path: str
        Path to log database for all models. Default: ``./log_db/``.

    reuse_lib: str
        Whether to reuse a previously generated lib.

    artifact_path: str
        Where to store the output.

    use_cache: int
        Whether to use previously pickled IRModule and skip trace.

    convert_weights_only: bool
        Whether to only convert model weights and not build the model. If both
        ``convert_weight_only`` and ``build_model_only`` are set, the behavior is undefined.

    build_model_only: bool
        Whether to only build model and do not convert model weights.

    debug_dump: bool
        Whether to dump debugging files during compilation.

    debug_load_script: bool
        Whether to load the script for debugging.

    llvm_mingw: str
        ``/path/to/llvm-mingw-root``, use llvm-mingw to cross compile to windows.

    system_lib: bool
        A parameter to ``relax.build``.

    sep_embed: bool
        Build with separated embedding layer, only applicable to LlaMa. This
        feature is in testing stage, and will be formally replaced after massive
        overhaul of embedding feature for all models and use cases.

    sliding_window: int
        The sliding window size in sliding window attention (SWA). This optional field
        overrides the `sliding_window` in config.json for those models that use SWA.
        Currently only useful when compiling Mistral.

    prefill_chunk_size: int
        The chunk size during prefilling. By default, the chunk size is the same as
        max sequence length. Currently only useful when compiling Mistral.

    attention_sink_size: int
        Number of attention sinks (https://arxiv.org/abs/2309.17453).
        Only supported on mistral yet.

    cc_path: str
        ``/path/to/cross_compiler_path``; currently only used for cross-compile
        for nvidia/jetson device.

    use_safetensors: bool
        Specifies whether to use ``.safetensors`` instead of the default ``.bin``
        when loading in model weights.

    enable_batching: bool
        Build the model for batched inference.
        This is a temporary flag used to control the model execution flow in single-
        sequence and batching settings for now. We will eventually merge two flows
        in the future and remove this flag then.

    no_cutlass_attn: bool
        Disable offloading attention operations to CUTLASS.

    no_cutlass_norm: bool
        Disable offloading layer and RMS norm operations to CUTLASS.

    no_cublas: bool
        Disable the step that offloads matmul to cuBLAS. Without this flag,
        matmul will be offloaded to cuBLAS if quantization mode is ``q0f16`` or
        ``q0f32``, target is CUDA and TVM has been built with cuBLAS enabled.

    use_cuda_graph: bool
        Specifies whether to enable CUDA Graph for the decoder. MLP and QKV
        projection between two attention layers are put into a graph.

    num_shards: int
        Number of shards to split the model into in tensor parallelism multi-gpu
        inference. Only useful when ``build_model_only`` is set.

    use_flash_attn_mqa: bool
        Offload multi-query attention workload to Flash Attention.

    pdb: bool
        If set, drop into a pdb debugger on error.

    use_vllm_attention: bool
        Use vLLM paged KV cache and attention kernel, only relevant when enable_batching=True.
    """
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
        metadata={"help": "Hugging Face path from which to download params, tokenizer, and config"},
    )
    quantization: str = field(
        default="q4f16_1",
        metadata={
            "help": "The quantization mode we use to compile.",
            "choices": [*utils.quantization_schemes.keys()],
        },
    )
    max_seq_len: int = field(
        default=-1,
        metadata={"help": "The maximum allowed sequence length for the model."},
    )
    max_vocab_size: int = field(
        default=40000,
        metadata={"help": "The maximum allowed vocabulary size for the model."},
    )
    target: str = field(
        default="auto",
        metadata={"help": "The target platform to compile the model for."},
    )
    reuse_lib: str = field(
        default=None, metadata={"help": "Whether to reuse a previously generated lib."}
    )
    artifact_path: str = field(default="dist", metadata={"help": "Where to store the output."})
    artifact_tag: str = field(default=None, metadata={"help": "User-defined tag for the artifact"})
    use_cache: int = field(
        default=1,
        metadata={"help": "Whether to use previously pickled IRModule and skip trace."},
    )
    convert_weights_only: bool = field(
        default=False,
        metadata={
            "dest": "convert_weights_only",
            "action": "store_true",
            "help": "Whether to only convert model weights and not build the model.",
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
        metadata={"help": "/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows."},
    )
    cc_path: str = field(
        default="",
        metadata={
            "help": (
                "/path/to/cross_compiler_path, Currently only used for "
                "cross-compile for nvidia/jetson device."
            )
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
    use_safetensors: bool = field(
        default=False,
        metadata={
            "help": (
                "Specifies whether to use ``.safetensors`` instead of the default "
                "``.bin`` when loading in model weights."
            ),
            "action": "store_true",
        },
    )
    enable_batching: bool = field(
        default=False,
        metadata={
            "help": (
                "Build the model for batched inference."
                "This is a temporary flag used to control the model execution flow in single-"
                "sequence and batching settings for now. We will eventually merge two flows"
                "in the future and remove this flag then."
            ),
            "action": "store_true",
        },
    )
    max_batch_size: int = field(
        default=80,
        metadata={
            "help": (
                "The maximum batch size for build. It has effect only when batching is enabled."
            ),
        },
    )
    no_cutlass_attn: bool = field(
        default=False,
        metadata={
            "help": ("Disable offloading attention operations to CUTLASS."),
            "action": "store_true",
        },
    )
    no_cutlass_norm: bool = field(
        default=False,
        metadata={
            "help": ("Disable offloading layer and RMS norm operations to CUTLASS."),
            "action": "store_true",
        },
    )
    no_cublas: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable the step that offloads matmul to cuBLAS. Without this flag, "
                "matmul will be offloaded to cuBLAS if quantization mode is q0f16 or q0f32, "
                "target is CUDA and TVM has been built with cuBLAS enabled."
            ),
            "action": "store_true",
        },
    )
    no_cache_dump: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable dumping `mod_cache_before_build.pkl`. When this flag is set, cached build would not be available."
            ),
            "action": "store_true",
        },
    )
    use_cuda_graph: bool = field(
        default=False,
        metadata={
            "help": (
                "Specifies whether to enable CUDA Graph for the decoder. MLP and QKV "
                "projection between two attention layers are put into a graph."
            ),
            "action": "store_true",
        },
    )
    num_shards: int = field(
        default=1,
        metadata={
            "help": (
                "Number of shards to split the model into in tensor parallelism multi-gpu "
                "inference. Only useful when --build-model-only is set."
            ),
        },
    )
    use_presharded_weights: bool = field(
        default=False,
        metadata={
            "action": "store_true",
            "help": "Produce separate weight sets for each shard.",
        },
    )
    use_flash_attn_mqa: bool = field(
        default=False,
        metadata={
            "help": ("Offload multi-query attention workload to Flash Attention."),
            "action": "store_true",
        },
    )
    sliding_window: int = field(
        default=-1,
        metadata={
            "help": (
                "The sliding window size in sliding window attention (SWA). "
                "This optional field overrides the `sliding_window` in config.json for "
                "those models that use SWA. Currently only useful when compiling Mistral."
            ),
        },
    )
    prefill_chunk_size: int = field(
        default=-1,
        metadata={
            "help": (
                "The chunk size during prefilling. By default, the chunk size is "
                "the same as the sliding window size or the max sequence length. "
                "Currently only useful when compiling Mistral."
            ),
        },
    )
    attention_sink_size: int = field(
        default=0,
        metadata={
            "help": (
                "The number of attention sinks to keep in cache."
                "Only supported on mistral yet."
            ),
        },
    )
    pdb: bool = field(
        default=False,
        metadata={
            "help": ("If set, drop into a pdb debugger on error"),
            "action": "store_true",
        },
    )
    use_vllm_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Use vLLM paged KV cache and attention kernel, only relevant when "
                "enable_batching=True."
            ),
            "action": "store_true",
        },
    )

    @property
    def convert_weight_only(self):
        """A backwards-compatibility helper"""
        return self.convert_weights_only


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
            args.add_argument(field_name, type=field.type, default=field.default, **kwargs)

    # Most models contain more than a single parameter (citation
    # needed), so "weights" should be plural.  The initial use of
    # "--convert-weight-only" caused enough typos that it is worth
    # fixing.  The old argument spelling is retained for backwards
    # compatibility.
    args.add_argument(
        "--convert-weight-only",
        default=False,
        dest="convert_weights_only",
        action="store_true",
        help="Equivalent to --convert-weights-only, retained for backwards compatibility.",
    )

    return args


def _parse_args(parsed) -> argparse.Namespace:
    assert parsed.max_seq_len == -1 or parsed.max_seq_len > 0
    if parsed.use_safetensors:
        try:
            import safetensors  # pylint: disable=import-outside-toplevel, unused-import
        except ImportError as error:
            raise ImportError(
                "`use_safetensors` option is toggled, please install safetensors package."
            ) from error

    parsed.export_kwargs = {}
    parsed.lib_format = "so"
    parsed.system_lib_prefix = None
    parsed = _setup_model_path(parsed)

    utils.parse_target(parsed)
    utils.argparse_postproc_common(parsed)

    if parsed.use_vllm_attention:
        assert parsed.enable_batching, "--enable_batching is required for using vLLM attention."
        assert parsed.target_kind == "cuda", "vLLM attention is only supported for CUDA."
        assert tvm.get_global_func(
            "tvm.contrib.vllm.single_query_cached_kv_attention", True
        ), "TVM needs to be built with -DUSE_VLLM=ON."

    model_name = [
        parsed.model,
        parsed.quantization.name,
    ]
    if parsed.use_presharded_weights:
        model_name.append(f"presharded-{parsed.num_shards}gpu")

    # TODO(@sunggg): currently, we overwrite the artifact_path which forces to rely on name deduction rule.
    # Ideally, it is better to separate its root path and name tag.
    # Until we make the change in upstream, this is a temporary hack.
    artifact_tag = parsed.artifact_tag if parsed.artifact_tag else "-".join(model_name)
    parsed.artifact_path = os.path.join(parsed.artifact_path, artifact_tag)

    parsed.lib_name = f"{parsed.model}-{parsed.quantization.name}-{parsed.target_kind}.{parsed.lib_format}"
    parsed.lib_path = os.path.join(parsed.artifact_path, parsed.lib_name)

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
            os.system(f"git clone https://huggingface.co/{args.hf_path} {args.model_path}")
            print(f"Downloaded weights to {args.model_path}")
        validate_config(args.model_path)
    elif args.model != "auto":
        if os.path.isdir(args.model):
            args.model = os.path.normpath(args.model)  # Remove potential trailing `/`
            args.model_path = args.model
            args.model = os.path.basename(args.model)
        else:
            args.model_path = os.path.join(args.artifact_path, "models", args.model)
        validate_config(args.model_path)
    else:
        lookup_path = os.path.join(args.artifact_path, "models")
        print(f'"--model" is set to "auto". Searching in {lookup_path} for existing models.')
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
            f"The model located in the directory {model_path} has already been compiled "
            "by MLC-LLM. There is no need to compile it again. If you wish to compile "
            "a new model, please provide a directory (or hf-path) that contains the "
            "pre-compiled model in raw HuggingFace format instead."
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


def get_cuda_sm_version():
    major, minor = parse_compute_version(tvm.cuda(0).compute_version)

    if major == 8:
        sm = 80
    else:
        sm = 10 * major + minor

    return sm


def mod_transform_before_build(
    mod: tvm.IRModule,
    param_manager: param_manager.ParamManager,
    args: argparse.Namespace,
    config: Dict,
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    if args.model.startswith("minigpt"):
        model_names = ["embed"]
    else:
        model_names = [
            "prefill",
            "decode",
        ]

        if not args.use_vllm_attention:
            model_names += [
                "create_kv_cache",
                "softmax_with_temperature",
                "get_metadata",
            ]
        else:
            # This is equivalent to prefill but without KV cache. It is used for
            # determining the number of paged cache blocks that can be allocated.
            model_names.append("evaluate")
            model_names.append("evaluate_multi_query")

        if args.sep_embed:
            model_names = ["embed", "prefill_with_embed"] + model_names[1:]
            if args.enable_batching:
                model_names[2] = "decode_with_embed"
        if args.model.lower().startswith("rwkv-"):
            model_names += ["reset_kv_cache"]

    mod = param_manager.transform_dequantize()(mod)
    mod = relax.transform.BundleModelParams()(mod)

    use_ft_quant = args.quantization.name in [
        "q4f16_ft",
        "q8f16_ft",
        "q4f16_ft_group",
        "q8f16_ft_group",
    ]
    mod = mlc_llm.transform.FuseDecodeTranspose(skip_gemm=not use_ft_quant)(mod)

    if (
        hasattr(config, "num_attention_heads")
        and hasattr(config, "hidden_size")
        and hasattr(config, "position_embedding_base")
        and getattr(config, "dtype", "float16") == "float16"
    ):
        max_seq_len = None
        if args.max_seq_len > 0:
            max_seq_len = args.max_seq_len
        elif hasattr(config, "max_sequence_length"):
            max_seq_len = config.max_sequence_length

        if max_seq_len:
            num_key_value_heads = config.get_num_key_value_heads()
            # pylint: disable=no-value-for-parameter
            mod = fuse_split_rotary_embedding(
                config.num_attention_heads // args.num_shards,
                num_key_value_heads // args.num_shards,
                config.hidden_size // args.num_shards,
                config.position_embedding_base,
                batched=args.enable_batching,
            )(mod)

    if args.target_kind == "cuda":
        patterns = []

        has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

        if has_cutlass and not args.no_cutlass_attn:
            # pylint: disable=no-value-for-parameter
            if args.use_flash_attn_mqa:
                mod = rewrite_attention(use_flash_mqa=True)(mod)
            mod = rewrite_attention(use_flash_mqa=False)(mod)
            patterns += get_patterns_with_prefix("cutlass.attention")

        if has_cutlass and not args.no_cutlass_norm:
            patterns += get_patterns_with_prefix("cutlass.layer_norm")
            patterns += get_patterns_with_prefix("cutlass.rms_norm")

        if has_cutlass and use_ft_quant:
            patterns += get_patterns_with_prefix("cutlass.decode_matmul")

        has_cublas = tvm.get_global_func("relax.ext.cublas", True)

        if has_cublas and args.quantization.name in ("q0f16", "q0f32") and not args.no_cublas:
            patterns += get_patterns_with_prefix("cublas")

        if len(patterns) > 0:
            os.makedirs("./tmp", exist_ok=True)

            sm = get_cuda_sm_version()

            # Some kernels are only supported for sm80 or lower. H100 can run a kernel compiled for older arches,
            # but in this particular case compilation fails due to architecture mismatch. This param is only used
            # for compiling sm 80 specific kernels. So for hopper sm_90 we set the parameter to sm_80.
            if sm == 90:
                sm = 80

            options = {"cutlass": {"sm": sm, "find_first_valid": False}}

            if hasattr(config, "rms_norm_eps"):
                options["cutlass"]["rms_eps"] = config.rms_norm_eps

            mod = tvm.transform.Sequential(
                [
                    relax.transform.FuseOpsByPattern(
                        patterns, bind_constants=False, annotate_codegen=True
                    ),
                    annotate_workspace,
                    relax.transform.AllocateWorkspace(),
                    relax.transform.RunCodegen(options, entry_functions=model_names),
                ]
            )(mod)

    mod = mlc_llm.transform.FuseTransposeMatmul()(mod)
    mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter
    mod = mlc_llm.transform.FuseDecodeMatmulEwise()(mod)
    mod = mlc_llm.transform.FuseDecodeTake()(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = mlc_llm.transform.CleanUpTIRAttrs()(mod)
    mod_deploy = mod

    utils.debug_dump_script(mod_deploy, "mod_deploy.py", args)

    return mod_deploy

def dump_build_config(
    args: argparse.Namespace
):
    build_config_path = os.path.join(args.artifact_path, "build_config.json")
    config: Dict[str, Any] = {
        "num_shards": args.num_shards,
        "quantization": args.quantization.name,
        "library_name": args.lib_name,
        "build_options": str(args)
    }
    with open(build_config_path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)

def dump_mlc_chat_config(
    args: argparse.Namespace,
    vocab_size: int,
    max_window_size: int,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    top_p: float = 0.95,
    mean_gen_len: int = 128,
    max_gen_len: int = 512,
    shift_fill_factor: float = 0.3,
    rwkv_world=False,
):
    args.params_path = os.path.join(args.artifact_path, "params")
    config: Dict[str, Any] = {}

    if args.reuse_lib:
        config["model_lib"] = f"{args.reuse_lib}"
        if not args.reuse_lib.endswith(args.quantization.name):
            raise RuntimeError(f"Trying to reuse lib without suffix {args.quantization.name}")
    else:
        config["model_lib"] = f"{args.model}-{args.quantization.name}"

    config["local_id"] = f"{args.model}-{args.quantization.name}"
    config["conv_template"] = args.conv_template
    config["temperature"] = temperature
    config["repetition_penalty"] = repetition_penalty
    config["top_p"] = top_p
    config["mean_gen_len"] = mean_gen_len
    config["max_gen_len"] = max_gen_len
    config["num_shards"] = args.num_shards
    config["use_presharded_weights"] = args.use_presharded_weights
    config["shift_fill_factor"] = shift_fill_factor
    if rwkv_world:
        config["tokenizer_files"] = ["tokenizer_model"]
    else:
        config["tokenizer_files"] = utils.get_tokenizer_files(args.params_path)
    config["model_category"] = args.model_category
    config["model_name"] = args.model
    config["vocab_size"] = vocab_size
    config["prefill_chunk_size"] = args.prefill_chunk_size
    if args.sliding_window != -1:
        # Do not add max window size if use sliding window
        config["sliding_window"] = args.sliding_window

        # only use sinks if sliding window enabled
        if args.attention_sink_size > 0:
            config["attention_sink_size"] = args.attention_sink_size
    else:
        config["max_window_size"] = max_window_size

    args.chat_config_path = os.path.join(args.params_path, "mlc-chat-config.json")
    with open(args.chat_config_path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)
    print(f"Finish exporting chat config to {args.chat_config_path}")


def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    target_kind = args.target_kind
    if args.system_lib_prefix:
        mod_deploy = mod_deploy.with_attrs({"system_lib_prefix": args.system_lib_prefix})

    utils.debug_dump_script(mod_deploy, "mod_before_build.py", args)
    utils.debug_dump_benchmark_script(
        mod_deploy, f"{args.model}_{args.quantization.name}".replace("-", "_"), args
    )

    if target_kind != "cpu":
        dispatch_target = (
            args.target
            if args.target_kind != "webgpu"
            else tvm.target.Target("apple/m1-gpu-restricted")
        )
        with dispatch_target:
            mod_deploy = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod_deploy)
            mod_deploy = (
                mlc_llm.transform.LiftTIRGlobalBufferAlloc()(  # pylint: disable=not-callable
                    mod_deploy
                )
            )
        if not args.enable_batching and target_kind != "cuda":
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    if args.debug_load_script:
        mod_deploy = utils.debug_load_script("mod_build_stage_debug.py", args)

    utils.debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    use_cuda_graph = args.use_cuda_graph and target_kind == "cuda"

    with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": use_cuda_graph}):
        # The num_input attribute is needed to capture transformed weights passed as input
        # into a cuda graph.
        # NOTE: CUDA graph for batching is not enabled and is left as a TODO item.
        if not args.enable_batching:
            mod_deploy["decode"] = mod_deploy["decode"].with_attr({"num_input": 3})
        ex = relax.build(mod_deploy, args.target, system_lib=args.system_lib)

    utils.debug_dump_shader(ex, f"{args.model}_{args.quantization.name}_{target_kind}", args)
    ex.export_library(args.lib_path, **args.export_kwargs)
    print(f"Finish exporting to {args.lib_path}")


def build_model_from_args(args: argparse.Namespace):
    if args.quantization == "q4f16_0":
        print(
            "WARNING: q4f16_1 is preferred to q4f16_0, "
            "and it is highly recommended to use q4f16_1 instead"
        )

    use_ft_quant = args.quantization.name in ["q4f16_ft", "q8f16_ft", "q4f16_ft_group", "q8f16_ft_group"]

    if args.num_shards > 1:
        if (not args.build_model_only) and (not args.convert_weights_only):
            raise ValueError(
                "`num_shards` should be used together with "
                "`--build-model-only` and `--convert-weight-only`"
            )

        if use_ft_quant:
            args.use_presharded_weights = True

    os.makedirs(args.artifact_path, exist_ok=True)
    if args.debug_dump:
        os.makedirs(os.path.join(args.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(args.artifact_path, "mod_cache_before_build.pkl")
    args.raw_params_path = os.path.join(args.artifact_path, "raw_params")
    use_cache = args.use_cache and os.path.isfile(cache_path)
    if args.sep_embed and args.model_category != "llama":
        raise ValueError(f"separate embedding not supported on {args.model}")

    if args.model_category == "minigpt":
        # Special case for minigpt, which neither provides nor requires a configuration.
        config = {}
    else:
        with open(os.path.join(args.model_path, "config.json"), encoding="utf-8") as i_f:
            config = json.load(i_f)

    if not use_cache or args.convert_weights_only or not os.path.exists(cache_path):
        model_generators = {
            "llama": llama,
            "mistral": mistral,
            "stablelm_epoch": stablelm_3b,
            "gpt_neox": gpt_neox,
            "gpt_bigcode": gpt_bigcode,
            "minigpt": minigpt,
            "gptj": gptj,
            "rwkv": rwkv,
            "rwkv_world": rwkv,
            "chatglm": chatglm,
            "mixtral": llama,
        }

        if args.use_vllm_attention:
            model_generators["llama"] = llama_batched_vllm
            model_generators["mistral"] = llama_batched_vllm
            model_generators["mixtral"] = llama_batched_vllm

        assert args.model_category in model_generators, f"Model {args.model} not supported"

        mod, param_manager, params, model_config = model_generators[args.model_category].get_model(
            args, config
        )

        if args.model_category == "mistral":
            args.sliding_window = model_config.sliding_window
            # This line is introduced by the merge with upstream
            #   see: https://github.com/octoml/mlc-llm/pull/52
            # However, HF config does not have this info and we don't need this info.
            # So commented out for now.
            # args.sliding_window_chunk_size = model_config.sliding_window_chunk_size

        for qspec_updater_class in param_manager.qspec_updater_classes:
            qspec_updater = qspec_updater_class(param_manager)
            qspec_updater.visit_module(mod)

        if not args.build_model_only:
            parameter_transforms = []

            # Run pre-quantization if provided.
            args.model_path = param_manager.run_pre_quantize(args.model_path)
            param_manager.init_torch_pname_to_bin_name(args.use_safetensors)
            parameter_transforms.append(param_manager.create_parameter_transformation(optimize_parameter_order=False)) # disable to prevent errors

            # Run pre-sharding if required
            if args.num_shards > 1 and args.use_presharded_weights:
                mod_shard = create_shard_transformation_func(param_manager, args, model_config)
                mod_shard = transform_params_for_each_rank(mod_shard, num_shards=args.num_shards)
                parameter_transforms.append(mod_shard)

            # Chain all parameter transforms together.  This allows
            # ReorderTransformFunc to be applied to the single
            # resulting parameter transformation function.
            mod_transform = functools.reduce(chain_parameter_transforms, parameter_transforms)

            seq = tvm.ir.transform.Sequential(
                [
                    relax.transform.CanonicalizeBindings(),
                    relax.transform.EliminateCommonSubexpr(),
                    relax.transform.DeadCodeElimination(),
                    # TODO(Lunderberg): Implement
                    # relax.transform.Simplify() that applies
                    # canonicalization, CSE, and DCE until
                    # convergence.
                    relax.transform.CanonicalizeBindings(),
                    relax.transform.EliminateCommonSubexpr(),
                    relax.transform.DeadCodeElimination(),
                    param_manager.optimize_transform_param_order(),
                ],
                name="SimplifyModTransform",
            )

            mod_transform = seq(mod_transform)

            params = utils.convert_weights(mod_transform, param_manager, params, args)

            if args.num_shards > 1 and use_ft_quant:
                preprocessed = []
                weight_preprocess_func = tvm.get_global_func("cutlass.ft_preprocess_weight")
                is_int4 = args.quantization.name in ["q4f16_ft", "q4f16_ft_group"]
                sm = get_cuda_sm_version()

                for p in params:
                    if p.dtype == "int8":
                        preprocessed.append(weight_preprocess_func(p, sm, is_int4))
                    else:
                        preprocessed.append(p)

                params = preprocessed

            utils.save_params(params, args.artifact_path, args.num_shards if args.use_presharded_weights else 1)

            if not args.enable_batching:
                if args.model_category == "rwkv" or args.model_category == "rwkv_world":
                    # TODO: refactor config into model definition
                    dump_mlc_chat_config(
                        args,
                        vocab_size=config["vocab_size"],
                        max_window_size=model_config.max_sequence_length,
                        max_gen_len=model_config.max_sequence_length,
                        top_p=0.6,
                        temperature=1.2,
                        repetition_penalty=0.996,
                        rwkv_world=True,
                    )
                elif args.model_category == "chatglm":
                    dump_mlc_chat_config(
                        args,
                        vocab_size=config["padded_vocab_size"],
                        max_window_size=model_config.max_sequence_length,
                        max_gen_len=model_config.max_sequence_length,
                    )
                else:
                    dump_mlc_chat_config(
                        args,
                        vocab_size=config["vocab_size"],
                        max_window_size=model_config.max_sequence_length,
                        max_gen_len=model_config.max_sequence_length,
                    )

        if args.enable_batching:
            # when batching is enabled, we dump info for mlc_serve runtime
            dump_build_config(args)
            model_info_path = os.path.join(args.artifact_path, "model")
            os.makedirs(model_info_path, exist_ok=True)
            mlc_model_config_path = os.path.join(model_info_path, "mlc-model-config.json")

            max_context_length = args.max_seq_len
            if args.max_seq_len == -1:
                # for llama-1 family
                if "max_sequence_length" in config:
                    max_context_length = config["max_sequence_length"]
                # for llama-2, mistral, etc.
                elif "max_position_embeddings" in config:
                    max_context_length = config["max_position_embeddings"]
                else:
                    raise Exception("The model config should contain information about maximum context length.")

            # Overwrite some configs
            config["max_context_length"] = max_context_length
            if args.sliding_window != -1 and "sliding_window" in config:
                config["sliding_window"] = args.sliding_window

            # copy hf config into mlc_model_config
            mlc_model_config = config.copy()

            with open(mlc_model_config_path, "w", encoding="utf-8") as outfile:
                json.dump(mlc_model_config, outfile, indent=4)

        if args.model_category != "minigpt":
            utils.copy_tokenizer(args)

        if args.convert_weights_only:
            exit(0)

        mod = mod_transform_before_build(mod, param_manager, args, model_config)
        if args.num_shards > 1:
            # We require a "create_sharding_info" function for all
            # multi-GPU models, even if they are using pre-sharded
            # weights.  When using pre-sharded weights, the list of
            # initialization-time transforms to apply is empty.
            sharding_module = create_shard_info_func(param_manager, args, model_config)
            mod.update(sharding_module)

        if not args.no_cache_dump:
            with open(cache_path, "wb") as outfile:
                pickle.dump(mod, outfile)
            print(f"Save a cached module to {cache_path}.")
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
        print(f"Reuse existing prebuilt lib {args.reuse_lib}...")


def build_model(args: BuildArgs) -> (Optional[str], Optional[str], Optional[str]):
    r"""Builds/compiles a model.

    Parameters
    ----------
    args : :class:`BuildArgs`
        A dataclass of arguments for building models.mlc_llm/core.py

    Returns
    ----------
    lib_path: Optional[str]
        The path to the model library file. Return ``None`` if not applicable.
    model_path: Optional[str]
        The path to the folder of the model's parameters. Return ``None`` if not applicable.
    chat_config_path: Optional[str]
        The path to the chat config `.json` file. Return ``None`` if not applicable.
    """
    # Convert BuildArgs to argparse.Namespace so that we can share the rest
    # of the code with the command line workflow
    build_args_as_dict = asdict(args)
    build_args_namespace = argparse.Namespace(**build_args_as_dict)
    args = _parse_args(build_args_namespace)
    build_model_from_args(args)

    # Prepare output; some workflows may or may not have the paths to return
    lib_path = args.lib_path if hasattr(args, "lib_path") else None
    model_path = args.params_path if hasattr(args, "params_path") else None
    chat_config_path = args.chat_config_path if hasattr(args, "chat_config_path") else None

    return lib_path, model_path, chat_config_path
