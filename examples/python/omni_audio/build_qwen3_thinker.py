"""Build the shared Qwen3-Omni Thinker text backbone from official model configs."""

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm_tirx
from tvm import relax

from mlc_llm.compiler_pass import pipeline as _pipeline  # noqa: F401
from mlc_llm.model import MODELS
from mlc_llm.protocol.artifact_manifest_v2 import (
    compute_interface_id_v2,
    validate_compiled_v2,
)
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support import auto_target

MODEL = MODELS["qwen3_moe"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_source_identity(repo, revision):
    if (
        repo
        not in {
            "Qwen/Qwen3-Omni-30B-A3B-" + variant
            for variant in ("Instruct", "Thinking", "Captioner")
        }
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("Use an explicit supported repository and pinned 40-digit revision")


def load_profile(path, repo, revision):
    validate_source_identity(repo, revision)
    config_path = path / "config.json"
    pinned = dict(config=json.loads(config_path.read_text()), repo=repo, revision=revision)
    config = pinned["config"]["thinker_config"]["text_config"]
    rope = config["rope_scaling"]
    if (
        config["model_type"] != "qwen3_omni_moe_text"
        or rope["rope_type"] != "default"
        or not config["use_qk_norm"]
        or config["use_sliding_window"]
        or config["mlp_only_layers"]
        or config["shared_expert_intermediate_size"] != 0
        or config["tie_word_embeddings"]
    ):
        raise ValueError(f"Unsupported Thinker backbone: {pinned['repo']}")
    cfg = MODEL.config.from_dict(
        dict(config, max_batch_size=1, prefill_chunk_size=256, tensor_parallel_shards=1)
    )
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    source_names = {
        name for name in index if name.startswith(("thinker.model.", "thinker.lm_head."))
    }
    return (
        cfg,
        source_names,
        {
            "repo": pinned["repo"],
            "revision": pinned["revision"],
            "config_sha256": sha(config_path),
            "index_sha256": sha(index_path),
            "audio_output": pinned["config"]["enable_audio_output"],
            "audio_token_id": pinned["config"]["thinker_config"]["audio_token_id"],
            "image_token_id": pinned["config"]["thinker_config"]["image_token_id"],
            "video_token_id": pinned["config"]["thinker_config"]["video_token_id"],
            "eos_token_ids": [pinned["config"]["im_end_token_id"]],
        },
    )


def add_reset(mod):
    builder = relax.BlockBuilder(mod)
    state = relax.Var("state", relax.AnyType())
    with builder.function("reset_state", [state], pure=False):
        builder.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.kv_state_clear"),
                [state],
                ty_args=[relax.TupleType([])],
            )
        )
        builder.emit_func_output(relax.Tuple([]))
    return builder.finalize()


def export_contract(out, schema, config, library, device):
    schema_json = json.dumps(
        sorted(schema, key=lambda p: p["name"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    weights_id = "sha256:" + hashlib.sha256(schema_json.encode()).hexdigest()
    inputs = dict(
        token_ids=dict(semantic="token.ids.v1", dtype="int32", shape=["seq_len"], delivery="value"),
        audio_embeddings=dict(
            semantic="audio.embeddings.v1",
            dtype="float32",
            shape=["audio_tokens", config.hidden_size],
            delivery="value",
        ),
    )
    output = dict(semantic="token.ids.v1", dtype="int32", shape=[1], delivery="stream")
    spec = dict(
        kind="autoregressive",
        contract="qwen3_omni.thinker_audio.v1",
        inputs=inputs,
        outputs=dict(token_ids=output),
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=weights_id),
        state=dict(kind="kv_cache", scope="request"),
    )
    task = dict(
        inputs={
            name: dict(spec=port, targets=[dict(component="thinker", port=name)])
            for name, port in inputs.items()
        },
        outputs=dict(
            token_ids=dict(
                spec=output,
                source=dict(component="thinker", port="token_ids"),
                optional=False,
            )
        ),
        edges=[],
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components=dict(thinker=spec),
        tasks={"thinker.generate": task},
    )
    package["interface_id"] = compute_interface_id_v2(package)
    sizes = [prod(param["shape"]) * np.dtype(param["dtype"]).itemsize for param in schema]
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components=dict(
            thinker=dict(
                spec=spec,
                module=library.name,
                exports=dict(
                    run="prefill",
                    prefill="prefill",
                    decode="decode",
                    embed="embed",
                    create_state="create_tir_paged_kv_cache",
                    reset_state="reset_state",
                ),
                resources=dict(
                    required_features=["shader-f16"] if device == "webgpu" else [],
                    max_storage_buffer_binding_size=max(sizes),
                    estimated_device_memory_bytes=sum(sizes),
                ),
            )
        ),
    )
    validate_compiled_v2(package, compiled)
    for name, value in [("model", package), ("compiled", compiled)]:
        (out / f"{name}-manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("webgpu", "metal"), default="webgpu")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg, names, profile = load_profile(args.source, args.repo, args.revision)
    mapping = MODEL.source["huggingface-safetensor"](cfg, QUANTIZATION["q0f16"])
    mapped_names = {f"thinker.{name}" for names in mapping.param_map.values() for name in names}
    token_fields = (
        "audio_token_id",
        "image_token_id",
        "video_token_id",
        "eos_token_ids",
    )
    token_ids = {name: profile[name] for name in token_fields}
    if names != mapped_names:
        raise ValueError(
            f"Thinker backbone mismatch for {profile['repo']}: "
            f"{len(names - mapped_names)} unused, {len(mapped_names - names)} missing"
        )
    profile["hf_tensor_count"] = len(names)
    quant = QUANTIZATION["q4f16_1"]
    model, _ = MODEL.quantize[quant.kind](cfg, quant)
    if args.device == "webgpu":
        auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(args.device)
    with target:
        mod, params, extern = model.export_tvm(model.get_default_spec(), allow_extern=True)
    mod = add_reset(mod)
    library = args.out / ("thinker-text.wasm" if args.device == "webgpu" else "thinker-text.dylib")
    schema = [dict(name=n, shape=list(p.shape), dtype=p.dtype) for n, p in params]
    artifact = export_contract(args.out, schema, cfg, library, args.device)
    metadata = dict(
        params=schema,
        artifact=artifact,
        thinker=dict(
            model_dtype=quant.model_dtype,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            active_vocab_size=cfg.vocab_size,
            context_window_size=cfg.context_window_size,
            prefill_chunk_size=cfg.prefill_chunk_size,
            **token_ids,
        ),
        model_type=MODEL.name,
        quantization="q4f16_1",
        context_window_size=cfg.context_window_size,
        prefill_chunk_size=cfg.prefill_chunk_size,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        max_batch_size=1,
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {
        "scope": "Thinker backbone with identical position IDs on all RoPE axes",
        "device": args.device,
        "models": [profile],
        "compiled_context": cfg.context_window_size,
        "source_parameters": len(mapping.param_map),
        "compiled_parameters": len(params),
        "interface_id": artifact["interface_id"],
    }
    path = args.out / "build-proof.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.export_only:
        print(json.dumps(receipt, indent=2))
        return
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                ext_mods=extern,
                variable_bounds=dict(
                    seq_len=cfg.prefill_chunk_size,
                    batch_size=1,
                    total_seq_len=cfg.context_window_size,
                ),
            ),
        )
    receipt["library_sha256"] = sha(library)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
