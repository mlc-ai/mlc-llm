"""Build a Qwen3-Omni audio encoder from a pinned Hugging Face snapshot."""

import argparse
import dataclasses
import hashlib
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

import tvm_tirx
from build_qwen3_thinker import sha, validate_source_identity
from safetensors import safe_open
from tvm import relax
from tvm.contrib import tvmjs

from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
from mlc_llm.model.qwen2_5_omni.audio_model import audio_positions
from mlc_llm.model.qwen3_omni.audio_model import (
    Qwen3OmniAudioConfig,
    Qwen3OmniAudioEncoder,
    stem_length,
)
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.support import auto_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("metal", "webgpu"), default="webgpu")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    validate_source_identity(args.repo, args.revision)
    out, device = args.out, args.device
    out.mkdir(parents=True, exist_ok=True)
    values = json.loads((args.source / "config.json").read_text())["thinker_config"]["audio_config"]
    config = Qwen3OmniAudioConfig(
        **{f.name: values[f.name] for f in dataclasses.fields(Qwen3OmniAudioConfig)}
    )
    model = Qwen3OmniAudioEncoder(config)
    mod, params, ext = model.export_tvm(model.get_default_spec(), allow_extern=True)
    schema = [
        dict(name=name, shape=list(tensor.shape), dtype=tensor.dtype) for name, tensor in params
    ]
    index_path = args.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = "thinker.audio_tower."
    mapped = {prefix + row["name"] for row in schema if row["name"] != "positions"}
    if mapped != {name for name in index if name.startswith(prefix)}:
        raise ValueError("Audio mapping does not cover the checkpoint namespace")
    source_info = dict(
        repo=args.repo,
        revision=args.revision,
        config_sha256=sha(args.source / "config.json"),
        index_sha256=sha(index_path),
    )
    if args.weights:
        if (out / "tensor-cache.json").exists():
            raise FileExistsError("Refusing to replace an existing tensor cache")
        tensors = {}
        source_info["shards_sha256"] = {}
        for shard in sorted({index[name] for name in mapped}):
            source_info["shards_sha256"][shard] = sha(args.source / shard)
            with safe_open(args.source / shard, framework="pt", device="cpu") as handle:
                for row in schema:
                    name = prefix + row["name"]
                    if name in mapped and index[name] == shard:
                        value = handle.get_tensor(name).float().numpy()
                        if list(value.shape) != row["shape"]:
                            raise ValueError(f"Audio weight shape mismatch: {name}")
                        tensors[row["name"]] = value
        tensors["positions"] = audio_positions(config)[: stem_length(config.n_window * 2)].astype(
            "float32"
        )
        if set(tensors) != {row["name"] for row in schema}:
            raise ValueError("Incomplete audio weights")
        tvmjs.dump_tensor_cache(tensors, str(out), encode_format="raw")
    (out / "source.json").write_text(json.dumps(source_info, indent=2) + "\n")
    schema_json = json.dumps(
        sorted(schema, key=lambda p: p["name"]), sort_keys=True, separators=(",", ":")
    )
    schema_id = "sha256:" + hashlib.sha256(schema_json.encode()).hexdigest()
    features = dict(
        semantic="audio.log_mel.v1",
        dtype="float32",
        shape=[config.num_mel_bins, "frames"],
        delivery="value",
    )
    embeddings = dict(
        semantic="audio.embeddings.v1",
        dtype="float32",
        shape=["audio_tokens", config.output_dim],
        delivery="value",
    )
    component = dict(
        kind="encoder",
        contract="qwen3_omni.audio_encoder.v1",
        inputs=dict(mel_features=features),
        outputs=dict(embeddings=embeddings),
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components=dict(audio=component),
        tasks={
            "audio.encode": dict(
                inputs=dict(
                    mel_features=dict(
                        spec=features, targets=[dict(component="audio", port="mel_features")]
                    )
                ),
                outputs=dict(
                    embeddings=dict(
                        spec=embeddings,
                        source=dict(component="audio", port="embeddings"),
                        optional=False,
                    )
                ),
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    sizes = [prod(row["shape"]) * 4 for row in schema]
    library = out / ("audio.wasm" if device == "webgpu" else "audio.dylib")
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components=dict(
            audio=dict(
                spec=component,
                module=library.name,
                exports=dict(run="encode_window", stem="stem", encode_window="encode_window"),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(sizes),
                    estimated_device_memory_bytes=sum(sizes),
                ),
            )
        ),
    )
    validate_compiled_v2(package, compiled)
    meta = dict(
        params=schema,
        artifact=compiled,
        audio_encoder=dict(
            hidden_size=config.d_model,
            n_window=config.n_window,
            n_window_infer=config.n_window_infer,
            max_encoded_window=104,
        ),
        pipeline_parallel_stages=1,
        tensor_parallel_shards=1,
        prefill_chunk_size=104,
    )
    for filename, payload in [
        ("metadata.json", meta),
        ("model-manifest.json", package),
        ("compiled-manifest.json", compiled),
    ]:
        (out / filename).write_text(json.dumps(payload, indent=2) + "\n")
    if args.export_only:
        return
    if device == "webgpu":
        auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(device)
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=meta,
                ext_mods=ext,
                variable_bounds=dict(
                    mel_frames=2 * config.n_window, window_frames=104, batch_size=1, vocab_size=8
                ),
            ),
        )
    proof = dict(
        source=source_info,
        parameter_count=len(params),
        interface_id=package["interface_id"],
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        max_cnn_window=2 * config.n_window,
        max_attention_window=104,
        device=device,
    )
    (out / "build-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
