"""Build a standalone real Qwen2.5 Omni audio component and validated v2 contracts."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm_tirx
from tvm import relax
from tvm.contrib import tvmjs

from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
from mlc_llm.model.qwen2_5_omni.audio_loader import huggingface
from mlc_llm.model.qwen2_5_omni.audio_model import Qwen25OmniAudioConfig, Qwen25OmniAudioEncoder
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.support import auto_target
from mlc_llm.support.constants import MLC_DSO_SUFFIX


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["llvm", "metal", "webgpu", "cuda"], default="webgpu")
    parser.add_argument(
        "--tiny", action="store_true", help="Synthetic fixture, never pretrained evidence"
    )
    parser.add_argument("--weights", action="store_true")
    parser.add_argument(
        "--export-only", action="store_true", help="Validate/export schemas without a library"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        config = Qwen25OmniAudioConfig(
            num_mel_bins=4,
            d_model=16,
            encoder_attention_heads=4,
            encoder_ffn_dim=32,
            encoder_layers=2,
            output_dim=8,
        )
    else:
        source_config = json.loads((args.source / "config.json").read_text())["thinker_config"][
            "audio_config"
        ]
        config = Qwen25OmniAudioConfig(
            **{
                k: source_config[k]
                for k in Qwen25OmniAudioConfig.__dataclass_fields__
                if k in source_config
            }
        )
    model = Qwen25OmniAudioEncoder(config)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    schema_id = (
        "sha256:"
        + hashlib.sha256(canonical(sorted(schema, key=lambda p: p["name"])).encode()).hexdigest()
    )
    input_port = dict(
        semantic="audio.log_mel.v1",
        dtype="float32",
        shape=[config.num_mel_bins, "frames"],
        delivery="value",
    )
    output_port = dict(
        semantic="audio.embeddings.v1",
        dtype="float32",
        shape=["tokens", config.output_dim],
        delivery="value",
    )
    spec = dict(
        kind="encoder",
        contract="qwen2_5_omni.audio_encoder.v1",
        inputs={"mel_features": input_port},
        outputs={"embeddings": output_port},
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components={"audio": spec},
        tasks={
            "audio.encode": dict(
                inputs={
                    "mel_features": dict(
                        spec=input_port, targets=[dict(component="audio", port="mel_features")]
                    )
                },
                outputs={
                    "embeddings": dict(
                        spec=output_port,
                        source=dict(component="audio", port="embeddings"),
                        optional=False,
                    )
                },
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    suffix = ".wasm" if args.device == "webgpu" else f".{MLC_DSO_SUFFIX}"
    library = args.out / ("audio" + suffix)
    sizes = [int(np.prod(p["shape"])) * 4 for p in schema]
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components={
            "audio": dict(
                spec=spec,
                module=library.name,
                exports=dict(run="encode_chunk", encode_chunk="encode_chunk", project="project"),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(sizes),
                    estimated_device_memory_bytes=sum(sizes),
                ),
            )
        },
    )
    validate_compiled_v2(package, compiled)
    metadata = dict(
        params=schema,
        artifact=compiled,
        audio_encoder=dict(
            hidden_size=config.d_model, n_window=config.n_window, max_pooled_frames=7500
        ),
        pipeline_parallel_stages=1,
        tensor_parallel_shards=1,
        prefill_chunk_size=200,
    )
    for name, value in [
        ("model-manifest.json", package),
        ("compiled-manifest.json", compiled),
        ("metadata.json", metadata),
        ("audio-config.json", vars(config)),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.weights:
        mapping = huggingface(config)
        if args.tiny:
            import torch
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoder,
                Qwen2_5OmniAudioEncoderConfig,
            )

            torch.manual_seed(17)
            hf = Qwen2_5OmniAudioEncoder(Qwen2_5OmniAudioEncoderConfig(**vars(config))).eval()
            state = {
                "thinker.audio_tower." + n: v.detach().numpy() for n, v in hf.state_dict().items()
            }
        else:
            from safetensors import safe_open

            index = json.loads((args.source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            state = {}
            requested = {n for names in mapping.param_map.values() for n in names}
            for shard in sorted({index[n] for n in requested}):
                with safe_open(args.source / shard, framework="pt", device="cpu") as file:
                    for name in requested:
                        if index[name] == shard:
                            state[name] = file.get_tensor(name).float().numpy()
        values = {
            name: mapping.map_func[name](*[state[n] for n in mapping.param_map[name]])
            for name, _ in params
        }
        for name, parameter in params:
            assert values[name].shape == tuple(parameter.shape), name
        tvmjs.dump_tensor_cache(values, str(args.out), encode_format="raw")
        del values, state
    if args.export_only:
        return
    if args.device == "webgpu":
        auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(args.device)
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        pipeline = relax.get_pipeline(
            "mlc_llm",
            target=target,
            metadata=metadata,
            ext_mods=ext,
            variable_bounds={"pooled_frames": 7500, "batch_size": 1, "vocab_size": 8},
        )
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=pipeline,
        )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "tiny": args.tiny,
            }
        )
    )


if __name__ == "__main__":
    main()
