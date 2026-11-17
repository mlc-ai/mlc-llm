"""Compile the pinned Omni Whisper PCM frontend as an independent graph component."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm_tirx
from transformers import WhisperFeatureExtractor
from tvm import relax
from tvm.contrib import tvmjs

from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
from mlc_llm.model.qwen2_5_omni.audio_processing import (
    Qwen25OmniAudioFeatures,
    whisper_frontend_parameters,
)
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.support import auto_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["metal", "webgpu"], default="webgpu")
    parser.add_argument(
        "--unpadded", action="store_true", help="Use Qwen3-Omni variable-length PCM framing"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((args.source / "preprocessor_config.json").read_text())
    for name, expected in [
        ("feature_size", 128),
        ("n_fft", 400),
        ("hop_length", 160),
        ("sampling_rate", 16000),
        ("dither", 0.0),
        ("padding_value", 0.0),
        ("padding_side", "right"),
    ]:
        if raw[name] != expected:
            raise ValueError(f"Unsupported frontend {name}")
    processor = WhisperFeatureExtractor.from_dict(raw)
    model = Qwen25OmniAudioFeatures(processor.n_samples, unpadded=args.unpadded)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    values = whisper_frontend_parameters(processor.mel_filters)
    tvmjs.dump_tensor_cache(values, str(args.out), encode_format="raw")
    schema = [dict(name=n, shape=list(p.shape), dtype=p.dtype) for n, p in params]
    schema_id = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                sorted(schema, key=lambda p: p["name"]), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    pcm = dict(semantic="audio.pcm_16khz.v1", dtype="float32", shape=["samples"], delivery="value")
    mel = dict(
        semantic="audio.log_mel.v1", dtype="float32", shape=[128, "frames"], delivery="value"
    )
    spec = dict(
        kind="transform",
        contract=(
            "qwen3_omni.audio_features.v1" if args.unpadded else "qwen2_5_omni.audio_features.v1"
        ),
        inputs={"pcm": pcm},
        outputs={"mel_features": mel},
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components={"features": spec},
        tasks={
            "audio.features": dict(
                inputs={"pcm": dict(spec=pcm, targets=[dict(component="features", port="pcm")])},
                outputs={
                    "mel_features": dict(
                        spec=mel,
                        source=dict(component="features", port="mel_features"),
                        optional=False,
                    )
                },
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    library = args.out / ("features.wasm" if args.device == "webgpu" else "features.dylib")
    sizes = [int(np.prod(p["shape"])) * 4 for p in schema]
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components={
            "features": dict(
                spec=spec,
                module=library.name,
                exports=dict(run="forward", forward="forward"),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(
                        max(sizes), processor.nb_max_frames * 402 * 4
                    ),
                    estimated_device_memory_bytes=sum(sizes),
                ),
            )
        },
    )
    validate_compiled_v2(package, compiled)
    metadata = dict(
        params=schema,
        artifact=compiled,
        audio_processor=dict(
            num_samples=processor.n_samples, sampling_rate=16000, hop_length=160, num_mel_bins=128
        ),
        pipeline_parallel_stages=1,
        tensor_parallel_shards=1,
        prefill_chunk_size=200,
    )
    for name, value in [
        ("model-manifest.json", package),
        ("compiled-manifest.json", compiled),
        ("metadata.json", metadata),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(args.device)
    bounds = {"batch_size": 1, "vocab_size": 8}
    if args.unpadded:
        bounds["num_samples"] = processor.n_samples
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                ext_mods=ext,
                variable_bounds=bounds,
            ),
        )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "samples": processor.n_samples,
            }
        )
    )


if __name__ == "__main__":
    main()
