"""Build the standalone reference-mel conditioner for a bounded unpadded clip."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm_tirx
from tvm import relax
from tvm.contrib import tvmjs

from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
from mlc_llm.model.qwen2_5_omni.ecapa_model import (
    Qwen25OmniECAPA,
    Qwen25OmniECAPAConfig,
    huggingface,
)
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.support import auto_target
from mlc_llm.support.constants import MLC_DSO_SUFFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["llvm", "metal", "webgpu", "cuda"], default="webgpu")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    source = json.loads((args.source / "config.json").read_text())["token2wav_config"]["dit_config"]
    config = Qwen25OmniECAPAConfig(
        **{k: source[k] for k in Qwen25OmniECAPAConfig.__dataclass_fields__ if k in source}
    )
    if not config.min_frames <= args.max_frames <= 32768:
        parser.error(f"--max-frames must be between {config.min_frames} and 32768")
    args.out.mkdir(parents=True, exist_ok=True)
    model = Qwen25OmniECAPA(config)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    schema_id = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                sorted(schema, key=lambda x: x["name"]), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    mel = dict(
        semantic="audio.reference_mel.v1",
        dtype="float32",
        shape=["frames", config.mel_dim],
        delivery="value",
    )
    embedding = dict(
        semantic="audio.reference_embedding.v1",
        dtype="float32",
        shape=[config.enc_dim],
        delivery="value",
    )
    spec = dict(
        kind="encoder",
        contract="qwen2_5_omni.reference_conditioner.v1",
        inputs={"mel": mel},
        outputs={"embedding": embedding},
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components={"conditioner": spec},
        tasks={
            "audio.condition_reference": dict(
                inputs={"mel": dict(spec=mel, targets=[dict(component="conditioner", port="mel")])},
                outputs={
                    "embedding": dict(
                        spec=embedding,
                        source=dict(component="conditioner", port="embedding"),
                        optional=False,
                    )
                },
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    library = args.out / (
        "conditioner.wasm" if args.device == "webgpu" else f"conditioner.{MLC_DSO_SUFFIX}"
    )
    sizes = [int(np.prod(p["shape"])) * 4 for p in schema]
    activation = 3 * config.enc_channels[-1] * args.max_frames * 4
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components={
            "conditioner": dict(
                spec=spec,
                module=library.name,
                exports=dict(run="forward", forward="forward"),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(max(sizes), activation),
                    estimated_device_memory_bytes=sum(sizes) + 16 * activation,
                ),
            )
        },
    )
    validate_compiled_v2(package, compiled)
    metadata = dict(
        params=schema,
        artifact=compiled,
        source_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(inspect.getfile(Qwen25OmniECAPA))]
        },
        reference_conditioner=dict(
            mel_dim=config.mel_dim,
            embedding_dim=config.enc_dim,
            min_mel_frames=config.min_frames,
            max_mel_frames=args.max_frames,
        ),
        pipeline_parallel_stages=1,
        tensor_parallel_shards=1,
        prefill_chunk_size=200,
    )
    for name, value in [
        ("model-manifest.json", package),
        ("compiled-manifest.json", compiled),
        ("metadata.json", metadata),
        ("ecapa-config.json", vars(config)),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.weights:
        from safetensors import safe_open

        index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
        mapping = huggingface(config)
        requested = {n for names in mapping.param_map.values() for n in names}
        actual = {
            n
            for n in index
            if n.startswith("token2wav.code2wav_dit_model.input_embed.spk_encoder.")
        }
        if requested != actual:
            raise ValueError("Reference conditioner checkpoint coverage mismatch")
        state = {}
        for shard in sorted({index[n] for n in requested}):
            with safe_open(args.source / shard, framework="pt", device="cpu") as file:
                for name in requested:
                    if index[name] == shard:
                        state[name] = file.get_tensor(name).float().numpy()
        values = {
            name: mapping.map_func[name](*[state[n] for n in mapping.param_map[name]])
            for name, _ in params
        }
        for name, param in params:
            if values[name].shape != tuple(param.shape) or not np.isfinite(values[name]).all():
                raise ValueError(f"Invalid checkpoint parameter {name}")
        tvmjs.dump_tensor_cache(values, str(args.out), encode_format="raw")
    if args.device == "webgpu":
        auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(args.device)
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        pipeline = relax.get_pipeline(
            "mlc_llm",
            target=target,
            metadata=metadata,
            ext_mods=ext,
            variable_bounds={"frames": args.max_frames, "batch_size": 1, "vocab_size": 8},
        )
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=pipeline,
        )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {"library": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
