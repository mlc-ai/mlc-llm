"""Build a standalone Qwen2.5 Omni waveform decoder component and validated v2 contracts."""

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
from mlc_llm.model.qwen2_5_omni.vocoder_loader import huggingface
from mlc_llm.model.qwen2_5_omni.vocoder_model import Qwen25OmniVocoder, Qwen25OmniVocoderConfig
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
    parser.add_argument("--max-frames", type=int, default=512)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        config = Qwen25OmniVocoderConfig(
            mel_dim=4,
            upsample_initial_channel=16,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
        )
    else:
        source_config = json.loads((args.source / "config.json").read_text())["token2wav_config"][
            "bigvgan_config"
        ]
        config = Qwen25OmniVocoderConfig(
            **{
                k: source_config[k]
                for k in Qwen25OmniVocoderConfig.__dataclass_fields__
                if k in source_config
            }
        )
    model = Qwen25OmniVocoder(config)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    schema_id = (
        "sha256:"
        + hashlib.sha256(canonical(sorted(schema, key=lambda p: p["name"])).encode()).hexdigest()
    )
    input_port = dict(
        semantic="audio.synthesis_log_mel.v1",
        dtype="float32",
        shape=[config.mel_dim, "frames"],
        delivery="value",
    )
    output_port = dict(
        semantic="audio.pcm_24khz.v1",
        dtype="float32",
        shape=["samples"],
        delivery="value",
    )
    spec = dict(
        kind="waveform_decoder",
        contract="qwen2_5_omni.waveform_decoder.v1",
        inputs={"log_mel": input_port},
        outputs={"pcm": output_port},
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components={"vocoder": spec},
        tasks={
            "audio.decode_waveform": dict(
                inputs={
                    "log_mel": dict(
                        spec=input_port, targets=[dict(component="vocoder", port="log_mel")]
                    )
                },
                outputs={
                    "pcm": dict(
                        spec=output_port,
                        source=dict(component="vocoder", port="pcm"),
                        optional=False,
                    )
                },
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    suffix = ".wasm" if args.device == "webgpu" else f".{MLC_DSO_SUFFIX}"
    library = args.out / ("vocoder" + suffix)
    sizes = [int(np.prod(p["shape"])) * 4 for p in schema]
    length = args.max_frames
    largest_activation = length * config.upsample_initial_channel
    for stage, rate in enumerate(config.upsample_rates):
        length *= rate
        largest_activation = max(
            largest_activation, length * (config.upsample_initial_channel // 2 ** (stage + 1))
        )
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components={
            "vocoder": dict(
                spec=spec,
                module=library.name,
                exports=dict(run="forward", forward="forward"),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(max(sizes), 2 * largest_activation * 4),
                    estimated_device_memory_bytes=sum(sizes) + 16 * largest_activation * 4,
                ),
            )
        },
    )
    validate_compiled_v2(package, compiled)
    source_files = [
        Path(__file__),
        Path(inspect.getfile(Qwen25OmniVocoder)),
        Path(inspect.getfile(huggingface)),
    ]
    metadata = dict(
        source_files={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        params=schema,
        artifact=compiled,
        waveform_decoder=dict(
            mel_dim=config.mel_dim,
            sampling_rate=24000,
            samples_per_frame=config.samples_per_frame,
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
        ("vocoder-config.json", vars(config)),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.weights:
        mapping = huggingface(config)
        if args.tiny:
            import torch
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniBigVGANConfig,
                Qwen2_5OmniToken2WavBigVGANModel,
            )

            torch.manual_seed(29)
            hf = Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniBigVGANConfig(**vars(config))).eval()
            state = {
                "token2wav.code2wav_bigvgan_model." + n: v.detach().numpy()
                for n, v in hf.state_dict().items()
            }
        else:
            from safetensors import safe_open

            index = json.loads((args.source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            state = {}
            requested = {n for names in mapping.param_map.values() for n in names}
            checkpoint = {n for n in index if n.startswith("token2wav.code2wav_bigvgan_model.")}
            if checkpoint != requested:
                raise ValueError("Vocoder checkpoint parameter coverage mismatch")
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
            {
                "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "tiny": args.tiny,
            }
        )
    )


if __name__ == "__main__":
    main()
