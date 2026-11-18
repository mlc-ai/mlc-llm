"""Build a standalone Qwen2.5 Omni flow predictor component and validated v2 contracts."""

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
from mlc_llm.model.qwen2_5_omni.dit_loader import huggingface
from mlc_llm.model.qwen2_5_omni.dit_model import Qwen25OmniDiT, Qwen25OmniDiTConfig
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
    if args.max_frames < 2:
        parser.error("--max-frames must be at least two")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        config = Qwen25OmniDiTConfig(
            hidden_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            head_dim=8,
            emb_dim=8,
            mel_dim=4,
            enc_dim=4,
            enc_emb_dim=4,
            num_embeds=17,
            block_size=4,
            look_ahead_layers=(1,),
            look_backward_layers=(0, 2),
        )
    else:
        source_config = json.loads((args.source / "config.json").read_text())["token2wav_config"][
            "dit_config"
        ]
        for name, legacy in (
            ("hidden_size", "dim"),
            ("num_hidden_layers", "depth"),
            ("num_attention_heads", "heads"),
        ):
            if name not in source_config and legacy in source_config:
                source_config[name] = source_config[legacy]
        config = Qwen25OmniDiTConfig(
            **{
                k: source_config[k]
                for k in Qwen25OmniDiTConfig.__dataclass_fields__
                if k in source_config
            }
        )
    if args.max_frames > config.max_position_embeddings or args.max_frames % config.repeats:
        parser.error("Frame bound must be a codec-repeat multiple within the positional limit")
    model = Qwen25OmniDiT(config)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    schema_id = (
        "sha256:"
        + hashlib.sha256(canonical(sorted(schema, key=lambda p: p["name"])).encode()).hexdigest()
    )

    def port(semantic, shape, dtype="float32"):
        return dict(semantic=semantic, dtype=dtype, shape=shape, delivery="value")

    inputs = dict(
        noise=port("audio.flow_state.v1", ["frames", config.mel_dim]),
        time=port("flow.time.v1", [1]),
        code=port("audio.codec_ids.v1", ["codes"], "int32"),
        reference=port("audio.reference_embedding.v1", [config.enc_dim]),
        speaker=port("audio.speaker_embedding.v1", [config.enc_emb_dim]),
    )
    output_port = port("audio.flow_velocity.v1", ["frames", config.mel_dim])
    spec = dict(
        kind="transform",
        contract="qwen2_5_omni.flow_predictor.v1",
        inputs=inputs,
        outputs={"velocity": output_port},
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="none", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components={"dit": spec},
        tasks={
            "audio.predict_flow": dict(
                inputs={
                    name: dict(spec=p, targets=[dict(component="dit", port=name)])
                    for name, p in inputs.items()
                },
                outputs={
                    "velocity": dict(
                        spec=output_port,
                        source=dict(component="dit", port="velocity"),
                        optional=False,
                    )
                },
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    suffix = ".wasm" if args.device == "webgpu" else f".{MLC_DSO_SUFFIX}"
    library = args.out / ("dit" + suffix)
    sizes = [int(np.prod(p["shape"])) * 4 for p in schema]
    largest_activation = max(
        args.max_frames * config.hidden_size * config.ff_mult,
        config.num_attention_heads * args.max_frames**2,
    )
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components={
            "dit": dict(
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
        Path(inspect.getfile(Qwen25OmniDiT)),
        Path(inspect.getfile(huggingface)),
    ]
    metadata = dict(
        source_files={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        params=schema,
        artifact=compiled,
        flow_predictor=dict(
            mel_dim=config.mel_dim,
            reference_dim=config.enc_dim,
            speaker_dim=config.enc_emb_dim,
            repeats=config.repeats,
            num_embeds=config.num_embeds,
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
        ("dit-config.json", vars(config)),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.weights:
        mapping = huggingface(config)
        if args.tiny:
            import torch
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniDiTConfig as HFConfig,
            )
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniToken2WavDiTModel,
            )

            torch.manual_seed(31)
            hf = Qwen2_5OmniToken2WavDiTModel(
                HFConfig(
                    **{k: v for k, v in vars(config).items() if k != "rope_theta"},
                    rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
                    enc_channels=(8, 8, 8, 8, 24),
                    enc_attention_channels=4,
                    enc_res2net_scale=2,
                    enc_se_channels=4,
                )
            ).eval()
            state = {
                "token2wav.code2wav_dit_model." + n: v.detach().numpy()
                for n, v in hf.state_dict().items()
            }
            state["token2wav.code2wav_dit_model.rotary_embed.inv_freq"] = (
                hf.rotary_embed.inv_freq.numpy()
            )
        else:
            from safetensors import safe_open

            index = json.loads((args.source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            state = {}
            requested = {n for names in mapping.param_map.values() for n in names}
            checkpoint = {
                n
                for n in index
                if n.startswith("token2wav.code2wav_dit_model.")
                and ".spk_encoder." not in n
                and not n.endswith("rotary_embed.inv_freq")
            }
            if checkpoint != requested:
                raise ValueError("DiT flow checkpoint parameter coverage mismatch")
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
            variable_bounds={
                "frames": args.max_frames,
                "codes": args.max_frames // config.repeats,
                "batch_size": 1,
                "vocab_size": 8,
            },
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
