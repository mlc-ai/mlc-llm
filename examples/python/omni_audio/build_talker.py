"""Bounded standalone pretrained Talker build; host reply alignment stays explicit."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm_tirx
from safetensors import safe_open
from tvm import relax
from tvm.contrib import tvmjs

from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
from mlc_llm.model.qwen2_5_omni.talker_loader import huggingface
from mlc_llm.model.qwen2_5_omni.talker_model import Qwen25OmniTalker, Qwen25OmniTalkerConfig
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support import auto_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["metal", "webgpu"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", action="store_true")
    parser.add_argument(
        "--export-only", action="store_true", help="Validate/export schemas without a library"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source
    config = Qwen25OmniTalkerConfig.from_file(source / "config.json")
    expected = dict(
        context_window_size=32768,
        hidden_size=896,
        embedding_size=2048,
        vocab_size=8448,
        num_hidden_layers=24,
    )
    if any(getattr(config, key) != value for key, value in expected.items()):
        parser.error("This component builder requires the pinned Qwen2.5-Omni-3B Talker dimensions")
    raw_config = json.loads((source / "config.json").read_text())["talker_config"]
    for name, value in dict(
        tts_codec_start_token_id=8293,
        tts_codec_pad_token_id=8292,
        tts_codec_end_token_id=8294,
        tts_codec_mask_token_id=8296,
    ).items():
        if raw_config.get(name) != value:
            parser.error(f"Unexpected Talker special token {name}")
    config.prefill_chunk_size = 256
    config.max_batch_size = 1
    model = Qwen25OmniTalker(config)
    model.to("float32")
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [dict(name=n, shape=list(v.shape), dtype=v.dtype) for n, v in params]
    metadata = dict(
        params=schema,
        model_type="qwen25_omni_talker_component",
        quantization="q0f32",
        context_window_size=32768,
        prefill_chunk_size=256,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        max_batch_size=1,
        talker=dict(
            model_dtype="float32",
            embedding_size=2048,
            hidden_size=896,
            vocab_size=8448,
            codec_bos_token_id=8293,
            codec_pad_token_id=8292,
            codec_eos_token_id=8294,
            context_window_size=32768,
            prefill_chunk_size=256,
        ),
    )

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    schema_id = (
        "sha256:"
        + hashlib.sha256(canonical(sorted(schema, key=lambda x: x["name"])).encode()).hexdigest()
    )
    prefix = dict(
        semantic="audio.talker_prefix.v1", dtype="float32", shape=["prefix", 2048], delivery="value"
    )
    reply = dict(
        semantic="audio.talker_reply.v1", dtype="float32", shape=["reply", 2048], delivery="value"
    )
    code = dict(semantic="audio.codec_ids.v1", dtype="int32", shape=["codes"], delivery="value")
    spec = dict(
        kind="autoregressive",
        contract="qwen2_5_omni.talker.v1",
        inputs=dict(prefix_embeddings=prefix, reply_embeddings=reply),
        outputs=dict(code=code),
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="kv_cache", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components=dict(talker=spec),
        tasks={
            "talker.generate": dict(
                inputs={
                    name: dict(spec=port, targets=[dict(component="talker", port=name)])
                    for name, port in spec["inputs"].items()
                },
                outputs=dict(
                    code=dict(
                        spec=code, source=dict(component="talker", port="code"), optional=False
                    )
                ),
                edges=[],
            )
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    compiled = dict(
        schema="mlc.compiled-program",
        schema_version=2,
        interface_id=package["interface_id"],
        components=dict(
            talker=dict(
                spec=spec,
                module="talker.wasm" if args.device == "webgpu" else "talker.dylib",
                exports=dict(
                    run="prefill",
                    embed="embed",
                    project="project",
                    prefill="prefill",
                    decode="decode",
                    create_state="create_tir_paged_kv_cache",
                    reset_state="reset_state",
                ),
                resources=dict(
                    required_features=[],
                    max_storage_buffer_binding_size=max(
                        int(np.prod(p["shape"])) * 4 for p in schema
                    ),
                    estimated_device_memory_bytes=sum(int(np.prod(p["shape"])) * 4 for p in schema),
                ),
            )
        ),
    )
    validate_compiled_v2(package, compiled)
    metadata["artifact"] = compiled
    for name, value in [("model-manifest.json", package), ("compiled-manifest.json", compiled)]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "talker-config.json").write_text(json.dumps(config.asdict(), indent=2) + "\n")
    if args.weights:
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        mapping = huggingface(config, QUANTIZATION["q0f32"])
        expected = {n for n in index if n.startswith("talker.")}
        used = {n for names in mapping.param_map.values() for n in names}
        if expected != used or len(used) != 293:
            raise ValueError("Checkpoint must match all 293 pinned Talker tensors")

        def values():
            for name, param in params:
                arrays = []
                for key in mapping.param_map[name]:
                    with safe_open(source / index[key], framework="pt", device="cpu") as f:
                        arrays.append(f.get_tensor(key).float().numpy())
                value = mapping.map_func[name](*arrays)
                if value.shape != tuple(param.shape) or value.dtype != np.float32:
                    raise ValueError(f"Unexpected converted parameter {name}")
                yield name, value

        tvmjs.dump_tensor_cache(values(), str(args.out), encode_format="raw")
    if args.export_only:
        return
    if args.device == "webgpu":
        auto_target.PRESET["webgpu:generic"]["target"]["host"]["mcpu"] = "mvp"
    target, build = auto_target.detect_target_and_host(args.device)
    lib = args.out / ("talker.wasm" if args.device == "webgpu" else "talker.dylib")
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=lib, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                ext_mods=ext,
                variable_bounds={"seq_len": 256, "batch_size": 1, "total_seq_len": 32768},
            ),
        )
    receipt = dict(
        device=args.device,
        library=lib.name,
        sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        compiled_context=32768,
        compiled_prefill=256,
        parameter_count=len(params),
        parameter_bytes=sum(int(np.prod(p["shape"])) * 4 for p in schema),
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        source_index_sha256=hashlib.sha256(
            (source / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
    )
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
