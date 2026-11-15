"""Build a standalone v2 text/audio Thinker component, not an Omni chat alias."""

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
from mlc_llm.model.qwen2_5_omni.thinker_loader import huggingface
from mlc_llm.model.qwen2_5_omni.thinker_model import Qwen25OmniThinker, Qwen25OmniThinkerConfig
from mlc_llm.protocol.artifact_manifest_v2 import compute_interface_id_v2, validate_compiled_v2
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support import auto_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=["metal", "webgpu"], default="webgpu")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--quantization", choices=["q0f16", "q0f32"], default="q0f16")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        config = Qwen25OmniThinkerConfig.from_dict(
            dict(
                hidden_act="silu",
                hidden_size=128,
                intermediate_size=256,
                num_attention_heads=2,
                num_hidden_layers=2,
                num_key_value_heads=1,
                rms_norm_eps=1e-6,
                rope_theta=1000000,
                vocab_size=64,
                context_window_size=32,
                prefill_chunk_size=16,
            )
        )
    else:
        config = Qwen25OmniThinkerConfig.from_file(args.source / "config.json")
        config.context_window_size = 4096
        config.prefill_chunk_size = 256
    config.max_batch_size = 1
    quant = QUANTIZATION[args.quantization]
    model = Qwen25OmniThinker(config)
    model.to(quant.model_dtype)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [dict(name=n, shape=list(p.shape), dtype=p.dtype) for n, p in params]

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    schema_id = (
        "sha256:"
        + hashlib.sha256(canonical(sorted(schema, key=lambda x: x["name"])).encode()).hexdigest()
    )
    tokens = dict(semantic="token.ids.v1", dtype="int32", shape=["tokens"], delivery="value")
    audio = dict(
        semantic="audio.embeddings.v1",
        dtype="float32",
        shape=["tokens", config.hidden_size],
        delivery="value",
    )
    output = dict(semantic="token.ids.v1", dtype="int32", shape=[1], delivery="stream")
    spec = dict(
        kind="autoregressive",
        contract="qwen2_5_omni.thinker_audio.v1",
        inputs=dict(token_ids=tokens, audio_embeddings=audio),
        outputs=dict(token_ids=output),
        weights=dict(manifest="tensor-cache.json", parameter_schema_id=schema_id),
        state=dict(kind="kv_cache", scope="request"),
    )
    package = dict(
        schema="mlc.model-package",
        schema_version=2,
        interface_id="sha256:" + "0" * 64,
        components=dict(thinker=spec),
        tasks={
            "thinker.generate": dict(
                inputs={
                    name: dict(spec=port, targets=[dict(component="thinker", port=name)])
                    for name, port in spec["inputs"].items()
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
        },
    )
    package["interface_id"] = compute_interface_id_v2(package)
    library = args.out / ("thinker.wasm" if args.device == "webgpu" else "thinker.dylib")
    sizes = [int(np.prod(p["shape"])) * (2 if p["dtype"] == "float16" else 4) for p in schema]
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
                    required_features=["shader-f16"]
                    if quant.model_dtype == "float16" and args.device == "webgpu"
                    else [],
                    max_storage_buffer_binding_size=max(sizes),
                    estimated_device_memory_bytes=sum(sizes),
                ),
            )
        ),
    )
    validate_compiled_v2(package, compiled)
    thinker = dict(
        model_dtype=quant.model_dtype,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        active_vocab_size=64 if args.tiny else 151665,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        eos_token_ids=[151643, 151645],
        context_window_size=config.context_window_size,
        prefill_chunk_size=config.prefill_chunk_size,
    )
    metadata = dict(
        params=schema,
        artifact=compiled,
        thinker=thinker,
        model_type="qwen2_5_omni_thinker_audio",
        quantization=args.quantization,
        context_window_size=config.context_window_size,
        prefill_chunk_size=config.prefill_chunk_size,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        max_batch_size=1,
        active_vocab_size=thinker["active_vocab_size"],
    )
    for name, value in [
        ("model-manifest.json", package),
        ("compiled-manifest.json", compiled),
        ("metadata.json", metadata),
        ("thinker-config.json", config.asdict()),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.tiny:
        import torch
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniTextConfig,
            Qwen2_5OmniThinkerTextModel,
        )

        torch.manual_seed(23)
        hfconfig = Qwen2_5OmniTextConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            rope_parameters=dict(
                rope_type="default", rope_theta=1000000, mrope_section=[8, 12, 12]
            ),
        )
        hfconfig._attn_implementation = "eager"
        hf = Qwen2_5OmniThinkerTextModel(hfconfig).eval()
        head = torch.nn.Linear(128, 64, bias=False).eval()
        state = {f"thinker.model.{n}": p.detach().numpy() for n, p in hf.state_dict().items()}
        state["thinker.lm_head.weight"] = head.weight.detach().numpy()
        mapping = huggingface(config, quant)
        values = {
            n: mapping.map_func[n](*[state[s] for s in mapping.param_map[n]]) for n, _ in params
        }
        tvmjs.dump_tensor_cache(values, str(args.out), encode_format="raw")
        cases = []
        for with_audio in [False, True]:
            ids = [3, 5, 7, 9, 11]
            features = np.zeros((0, 128), dtype="float32")
            with torch.no_grad():
                emb = hf.embed_tokens(torch.tensor([ids]))
                if with_audio:
                    features = np.linspace(-0.2, 0.2, 256, dtype="float32").reshape(2, 128)
                    emb[:, 1:3] = torch.from_numpy(features)
                    ids[1:3] = [151646, 151646]
                generated = []
                embeds = emb
                for _ in range(4):
                    logits = head(hf(inputs_embeds=embeds, use_cache=False).last_hidden_state)[
                        0, -1
                    ]
                    token = int(logits.argmax())
                    generated.append(token)
                    embeds = torch.cat([embeds, hf.embed_tokens(torch.tensor([[token]]))], dim=1)
            cases.append(
                dict(
                    token_ids=ids, audio_embeddings=features.tolist(), expected_token_ids=generated
                )
            )
        (args.out / "fixtures.json").write_text(json.dumps(cases, indent=2) + "\n")
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
                "seq_len": config.prefill_chunk_size,
                "batch_size": 1,
                "total_seq_len": config.context_window_size,
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
            dict(
                library=str(library),
                sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                tiny=args.tiny,
            )
        )
    )


if __name__ == "__main__":
    main()
