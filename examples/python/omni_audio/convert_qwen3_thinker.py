"""Convert pinned Qwen3-Omni Thinker weights using the shared Qwen3 MoE loader."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_qwen3_thinker import MODEL, load_profile, sha
from tvm_tirx import cpu

from mlc_llm.interface.convert_weight import convert_weight
from mlc_llm.quantization import QUANTIZATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, expected_names, profile = load_profile(args.source, args.repo, args.revision)
    build = json.loads((args.compiled / "build-proof.json").read_text())
    if not any(
        all(
            model[key] == profile[key]
            for key in ("repo", "revision", "index_sha256", "config_sha256")
        )
        for model in build["models"]
    ):
        raise ValueError("Thinker checkpoint does not match the compiled profile")
    index_path = args.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    mapping = {name: shard for name, shard in index.items() if name in expected_names}
    required = set(mapping.values())
    source_hashes = {name: sha(args.source / name) for name in sorted(required)}

    source_format = "huggingface-safetensor"
    upstream_loader = MODEL.source[source_format]

    def thinker_loader(cfg, quant):
        result = upstream_loader(cfg, quant)
        result.param_map = {
            name: [f"thinker.{source}" for source in sources]
            for name, sources in result.param_map.items()
        }
        mapped_names = {name for names in result.param_map.values() for name in names}
        result.unused_params = set(index) - mapped_names
        if mapped_names != expected_names:
            raise ValueError("Thinker mapping does not cover the pinned text tensors")
        return result

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "tensor-cache.json"
    if cache.exists():
        raise FileExistsError(f"Existing cache: {cache}")
    config_path = args.out / "thinker-config.json"
    config_path.write_text(json.dumps(config.asdict(), indent=2) + "\n")
    index_file = args.out / "thinker-source.index.json"
    index_file.write_text(
        json.dumps(
            {
                "weight_map": {
                    name: str((args.source / shard).resolve()) for name, shard in mapping.items()
                }
            },
            indent=2,
        )
        + "\n"
    )
    convert_weight(
        config=config_path,
        quantization=QUANTIZATION["q4f16_1"],
        model=replace(MODEL, source={source_format: thinker_loader}),
        device=cpu(),
        source=index_file,
        source_format=source_format,
        output=args.out,
        encode_format="raw",
    )
    compiled_params = json.loads((args.compiled / "metadata.json").read_text())["params"]
    expected = {param["name"]: (param["shape"], param["dtype"]) for param in compiled_params}
    actual = {}
    for shard in json.loads(cache.read_text())["records"]:
        path = args.out / shard["dataPath"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != shard["md5sum"] or path.stat().st_size != shard["nbytes"]:
            raise ValueError(f"Converted Thinker cache checksum mismatch: {path}")
        for param in shard["records"]:
            name = param["name"]
            if name in actual:
                raise ValueError(f"Duplicate Thinker cache parameter: {name}")
            actual[name] = (param["shape"], param["dtype"])
    if actual != expected:
        raise ValueError("Converted Thinker cache differs from the compiled schema")
    (args.out / "weight-conversion.json").write_text(
        json.dumps(
            {
                "source": profile,
                "source_shards_sha256": source_hashes,
                "parameter_count": len(actual),
                "tensor_cache_sha256": sha(cache),
                "library_sha256": build.get("library_sha256"),
                "quantization": "q4f16_1",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
