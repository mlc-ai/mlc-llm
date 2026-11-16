"""Export the pinned Qwen2.5-Omni-3B speaker assets for the browser example."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

SOURCE_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
SOURCE_SHA256 = "6a05609b28f5d42b7b748f0f07592545c8f1f6885b9ae8fff64baf56e86b2a18"
SPEAKERS = {"Ethan": 151870, "Chelsie": 151872}


def validate_speakers(source):
    """Return JSON-compatible FP32 conditioning after validating the pinned ABI."""
    if not isinstance(source, dict) or source.keys() != SPEAKERS.keys():
        raise ValueError("Expected the pinned Ethan and Chelsie voices")
    result = {}
    for name, bos in SPEAKERS.items():
        voice = source[name]
        if not isinstance(voice, dict) or set(voice) != {"bos_token", "cond", "ref_mel"}:
            raise ValueError(f"Invalid {name} speaker fields")
        if type(voice["bos_token"]) is not int or voice["bos_token"] != bos:
            raise ValueError(f"Invalid {name} speaker BOS")
        value = {"bos_token": bos}
        for field, shape in (("cond", (1, 192)), ("ref_mel", (1, 400, 80))):
            tensor = voice[field]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid {name} {field} tensor")
            value[field] = tensor.tolist()
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned spk_dict.pt file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        parser.error("spk_dict.pt does not match the pinned Qwen2.5-Omni-3B revision")
    values = validate_speakers(
        torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
    )
    encoded = (json.dumps(values, allow_nan=False) + "\n").encode()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "speakers.json").write_bytes(encoded)
    receipt = dict(
        repo_id="Qwen/Qwen2.5-Omni-3B",
        revision=SOURCE_REVISION,
        filename="spk_dict.pt",
        source_sha256=digest,
        speakers_sha256=hashlib.sha256(encoded).hexdigest(),
        speakers=SPEAKERS,
    )
    (args.out / "speaker-export.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
