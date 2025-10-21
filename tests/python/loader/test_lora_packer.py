import tempfile
from pathlib import Path

import numpy as np
import torch

from mlc_llm.loader.lora_packer import pack_lora_adapter


def _create_fake_peft_adapter(tmpdir: Path) -> Path:
    """Create a minimal PEFT-like LoRA checkpoint for testing."""

    in_feat, out_feat, r = 4, 3, 2

    a = torch.randn(r, in_feat, dtype=torch.float32)
    b = torch.randn(out_feat, r, dtype=torch.float32)

    state_dict = {
        "layer0.lora_A.weight": a,
        "layer0.lora_B.weight": b,
    }

    ckpt_path = tmpdir / "adapter_model.bin"
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def test_pack_lora_adapter_roundtrip(tmp_path):
    ckpt = _create_fake_peft_adapter(tmp_path)
    out_file = tmp_path / "packed" / "adapter.npz"

    packed_path = pack_lora_adapter(ckpt, out_file)

    # Check files exist
    assert packed_path.exists()
    manifest_json = packed_path.with_suffix(".json")
    assert manifest_json.exists()

    # Load npz and verify delta matrix matches B @ A
    data = np.load(packed_path)
    delta_key = "delta.layer0"
    assert delta_key in data.files

    with torch.no_grad():
        tensors = torch.load(ckpt, map_location="cpu")
        delta_ref = tensors["layer0.lora_B.weight"] @ tensors["layer0.lora_A.weight"]

    np.testing.assert_allclose(data[delta_key], delta_ref.numpy().astype(np.float16), rtol=1e-3, atol=1e-3) 