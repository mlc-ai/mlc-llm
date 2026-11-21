"""Reference conditioner parity, including reflected boundaries and null mel."""

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.qwen2_5_omni.ecapa_model import (
    Qwen25OmniECAPA,
    Qwen25OmniECAPAConfig,
    huggingface,
)


@pytest.fixture(scope="module")
def models():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniECAPAConfig(
        mel_dim=4,
        enc_dim=4,
        enc_channels=(8, 8, 8, 8, 24),
        enc_attention_channels=4,
        enc_se_channels=4,
    )
    torch.manual_seed(71)
    reference = hf.ECAPA_TimeDelayNet(hf.Qwen2_5OmniDiTConfig(**vars(config))).eval()
    model = Qwen25OmniECAPA(config)
    mod, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    state = reference.state_dict()
    assert {name for name, _ in named} == set(state)
    parameters = [tvm_tirx.runtime.tensor(state[name].detach().numpy()) for name, _ in named]
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    return config, reference, vm, parameters, torch


@pytest.mark.parametrize("frames", [5, 6, 9, 17, 32])
@pytest.mark.parametrize("null", [False, True])
def test_reference_conditioner(models, frames, null):
    config, reference, vm, params, torch = models
    mel = np.random.default_rng(frames).normal(size=(frames, config.mel_dim)).astype("float32")
    if null:
        mel[:] = 0
    with torch.no_grad():
        expected = reference(torch.from_numpy(mel[None]))[0].numpy()
    actual = vm["forward"](tvm_tirx.runtime.tensor(mel), params).numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    if null:
        assert np.max(np.abs(expected)) > 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enc_channels": (8, 16, 16, 16, 48)},
        {"enc_channels": (8, 8, 8, 8, 16)},
        {"enc_kernel_sizes": (4, 3, 3, 3, 1)},
        {"enc_res2net_scale": 3},
        {"enc_dilations": (1, 2)},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Qwen25OmniECAPAConfig(**kwargs)


def test_default_schema():
    hf = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniECAPAConfig()
    model = Qwen25OmniECAPA(config)
    _, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    assert len(named) == 40
    reference = hf.ECAPA_TimeDelayNet(hf.Qwen2_5OmniDiTConfig()).state_dict()
    assert {name: tuple(param.shape) for name, param in named} == {
        name: tuple(param.shape) for name, param in reference.items()
    }
    mapping = huggingface(config)
    assert set(mapping.param_map) == {name for name, _ in named}
    for name, _ in named:
        assert mapping.param_map[name] == [
            f"token2wav.code2wav_dit_model.input_embed.spk_encoder.{name}"
        ]
    assert config.min_frames == 5
    with pytest.raises(ValueError):
        model.to("float16")
