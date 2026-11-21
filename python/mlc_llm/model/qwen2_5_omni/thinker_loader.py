"""Map the dense text/audio Thinker weights from a full Omni checkpoint."""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .thinker_model import Qwen25OmniThinker

huggingface = make_standard_hf_loader(
    model_cls=Qwen25OmniThinker,
    qkv_target_name="c_attn",
    add_qkv_bias=True,
    name_transform=lambda name: f"thinker.{name}",
    add_unused=["rotary_emb.inv_freq"],
)
