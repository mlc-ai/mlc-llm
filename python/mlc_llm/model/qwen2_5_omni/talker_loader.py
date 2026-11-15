"""Map standalone Talker parameters from the full Omni checkpoint."""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .talker_model import Qwen25OmniTalker


def _checkpoint_name(name):
    if name == "lm_head.weight":
        return "talker.codec_head.weight"
    return f"talker.{name}"


huggingface = make_standard_hf_loader(
    model_cls=Qwen25OmniTalker,
    qkv_target_name="c_attn",
    add_qkv_bias=True,
    name_transform=_checkpoint_name,
    add_unused=["rotary_emb.inv_freq"],
)
