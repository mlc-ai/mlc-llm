"""Configuration classes for Qwen2 VL model."""

import dataclasses
from typing import Any, Dict, Optional

from mlc_llm.support.config import ConfigBase
from mlc_llm.model.qwen2.qwen2_model import QWen2Config

@dataclasses.dataclass
class QWen2VLVisionConfig(ConfigBase):
    """Configuration for the vision part of Qwen2 VL."""
    
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int = 14
    merge_size: int = 2
    image_size: int = 448  # Default max image size
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    max_patches: int = 1024  # Maximum number of patches after merging
    hidden_act: str = "gelu"
    dtype: str = "float32"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    num_rope_scales: int = 4
    rope_theta: float = 10000


def qwen2vl_vision_from_official(vision_cfg: Dict[str, Any]) -> QWen2VLVisionConfig:
    """
    Map HF Qwen2-VL 'vision_config' dict to our QWen2VLVisionConfig dataclass.
    - Prefer 'embed_dim' for hidden size (fallback to 'hidden_size' if needed).
    - Compute intermediate_size = int(hidden_size * mlp_ratio).
    - Map depth -> num_hidden_layers, num_heads -> num_attention_heads,
      in_chans -> num_channels, spatial_merge_size -> merge_size,
      spatial_patch_size/patch_size -> patch_size.
    - Carry through unrecognized keys to kwargs.
    """
    # Prefer embed_dim as the model's hidden size for the ViT
    hidden_size = vision_cfg.get("embed_dim", vision_cfg.get("hidden_size"))
    if hidden_size is None:
        raise ValueError("Neither 'embed_dim' nor 'hidden_size' found in vision_config.")

    mlp_ratio = vision_cfg.get("mlp_ratio", 4)
    intermediate_size = int(round(hidden_size * mlp_ratio))

    # Patch size: HF sometimes has both 'patch_size' and 'spatial_patch_size'
    patch_size = vision_cfg.get("spatial_patch_size", vision_cfg.get("patch_size", 14))

    # Merge size for spatial pooling
    merge_size = vision_cfg.get("spatial_merge_size", 2)

    cfg = QWen2VLVisionConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=vision_cfg.get("depth", 24),
        num_attention_heads=vision_cfg.get("num_heads", vision_cfg.get("num_attention_heads", 16)),
        patch_size=patch_size,
        merge_size=merge_size,
        image_size=448,  # not provided in HF vision_config; keep our default
        num_channels=vision_cfg.get("in_chans", 3),
        layer_norm_eps=1e-6,
        max_patches=1024,
        hidden_act="gelu",
        dtype="float32",
        # keep extra fields (e.g., temporal_patch_size, embed_dim duplicate, etc.)
        kwargs={
            k: v for k, v in vision_cfg.items()
            if k not in {
                "depth", "embed_dim", "mlp_ratio", "num_heads", "in_chans",
                "hidden_size", "patch_size", "spatial_merge_size",
                "spatial_patch_size", "temporal_patch_size"
            }
            # but still keep useful extras explicitly
        } | {
            "temporal_patch_size": vision_cfg.get("temporal_patch_size"),
            "orig_embed_dim": vision_cfg.get("embed_dim"),
            "orig_hidden_size": vision_cfg.get("hidden_size"),
        },
        num_rope_scales=4,
        rope_theta=10000.0,
    )
    return cfg

@dataclasses.dataclass
class QWen2VLConfig(QWen2Config):
    """Configuration for the complete Qwen2 VL model."""

    vision_config: Optional[QWen2VLVisionConfig] = None
    image_size: int = 448
    min_image_size: int = 224
    max_image_size: int = 448
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_patch_size: int = 14
    max_patch_size: int = 28
    min_pixels: int = 56*56
    max_pixels: int = 28*28*1280
    dtype: str = "float32"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # First run parent class post init
        super().__post_init__()

        # Set up vision config if not provided
        if self.vision_config is None:
            self.vision_config = QWen2VLVisionConfig(
                hidden_size=1024,  # Vision hidden size
                intermediate_size=4096,  # Vision MLP size
                num_hidden_layers=24,  # Number of vision transformer layers
                num_attention_heads=16,  # Number of vision attention heads
                patch_size=self.patch_size,
                merge_size=self.merge_size,
                image_size=self.image_size,
                layer_norm_eps=1e-6,
                dtype=self.dtype,
            )
        else:
            self.vision_config = qwen2vl_vision_from_official(self.vision_config)

        # Validate configuration
        if self.patch_size < self.min_patch_size or self.patch_size > self.max_patch_size:
            raise ValueError(
                f"patch_size must be between {self.min_patch_size} and {self.max_patch_size}, "
                f"got {self.patch_size}"
            )
        
        if self.image_size < self.min_image_size or self.image_size > self.max_image_size:
            raise ValueError(
                f"image_size must be between {self.min_image_size} and {self.max_image_size}, "
                f"got {self.image_size}"
            )

        # Calculate maximum patches based on image size and patch size
        max_h = self.max_image_size // (self.patch_size * self.merge_size)
        max_w = self.max_image_size // (self.patch_size * self.merge_size)
        self.vision_config.max_patches = max_h * max_w

        # Add any additional kwargs
        for k, v in self.kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v) 


'''
python -m mlc_llm gen_config ../mlc-models/Qwen2-VL-7B-Instruct/ --conv-template qwen2-vl -o ../mlc-models/mlc-qwen --quantization q0f16
'''