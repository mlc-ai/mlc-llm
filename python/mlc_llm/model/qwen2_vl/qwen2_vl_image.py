# qwen2_vl_image.py
# Contains image preprocessing, ViT definition, and other image-related operations.

from typing import List, Optional, Tuple

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor, op

from mlc_llm.model.vision import ImageProcessor
from mlc_llm.support.config import ConfigBase
from .qwen2_vl_config import QWen2VLConfig, QWen2VLVisionConfig
from mlc_llm.nn import RopeMode
from .rope import apply_rotary_emb, precompute_rope_cache

# Constants from CLIP
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class QWen2VLImagePreprocessor(nn.Module):
    """Image preprocessing for QWen2 VL, including smart resize and normalization."""
    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = True,
        image_mean: List[float] = OPENAI_CLIP_MEAN,
        image_std: List[float] = OPENAI_CLIP_STD,
        min_pixels: int = 56*56,
        max_pixels: int = 28*28*1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ):
        super().__init__()
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_processor = ImageProcessor()

    def smart_resize(self, height: int, width: int, factor: int = 28) -> Tuple[int, int]:
        """
        Rescales the image dimensions to meet the following conditions:
        1. Both dimensions are divisible by factor (patch_size * merge_size)
        2. Total pixels within [min_pixels, max_pixels]
        3. Maintains aspect ratio as much as possible
        """
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )

        # Round to nearest multiple of factor
        h_bar = tir.round(height / factor) * factor
        w_bar = tir.round(width / factor) * factor

        # Scale if outside pixel bounds
        if h_bar * w_bar > self.max_pixels:
            beta = tir.sqrt((height * width) / self.max_pixels)
            h_bar = tir.floor(height / beta / factor) * factor
            w_bar = tir.floor(width / beta / factor) * factor
        elif h_bar * w_bar < self.min_pixels:
            beta = tir.sqrt(self.min_pixels / (height * width))
            h_bar = tir.ceil(height * beta / factor) * factor
            w_bar = tir.ceil(width * beta / factor) * factor

        return h_bar, w_bar

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Process images through resize, rescale and normalize steps."""
        # Convert NHWC to NCHW
        pixel_values = op.permute_dims(pixel_values, axes=(0, 3, 1, 2))

        if self.do_resize:
            factor = self.patch_size * self.merge_size
            h_bar, w_bar = self.smart_resize(
                pixel_values.shape[2], 
                pixel_values.shape[3],
                factor=factor
            )
            pixel_values = self.image_processor.resize(
                pixel_values, 
                params={"height": h_bar, "width": w_bar}
            )

        if self.do_rescale:
            pixel_values = self.image_processor.rescale(
                pixel_values,
                rescale_factor=self.rescale_factor
            )

        if self.do_normalize:
            pixel_values = self.image_processor.normalize(pixel_values)

        return pixel_values

class QWen2VLVisionEmbeddings(nn.Module):
    """Patch and position embeddings with 2D patch merging for vision input."""
    def __init__(self, config: QWen2VLVisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.merge_size = config.merge_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True
        )

        # Position embedding will be added after patch merging
        self.pos_embed = nn.Parameter((config.max_patches, config.hidden_size))

    def merge_patches(self, patches: Tensor) -> Tensor:
        """Merge 2x2 neighboring patches."""
        B, H, W, C = patches.shape
        
        # Reshape to group 2x2 patches: B, H/2, 2, W/2, 2, C
        patches = op.reshape(patches, (B, H//2, 2, W//2, 2, C))
        
        # Permute to B, H/2, W/2, 2, 2, C
        patches = op.permute_dims(patches, (0, 1, 3, 2, 4, 5))
        
        # Merge the 2x2 patches: B, H/2, W/2, 4*C
        patches = op.reshape(patches, (B, H//2, W//2, 4*C))
        
        return patches

    def forward(self, pixel_values: Tensor) -> Tensor:
        # Get patches: B, C, H, W -> B, hidden_size, H//patch_size, W//patch_size
        patches = self.patch_embed(pixel_values)
        
        B, C, H, W = patches.shape
        
        # Reshape to B, H, W, C for patch merging
        patches = op.permute_dims(patches, (0, 2, 3, 1))
        
        # Merge 2x2 patches
        patches = self.merge_patches(patches)
        
        # Reshape to sequence: B, (H/2)*(W/2), 4*hidden_size
        patches = op.reshape(patches, (B, -1, 4*C))
        
        # Add position embeddings
        seq_length = patches.shape[1]
        position_embeddings = self.pos_embed[:seq_length]
        patches = patches + position_embeddings
        
        return patches

class QWen2VLVisionTransformer(nn.Module):
    """Vision transformer with patch merging for QWen2 VL."""
    def __init__(self, config: QWen2VLConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = QWen2VLVisionEmbeddings(config.vision_config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            QWen2VLVisionLayer(config.vision_config) for _ in range(config.vision_config.num_hidden_layers)
        ])
        
        # Final layernorm
        self.post_layernorm = nn.LayerNorm(
            config.vision_config.hidden_size * 4,  # *4 because of patch merging
            eps=config.vision_config.layer_norm_eps
        )
        
    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Final layernorm
        hidden_states = self.post_layernorm(hidden_states)
        
        return hidden_states

class QWen2VLVisionLayer(nn.Module):
    """Single transformer layer for vision processing."""
    def __init__(self, config: QWen2VLVisionConfig):
        super().__init__()
        hidden_size = config.hidden_size * 4  # *4 because of patch merging
        self.attention = QWen2VLVisionAttention(config, hidden_size)
        self.mlp = QWen2VLVisionMLP(config, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class QWen2VLVisionAttention(nn.Module):
    """Multi-head attention with M-ROPE for vision transformer."""
    def __init__(self, config: QWen2VLVisionConfig, hidden_size: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # M-ROPE parameters
        self.rope_mode = RopeMode.NORMAL  # Using normal ROPE mode but with multiple scales
        self.num_rope_scales = 4  # Number of frequency bands for M-ROPE
        self.rope_scale = 1.0
        self.rope_theta = 10000
        self.max_position_embeddings = config.max_patches
        
        # Initialize rope cache with multiple scales
        self.rope_cache = {}
        for scale_idx in range(self.num_rope_scales):
            scale = 1.0 / (2 ** scale_idx)  # Geometric progression of scales
            self.rope_cache[f"scale_{scale_idx}"] = precompute_rope_cache(
                dim=self.head_dim,
                num_heads=self.num_attention_heads,
                max_seq_len=self.max_position_embeddings,
                rope_mode=self.rope_mode,
                rope_scale=scale * self.rope_scale,
                rope_theta=self.rope_theta,
            )
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        B, L, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = op.reshape(q, (B, L, self.num_attention_heads, self.head_dim))
        k = op.reshape(k, (B, L, self.num_attention_heads, self.head_dim))
        v = op.reshape(v, (B, L, self.num_attention_heads, self.head_dim))

        # Apply M-ROPE: split heads into groups and apply different scales
        heads_per_scale = self.num_attention_heads // self.num_rope_scales
        q_scaled = []
        k_scaled = []
        
        for scale_idx in range(self.num_rope_scales):
            start_idx = scale_idx * heads_per_scale
            end_idx = start_idx + heads_per_scale
            
            # Get current scale's rope cache
            rope_cache = self.rope_cache[f"scale_{scale_idx}"]
            
            # Apply rotary embeddings with current scale
            q_part = q[:, :, start_idx:end_idx, :]
            k_part = k[:, :, start_idx:end_idx, :]
            
            q_part_scaled = apply_rotary_emb(
                q_part,
                rope_cache,
                offset=0,
                num_heads=heads_per_scale,
            )
            k_part_scaled = apply_rotary_emb(
                k_part,
                rope_cache,
                offset=0,
                num_heads=heads_per_scale,
            )
            
            q_scaled.append(q_part_scaled)
            k_scaled.append(k_part_scaled)
        
        # Concatenate all scaled versions
        q = op.concatenate(q_scaled, axis=2)
        k = op.concatenate(k_scaled, axis=2)
        
        # Compute attention with scaled Q, K, V
        attn_output = op_ext.attention(q, k, v)
        
        # Project output
        output = self.o_proj(attn_output)
        return output

class QWen2VLVisionMLP(nn.Module):
    """MLP layer for vision transformer."""
    def __init__(self, config: QWen2VLVisionConfig, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, hidden_size)
        self.act = nn.GELU()
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

