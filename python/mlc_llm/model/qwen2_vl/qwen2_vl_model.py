"""Implementation of the Qwen2 VL model."""

from typing import Optional, Tuple

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.qwen2.qwen2_model import QWen2Model
from mlc_llm.nn import PagedKVCache

from .qwen2_vl_config import QWen2VLConfig
from .qwen2_vl_image import QWen2VLImagePreprocessor, QWen2VLVisionTransformer

class QWen2VLProjection(nn.Module):
    """Projects vision features to language model dimension."""
    def __init__(self, config: QWen2VLConfig):
        super().__init__()
        # Input is 4x vision hidden size due to patch merging
        vision_hidden_size = config.vision_config.hidden_size * 4
        
        # Project to language model dimension with two-layer MLP
        self.linear_1 = nn.Linear(vision_hidden_size, config.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class QWen2VLForCausalLM(nn.Module):
    """Qwen2 VL model combining vision and language capabilities."""
    
    def __init__(self, config: QWen2VLConfig):
        super().__init__()
        self.config = config
        
        # Vision components
        self.image_processor = QWen2VLImagePreprocessor(
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            patch_size=config.patch_size,
            merge_size=config.merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        self.vision_model = QWen2VLVisionTransformer(config)
        self.vision_projection = QWen2VLProjection(config)
        
        # Language model
        self.language_model = QWen2Model(config)
        
        # Final LM head (reuse embedding weight if tied)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
        # Model attributes needed for integration
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def image_preprocess(self, pixel_values: Tensor) -> Tensor:
        """Preprocess images for the vision encoder."""
        return self.image_processor(pixel_values)

    def image_embed(self, pixel_values: Tensor) -> Tensor:
        """Get image embeddings from preprocessed images."""
        # Process through vision transformer
        vision_outputs = self.vision_model(pixel_values)
        
        # Project to language model dimension
        image_embeds = self.vision_projection(vision_outputs)
        
        return image_embeds

    def embed(self, input_ids: Tensor):
        """Get text embeddings."""
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.language_model.embed_tokens(input_ids)

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        """Forward pass for both vision and language."""
        op_ext.configure()

        hidden_states = self.language_model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)

        if self.tie_word_embeddings:
            logits = self.language_model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        """Prefill KV cache."""
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.language_model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        
        if self.tie_word_embeddings:
            logits = self.language_model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        """Decode step."""
        op_ext.configure()

        hidden_states = self.language_model(input_embed, paged_kv_cache)
        if self.tie_word_embeddings:
            logits = self.language_model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        """Batch prefill operation."""
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        """Batch decode operation."""
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        """Create paged KV cache."""
        return self.language_model.create_paged_kv_cache(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
        )

    def get_default_spec(self):
        """Get the default module spec."""
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_preprocess": {
                "pixel_values": nn.spec.Tensor([1, "image_height", "image_width", 3], "uint8"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor([1, 3, "image_height", "image_width"], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

