"""Qwen3.5 Vision-Language model wrapper (hybrid DeltaNet + full attention)."""

import dataclasses
from typing import Any, Dict, Optional  # noqa: UP035

from tvm import relax, target, te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Object, Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.vision import ImageProcessor
from mlc_llm.nn.kv_cache import PagedKVCache
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.protocol.artifact_manifest import ArtifactDefinition
from mlc_llm.support.config import ConfigBase

from .qwen35_model import Qwen35Config, Qwen35LMHeadModel
from .qwen35_vision import Qwen35VisionConfig, Qwen35VisionModel

# pylint: disable=invalid-name,missing-docstring,too-many-instance-attributes


@dataclasses.dataclass
class Qwen35VConfig(ConfigBase):
    """Configuration for Qwen3.5 Vision-Language model."""

    text_config: Qwen35Config = None
    vision_config: Qwen35VisionConfig = None
    image_token_id: int = 248056
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    image_size: int = 448
    vocab_size: int = -1
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    prefill_chunk_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        # Parse text_config
        if self.text_config is None:
            raise ValueError("Qwen35VConfig requires text_config")

        if isinstance(self.text_config, Qwen35Config):
            text_dict = dataclasses.asdict(self.text_config)
        else:
            text_dict = dict(self.text_config)
        # Flatten nested kwargs to avoid double-kwargs in from_dict round-trips
        for k, v in text_dict.pop("kwargs", {}).items():
            text_dict[k] = v
        text_dict["tensor_parallel_shards"] = self.tensor_parallel_shards
        # Qwen3.5-9B sets tie_word_embeddings at the top level only.
        if "tie_word_embeddings" in self.kwargs:
            text_dict.setdefault("tie_word_embeddings", self.kwargs.pop("tie_word_embeddings"))
        self.text_config = Qwen35Config.from_dict(text_dict)

        # Parse vision_config
        if isinstance(self.vision_config, Qwen35VisionConfig):
            vision_dict = dataclasses.asdict(self.vision_config)
        elif self.vision_config is not None:
            vision_dict = dict(self.vision_config)
        else:
            raise ValueError("Qwen35VConfig requires vision_config")
        for k, v in vision_dict.pop("kwargs", {}).items():
            vision_dict[k] = v
        self.vision_config = Qwen35VisionConfig.from_dict(vision_dict)

        for k in ["vocab_size", "context_window_size", "prefill_chunk_size"]:
            if getattr(self, k) <= 0:
                setattr(self, k, getattr(self.text_config, k))

    @property
    def grid(self) -> int:
        return self.image_size // self.vision_config.patch_size

    @property
    def tokens_per_image(self) -> int:
        return (self.grid // self.vision_config.spatial_merge_size) ** 2


class Qwen35VForCausalLM(nn.Module):
    # Image tokens take ordinary 1-D positions where the reference uses M-RoPE. Only every
    # fourth layer applies rotary embeddings, to a quarter of the head dims, and the next
    # token distribution after an image prompt matches the reference to a KL of 0.006.
    def __init__(self, config: Qwen35VConfig):
        self.config = config
        self.language_model = Qwen35LMHeadModel(config.text_config)
        self.visual = Qwen35VisionModel(config.vision_config, config.image_size)
        self.image_processor = ImageProcessor()

        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.text_config.vocab_size
        self.num_hidden_layers = config.text_config.num_hidden_layers
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.dtype = "float32"
        self.image_dtype = (
            "uint32"
            if target.Target.current() and target.Target.current().kind.name == "webgpu"
            else "uint8"
        )

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    # pylint: disable=protected-access
    def image_preprocess(self, pixel_values: Tensor) -> Tensor:
        pixel_values = op.permute_dims(pixel_values, axes=[0, 3, 1, 2])  # NHWC -> NCHW
        image_size = self.config.image_size
        pixel_values = self.image_processor.resize(
            pixel_values, params={"height": image_size, "width": image_size}
        )
        pixel_values = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                pixel_values._expr,
                relax.TensorType([1, 3, image_size, image_size], pixel_values.dtype),
            ),
            "resized_image",
        )
        pixel_values = self.image_processor.rescale(pixel_values)
        # The checkpoints normalize with mean and std 0.5, which normalize_siglip applies.
        return self.image_processor.normalize_siglip(pixel_values)

    def image_embed(  # pylint: disable=too-many-arguments,unused-argument
        self,
        pixel_values: Tensor,
        resized_height,
        resized_width,
        crop_height,
        crop_width,
    ) -> Tensor:
        return self.embed_image(pixel_values)

    def embed_image(self, pixel_values: Tensor) -> Tensor:
        pixel_values = self.image_preprocess(pixel_values).astype(self.dtype)
        vision_outputs = self.visual(pixel_values)
        return op.reshape(vision_outputs, (self.config.tokens_per_image, self.hidden_size))

    def embed(self, input_ids: Tensor):
        return self.language_model.embed(input_ids)

    def get_logits(self, hidden_states: Tensor):
        language_model = self.language_model
        if language_model.tie_word_embeddings:
            logits = language_model.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = language_model.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, rnn_state: RNNState):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:, -1, :]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states, rnn_state = self.language_model.model(input_embed, paged_kv_cache, rnn_state)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        return self.get_logits(hidden_states), paged_kv_cache, rnn_state

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, rnn_state: RNNState):
        op_ext.configure()
        hidden_states, rnn_state = self.language_model.model(input_embed, paged_kv_cache, rnn_state)
        return self.get_logits(hidden_states), paged_kv_cache, rnn_state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_prefill(
            input_embeds, logit_positions, paged_kv_cache, rnn_state
        )

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_decode(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.language_model.batch_verify(input_embeds, paged_kv_cache, rnn_state)

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> Object:
        return self.language_model.create_paged_kv_cache(
            max_batch_size,
            max_total_seq_len,
            prefill_chunk_size,
            page_size,
            support_sliding_window,
        )

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
    ) -> Object:
        return self.language_model.create_rnn_state(max_batch_size, max_history)

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor(
                    [1, "image_height", "image_width", 3], self.image_dtype
                ),
                "resized_height": nn.spec.Int(),
                "resized_width": nn.spec.Int(),
                "crop_height": nn.spec.Int(),
                "crop_width": nn.spec.Int(),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "embed_image": {
                "pixel_values": nn.spec.Tensor(
                    [1, "image_height", "image_width", 3], self.image_dtype
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
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
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


def qwen35v_artifact_tasks(config: Qwen35VConfig):
    image_size = config.image_size
    return {
        "chat.completions": {
            "executor": "generation",
            "inputs": {
                "text": {"processor": "tokenizer"},
                "image": {
                    "processor": {
                        "kind": "image_decode",
                        "format": "rgb_u8",
                        "layout": "nhwc",
                        "resize": {"mode": "stretch", "height": image_size, "width": image_size},
                        "num_embeddings": config.tokens_per_image,
                    },
                    "adapter": "image",
                    "prompt": {
                        "prefix_token_ids": [config.vision_start_token_id],
                        "placeholder_token_id": config.image_token_id,
                        "suffix_token_ids": [config.vision_end_token_id],
                    },
                },
            },
            "output": "text",
        }
    }


def qwen35v_artifact_programs(_config: Qwen35VConfig):
    # The recurrent layers keep their state next to the KV cache, so prefill and decode take
    # the RNN state as a third argument and the program declares create_rnn_state.
    program = {
        "kind": "token_generation",
        "exports": {
            "embed_tokens": "embed",
            "prefill_embeds": "prefill",
            "decode_embeds": "decode",
            "create_kv_cache": "create_tir_paged_kv_cache",
            "create_rnn_state": "create_rnn_state",
        },
        "adapters": {"image": "embed_image"},
    }
    current = target.Target.current()
    if current and current.kind.name == "webgpu":
        program["adapter_dtypes"] = {"image": "uint32"}
    return {"generation": program}


QWEN35V_ARTIFACT = ArtifactDefinition(
    tasks=qwen35v_artifact_tasks, programs=qwen35v_artifact_programs
)
