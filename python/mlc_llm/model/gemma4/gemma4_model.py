"""Text+audio implementation of the dense Gemma 4 E2B architecture."""

from __future__ import annotations

import math
from typing import Dict, Tuple  # noqa: UP035

from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.gemma.gemma_model import GemmaEmbedding
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.protocol.artifact_manifest import ArtifactDefinition

from .gemma4_audio import (
    Gemma4AudioFeatureExtractor,
    Gemma4AudioModel,
    Gemma4MultimodalEmbedder,
    Gemma4RMSNorm,
)
from .gemma4_config import Gemma4Config, Gemma4TextConfig

_PHYSICAL_HEAD_DIM = 512


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        is_shared = layer_idx >= config.first_kv_shared_layer
        intermediate_size = config.intermediate_size
        if config.use_double_wide_mlp and is_shared:
            intermediate_size *= 2
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = op.split(self.gate_up_proj(hidden_states), 2, axis=-1)
        return self.down_proj(op.gelu(gate, approximate="tanh") * up)


class Gemma4TextRotaryEmbedding(nn.Module):
    """Default local RoPE and Gemma 4's proportional global RoPE."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        self.is_global = config.layer_types[layer_idx] == "full_attention"
        self.head_dim = config.head_dim_for_layer(layer_idx)
        rope = config.rope_parameters[config.layer_types[layer_idx]]
        self.theta = float(rope["rope_theta"])
        self.active_frequencies = (
            int(self.head_dim * float(rope.get("partial_rotary_factor", 1.0))) // 2
        )

    def _apply(self, values: Tensor, positions: Tensor, name: str) -> Tensor:
        def _rope(values: te.Tensor, position_map: te.Tensor):
            batch, seq_len, _, head_dim = values.shape
            half_dim = head_dim // 2
            dtype = values.dtype

            def _value(b: tirx.Var, s: tirx.Var, h: tirx.Var, d: tirx.Var):
                frequency_index = d % half_dim
                angle = tirx.if_then_else(
                    frequency_index < self.active_frequencies,
                    position_map[b * seq_len + s]
                    / tirx.power(
                        self.theta,
                        (2 * frequency_index) / tirx.const(self.head_dim, "float32"),
                    ),
                    tirx.const(0, "float32"),
                )
                partner = tirx.if_then_else(d < half_dim, d + half_dim, d - half_dim)
                sign = tirx.if_then_else(
                    d < half_dim,
                    tirx.const(-1, dtype),
                    tirx.const(1, dtype),
                )
                value = values[b, s, h, d]
                rotated = values[b, s, h, partner] * sign
                return (value * tirx.cos(angle) + rotated * tirx.sin(angle)).astype(dtype)

            return te.compute(values.shape, _value, name="gemma4_rope")

        return op.tensor_expr_op(_rope, name, [values, positions])

    def forward(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._apply(query, positions, "gemma4_query_rope"),
            self._apply(key, positions, "gemma4_key_rope"),
        )

    def apply_query(self, query: Tensor, positions: Tensor) -> Tensor:
        return self._apply(query, positions, "gemma4_query_rope")


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_shared = layer_idx >= config.first_kv_shared_layer
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        physical_layer_types = config.layer_types[: config.first_kv_shared_layer]
        self.source_layer_id = (
            len(physical_layer_types) - 1 - physical_layer_types[::-1].index(self.layer_type)
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_q_heads * self.head_dim,
            bias=False,
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        if not self.is_shared:
            self.k_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(
                self.head_dim,
                config.rms_norm_eps,
                with_scale=False,
            )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.rotary_emb = Gemma4TextRotaryEmbedding(config, layer_idx)

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        positions: Tensor,
        shared_kv: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape
        query = op.reshape(
            self.q_proj(hidden_states),
            (batch, seq_len, self.num_q_heads, self.head_dim),
        )
        query = self.q_norm(query)

        if self.is_shared:
            if shared_kv is None:
                raise ValueError(f"Missing shared {self.layer_type} K/V source")
            key, value = shared_kv
            query = self.rotary_emb.apply_query(query, positions)
            query = _pad_head_dim(query, self.head_dim)
            output = paged_kv_cache.attention_with_q_from_cache(
                self.source_layer_id,
                query,
                key,
                value,
                sm_scale=1.0,
            )
            output = _slice_head_dim(output, self.head_dim)
            output = op.reshape(
                output,
                (batch, seq_len, self.num_q_heads * self.head_dim),
            )
            return self.o_proj(output), None

        key = op.reshape(
            self.k_proj(hidden_states),
            (batch, seq_len, self.num_kv_heads, self.head_dim),
        )
        value = op.reshape(
            self.v_proj(hidden_states),
            (batch, seq_len, self.num_kv_heads, self.head_dim),
        )
        key = self.k_norm(key)
        value = self.v_norm(value)
        query, key = self.rotary_emb(query, key, positions)
        query = _pad_head_dim(query, self.head_dim)
        key = _pad_head_dim(key, self.head_dim)
        value = _pad_head_dim(value, self.head_dim)
        qkv = op.concat([query, key, value], dim=2)
        output = paged_kv_cache.attention_with_fused_qkv(
            self.layer_idx,
            qkv,
            self.num_q_heads,
            sm_scale=1.0,
        )
        output = _slice_head_dim(output, self.head_dim)
        output = op.reshape(output, (batch, seq_len, self.num_q_heads * self.head_dim))
        return self.o_proj(output), (key, value)


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
        )
        self.post_per_layer_input_norm = Gemma4RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.layer_scalar = nn.Parameter((1,))

    def forward(
        self,
        hidden_states: Tensor,
        per_layer_input: Tensor,
        paged_kv_cache: PagedKVCache,
        positions: Tensor,
        shared_kv: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        residual = hidden_states
        attention, current_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            paged_kv_cache,
            positions,
            shared_kv,
        )
        hidden_states = residual + self.post_attention_layernorm(attention)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = op.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = residual + self.post_per_layer_input_norm(hidden_states)
        return hidden_states * self.layer_scalar, current_kv


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        self.config = config
        self.embed_tokens = GemmaEmbedding(config.vocab_size, config.hidden_size)
        self.embed_tokens_per_layer = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size_per_layer_input, config.hidden_size_per_layer_input)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(
            config.hidden_size_per_layer_input,
            config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        physical_layer_types = config.layer_types[: config.first_kv_shared_layer]
        self.shared_kv_source_layers = {
            len(physical_layer_types) - 1 - physical_layer_types[::-1].index(layer_type)
            for layer_type in set(config.layer_types[config.first_kv_shared_layer :])
        }
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

    def _per_layer_inputs(
        self,
        input_embeds: Tensor,
        token_ids: Tensor | None,
        modality_ids: Tensor | None,
    ) -> list[Tensor]:
        batch, seq_len, _ = input_embeds.shape
        projection_embeds = input_embeds
        if modality_ids is not None:
            pad_embedding = self.embed_tokens(
                op.full([1], self.config.pad_token_id, dtype="int32")
            ) * math.sqrt(self.config.hidden_size)
            projection_embeds = _replace_modality_embeddings(
                input_embeds,
                modality_ids,
                pad_embedding,
            )
        projected = self.per_layer_model_projection(projection_embeds)
        projected = projected * (self.config.hidden_size**-0.5)
        projected = op.reshape(
            projected,
            (
                batch,
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            ),
        )
        projected = self.per_layer_projection_norm(projected)
        projected_layers = [
            op.squeeze(item, axis=2)
            for item in op.split(projected, self.config.num_hidden_layers, axis=2)
        ]
        if token_ids is None:
            return projected_layers

        if modality_ids is not None:
            token_ids = _replace_modality_token_ids(
                token_ids,
                modality_ids,
                self.config.pad_token_id,
            )
        identity_scale = math.sqrt(self.config.hidden_size_per_layer_input)
        combined_scale = 2.0**-0.5
        return [
            (
                projected_layers[index]
                + self.embed_tokens_per_layer[index](token_ids) * identity_scale
            )
            * combined_scale
            for index in range(self.config.num_hidden_layers)
        ]

    def forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        token_ids: Tensor | None = None,
        modality_ids: Tensor | None = None,
    ) -> Tensor:
        positions = paged_kv_cache.get_query_positions(
            input_embeds.shape[0] * input_embeds.shape[1]
        )
        per_layer_inputs = self._per_layer_inputs(input_embeds, token_ids, modality_ids)
        hidden_states = input_embeds
        shared_kv: Dict[str, Tuple[Tensor, Tensor]] = {}  # noqa: UP006
        for layer_idx, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[layer_idx]
            hidden_states, current_kv = layer(
                hidden_states,
                per_layer_inputs[layer_idx],
                paged_kv_cache,
                positions,
                shared_kv.get(layer_type),
            )
            if layer_idx in self.shared_kv_source_layers:
                if current_kv is None:
                    raise ValueError("The shared-KV source layer did not produce K/V states")
                shared_kv[layer_type] = current_kv
        return self.norm(hidden_states)


class Gemma4ForConditionalGeneration(nn.Module):
    """Gemma 4 E2B with text and audio inputs and text generation."""

    def __init__(self, config: Gemma4Config):
        self.config = config
        self.language_model = Gemma4TextModel(config.text_config)
        self.audio_preprocessor = Gemma4AudioFeatureExtractor(config.audio_config)
        self.audio_tower = Gemma4AudioModel(config.audio_config)
        self.embed_audio = Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
        self.dtype = "float32"

    def to(self, dtype: str | None = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.language_model.embed(input_ids)

    def audio_embed(self, samples: Tensor) -> Tensor:
        features = self.audio_preprocessor(samples)
        hidden_states = self.audio_tower(op.astype(features, self.dtype))
        hidden_states = self.embed_audio(hidden_states)
        return op.squeeze(hidden_states, axis=0)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.language_model.embed_tokens.lm_head_forward(hidden_states)
        cap = self.config.text_config.final_logit_softcapping
        if cap is not None:
            logits = op.tanh(logits / cap) * cap
        return logits

    def _forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        token_ids: Tensor | None = None,
        modality_ids: Tensor | None = None,
        logit_positions: Tensor | None = None,
    ) -> Tensor:
        op_ext.configure()
        hidden_states = self.language_model(
            input_embeds,
            paged_kv_cache,
            token_ids=token_ids,
            modality_ids=modality_ids,
        )
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def prefill_prompt(
        self,
        input_embeds: Tensor,
        token_ids: Tensor,
        modality_ids: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        op_ext.configure()
        hidden_states = self.language_model(
            input_embeds,
            paged_kv_cache,
            token_ids=token_ids,
            modality_ids=modality_ids,
        )
        return self.get_logits(index_last_token(hidden_states)), paged_kv_cache

    def decode_tokens(self, token_ids: Tensor, paged_kv_cache: PagedKVCache):
        input_embeds = self.language_model.embed(token_ids)
        logits = self._forward(
            input_embeds,
            paged_kv_cache,
            token_ids=token_ids,
        )
        return logits, paged_kv_cache

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.language_model(input_embed, paged_kv_cache)
        return self.get_logits(index_last_token(hidden_states)), paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        return self._forward(input_embed, paged_kv_cache), paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        return self._forward(
            input_embeds,
            paged_kv_cache,
            logit_positions=logit_positions,
        ), paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self._forward(input_embeds, paged_kv_cache), paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self._forward(input_embeds, paged_kv_cache), paged_kv_cache

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        text = self.config.text_config
        physical_layers = text.first_kv_shared_layer
        return PagedKVCache.create_generic(
            attn_kind=[
                "mha" if text.layer_types[index] == "full_attention" else "mha_sliding"
                for index in range(physical_layers)
            ],
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=physical_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            qk_head_dim=_PHYSICAL_HEAD_DIM,
            v_head_dim=_PHYSICAL_HEAD_DIM,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=10_000,
            dtype=self.dtype,
            layer_sliding_window_size=text.sliding_window,
        )

    def get_default_spec(self):
        hidden_size = self.config.text_config.hidden_size
        cache_arg = nn.spec.Object(object_type=PagedKVCache)
        packed = {"param_mode": "packed", "effect_mode": "none"}
        none = {"param_mode": "none", "effect_mode": "none"}
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": packed,
            },
            "audio_embed": {
                "samples": nn.spec.Tensor(["num_samples"], "float32"),
                "$": packed,
            },
            "prefill_prompt": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", hidden_size], self.dtype),
                "token_ids": nn.spec.Tensor([1, "seq_len"], "int32"),
                "modality_ids": nn.spec.Tensor([1, "seq_len"], "int32"),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "decode_tokens": {
                "token_ids": nn.spec.Tensor(["batch_size", 1], "int32"),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", hidden_size], self.dtype),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, hidden_size], self.dtype),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, hidden_size], self.dtype),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", hidden_size], self.dtype),
                "paged_kv_cache": cache_arg,
                "$": packed,
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": none,
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


def gemma4_artifact_tasks(config: Gemma4Config):
    return {
        "chat.completions": {
            "executor": "generation",
            "inputs": {
                "text": {"processor": "tokenizer"},
                "audio": {
                    "processor": {
                        "kind": "audio_decode",
                        "format": "pcm_f32",
                        "sample_rate_hz": config.audio_config.sampling_rate,
                        "channels": 1,
                        "min_samples": 161,
                        "max_samples": config.audio_config.max_samples,
                    },
                    "adapter": "audio",
                    "prompt": {
                        "prefix_token_ids": [config.boa_token_id],
                        "placeholder_token_id": config.audio_token_id,
                        "suffix_token_ids": [config.eoa_token_index],
                    },
                },
            },
            "output": "text",
        }
    }


def gemma4_artifact_programs(_config: Gemma4Config):
    return {
        "generation": {
            "kind": "token_generation",
            "exports": {
                "embed_tokens": "embed",
                "prefill_prompt": "prefill_prompt",
                "decode_tokens": "decode_tokens",
                "create_kv_cache": "create_tir_paged_kv_cache",
            },
            "adapters": {"audio": "audio_embed"},
        }
    }


GEMMA4_ARTIFACT = ArtifactDefinition(
    tasks=gemma4_artifact_tasks,
    programs=gemma4_artifact_programs,
    required_features=("shader-f16",),
)


def _pad_head_dim(hidden_states: Tensor, head_dim: int) -> Tensor:
    if head_dim == _PHYSICAL_HEAD_DIM:
        return hidden_states
    return op.pad(hidden_states, [0, 0, 0, 0, 0, 0, 0, _PHYSICAL_HEAD_DIM - head_dim])


def _slice_head_dim(hidden_states: Tensor, head_dim: int) -> Tensor:
    if head_dim == _PHYSICAL_HEAD_DIM:
        return hidden_states
    return op.split(hidden_states, [head_dim], axis=-1)[0]


def _replace_modality_token_ids(
    token_ids: Tensor,
    modality_ids: Tensor,
    pad_token_id: int,
) -> Tensor:
    def _replace(ids: te.Tensor, modalities: te.Tensor):
        return te.compute(
            ids.shape,
            lambda *indices: tirx.if_then_else(
                modalities[indices] == 0,
                ids[indices],
                tirx.const(pad_token_id, ids.dtype),
            ),
            name="gemma4_replace_modality_token_ids",
        )

    return op.tensor_expr_op(
        _replace,
        "gemma4_replace_modality_token_ids",
        [token_ids, modality_ids],
    )


def _replace_modality_embeddings(
    input_embeds: Tensor,
    modality_ids: Tensor,
    pad_embedding: Tensor,
) -> Tensor:
    """Use the scaled PAD embedding for the PLE projection at soft-token positions."""

    def _replace(embeds: te.Tensor, modalities: te.Tensor, pad: te.Tensor):
        return te.compute(
            embeds.shape,
            lambda batch, seq, hidden: tirx.if_then_else(
                modalities[batch, seq] == 0,
                embeds[batch, seq, hidden],
                pad[0, hidden],
            ),
            name="gemma4_replace_modality_embeddings",
        )

    return op.tensor_expr_op(
        _replace,
        "gemma4_replace_modality_embeddings",
        [input_embeds, modality_ids, pad_embedding],
    )


__all__ = [
    "GEMMA4_ARTIFACT",
    "Gemma4ForConditionalGeneration",
    "gemma4_artifact_programs",
    "gemma4_artifact_tasks",
]
