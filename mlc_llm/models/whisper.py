import dataclasses
from typing import Tuple, Optional

import tvm
from tvm import te, tir
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn import (
    Embedding,
    KVCache,
    Linear,
    LayerNorm,
    Conv1D,
    Module,
    ModuleList,
    Parameter,
    Tensor,
    tensor_expr_op,
    permute_dims,
    reshape,
    squeeze,
    matmul,
    maximum,
    minimum,
    softmax,
    gelu,
    print_,
)


@dataclasses.dataclass
class WhisperConfig:
    def __init__(
        self,
        dtype="float32",
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=6,
        encoder_attention_heads=4,
        decoder_layers=6,
        decoder_attention_heads=4,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        decoder_start_token_id=50257,
        d_model=256,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50257,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
        forced_decoder_ids=None,
        **kwargs,
    ):
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_hidden_layers = encoder_layers
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.forced_decoder_ids = forced_decoder_ids
        self.kwargs = kwargs


class WhisperPositionalEmbedding(Module):
    def __init__(self, max_seq_len: int, embed_dim: int, dtype: str):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.weight = Parameter((max_seq_len, embed_dim), dtype=dtype)

    def forward(self, x: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, embed: te.Tensor, offset: tir.Var):
            def compute(i: tir.Var, j: tir.Var, k: tir.Var):
                return embed[offset + j, k]

            return tvm.te.compute([*x.shape, embed.shape[-1]], compute, name="position_embedding")

        pos_embed = tensor_expr_op(te_op, "position_embedding", args=[x, self.weight, offset])
        return pos_embed


class WhisperAttention(Module):
    def __init__(
        self, embed_dim: int, num_heads: int, kv_cache_len: int, dtype: str, bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = Linear(self.embed_dim, self.embed_dim, dtype=dtype, bias=False)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, dtype=dtype, bias=bias)
        self.q_proj = Linear(self.embed_dim, self.embed_dim, dtype=dtype, bias=bias)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, dtype=dtype, bias=bias)
        if kv_cache_len > 0:
            self.k_cache = KVCache(kv_cache_len, [self.num_heads, self.head_dim])
            self.v_cache = KVCache(kv_cache_len, [self.num_heads, self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        cached_cross_attn_states: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        total_seq_len: Optional[tir.Var] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        is_cross_attention = key_value_states is not None or cached_cross_attn_states is not None

        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        q = reshape(self.q_proj(hidden_states) * self.scaling, (bsz, q_len, h, d))

        if is_cross_attention:
            # cross attention
            if cached_cross_attn_states is None:
                # no cache, cross attentions
                kv_len = key_value_states.shape[1]
                k = reshape(self.k_proj(key_value_states), (bsz, kv_len, h, d))
                v = reshape(self.v_proj(key_value_states), (bsz, kv_len, h, d))
                cached_kv = (k, v)
            else:
                # reuse cached k,v, cross_attentions
                k, v = cached_cross_attn_states
        else:
            # self attention
            k = reshape(self.k_proj(hidden_states), (bsz, q_len, h, d))
            v = reshape(self.v_proj(hidden_states), (bsz, q_len, h, d))

            if total_seq_len is not None:
                # reuse cached k, v, self_attention
                self.k_cache.append(squeeze(k, axis=0))
                self.v_cache.append(squeeze(v, axis=0))
                k = reshape(self.k_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
                v = reshape(self.v_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
            else:
                # encode self attention, no cache
                ...

        q = permute_dims(q, [0, 2, 1, 3])  # [b, h, q_len, d]
        k = permute_dims(k, [0, 2, 1, 3])  # [b, h, q_len, d]
        v = permute_dims(v, [0, 2, 1, 3])  # [b, h, q_len, d]

        attn_weights = matmul(q, (permute_dims(k, [0, 1, 3, 2])))  # [b, h, q_len, q_len]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype))
        attn_weights = attn_weights.minimum(tir.max_value(dtype))
        if dtype == "float32":
            attn_weights = softmax(attn_weights, axis=-1)
        else:
            attn_weights = softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        attn_output = matmul(attn_weights, v)  # [b, h, q_len, d]

        attn_output = permute_dims(attn_output, [0, 2, 1, 3])  # [b, q_len, h, d]
        attn_output = reshape(attn_output, (bsz, q_len, self.embed_dim))  # [b, q_len, h * d]

        attn_output = self.out_proj(attn_output)

        if is_cross_attention and cached_cross_attn_states is None:
            return attn_output, cached_kv
        else:
            return attn_output, None


class EncoderLayer(Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            kv_cache_len=0,  # no need for kv_cache
            dtype=config.dtype,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, dtype=config.dtype)
        self.fc1 = Linear(self.embed_dim, config.encoder_ffn_dim, dtype=config.dtype)
        self.fc2 = Linear(config.encoder_ffn_dim, self.embed_dim, dtype=config.dtype)
        self.final_layer_norm = LayerNorm(self.embed_dim, dtype=config.dtype)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
            total_seq_len=None,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.maximum(tir.min_value(hidden_states.dtype))
        hidden_states = hidden_states.minimum(tir.max_value(hidden_states.dtype))

        return hidden_states


class DecoderLayer(Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            kv_cache_len=100,  # TODO
            dtype=config.dtype,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, dtype=config.dtype)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            kv_cache_len=100,  # TODO
            dtype=config.dtype,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, dtype=config.dtype)
        self.fc1 = Linear(self.embed_dim, config.decoder_ffn_dim, dtype=config.dtype)
        self.fc2 = Linear(config.decoder_ffn_dim, self.embed_dim, dtype=config.dtype)
        self.final_layer_norm = LayerNorm(self.embed_dim, dtype=config.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cached_encoder_hidden_states: Tensor,
        total_seq_len: tir.Var,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=encoder_hidden_states,
            cached_cross_attn_states=cached_encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if cached_encoder_hidden_states is None:
            return hidden_states, cross_attn_key_value
        else:
            return hidden_states, None


class WhisperEncoder(Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = 1.0

        self.conv1 = Conv1D(
            self.num_mel_bins, embed_dim, kernel_size=3, padding=1, dtype=config.dtype
        )
        self.conv2 = Conv1D(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, dtype=config.dtype
        )

        self.embed_positions = Embedding(self.max_source_positions, embed_dim, dtype=config.dtype)

        self.layers = ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model, dtype=config.dtype)

    def forward(self, input_features: Tensor) -> Tensor:
        inputs_embeds = gelu(self.conv1(input_features))
        inputs_embeds = gelu(self.conv2(inputs_embeds))

        inputs_embeds = permute_dims(inputs_embeds, [0, 2, 1])
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class WhisperDecoder(Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = 1.0
        self.embed_tokens = Embedding(config.vocab_size, config.d_model, dtype=config.dtype)
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model, dtype=config.dtype
        )

        self.layers = ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = LayerNorm(config.d_model, dtype=config.dtype)

    def forward(
        self,
        input_ids: Tensor,
        total_seq_len: Optional[tir.Var] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        cached_encoder_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        # total_seq_len = Length of generated tokens
        input_embeds = self.embed_tokens(input_ids)
        past_seq_len = total_seq_len - 1
        position_embeds = self.embed_positions(input_ids, offset=past_seq_len)

        hidden_states = input_embeds + position_embeds

        all_encoder_key_value = ()
        for idx, decoder_layer in enumerate(self.layers):
            ith_cached_encoder_key_value = (
                cached_encoder_key_value[idx] if cached_encoder_key_value is not None else None
            )
            hidden_states, encoder_key_value = decoder_layer(
                hidden_states=hidden_states,
                total_seq_len=total_seq_len,
                encoder_hidden_states=encoder_hidden_states,
                cached_encoder_hidden_states=ith_cached_encoder_key_value,
                attention_mask=attention_mask,
            )
            if cached_encoder_key_value is None:
                all_encoder_key_value += (encoder_key_value,)

        hidden_states = self.layer_norm(hidden_states)

        if cached_encoder_key_value is None:
            return hidden_states, all_encoder_key_value
        else:
            return hidden_states, None


class WhisperModel(Module):
    def __init__(self, config: WhisperConfig):
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(Module):
    def __init__(self, config: WhisperConfig):
        self.model = WhisperModel(config)
        self.proj_out = Linear(config.d_model, config.vocab_size, bias=False, dtype=config.dtype)

    def encode(self, input_features: Tensor) -> Tensor:
        return self.model.encoder(input_features)

    def decode(
        self, input_ids: Tensor, total_seq_len: int, encoder_hidden_states: Tensor
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor]]]:
        hidden_states, all_encoder_key_value = self.model.decoder.forward(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=encoder_hidden_states,
            cached_encoder_key_value=None,
            attention_mask=None,
        )
        lm_logits = self.proj_out(hidden_states)
        return lm_logits, all_encoder_key_value

    def prefill(
        self, input_ids: Tensor, total_seq_len: int, cached_encoder_key_value: Tuple[Tuple[Tensor]]
    ) -> Tensor:
        hidden_states, _ = self.model.decoder.forward(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=None,
            cached_encoder_key_value=cached_encoder_key_value,
            attention_mask=None,
        )
        lm_logits = self.proj_out(hidden_states)
        return lm_logits
