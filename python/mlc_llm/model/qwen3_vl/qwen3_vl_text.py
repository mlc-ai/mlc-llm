from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .qwen3_vl_config import Qwen3VLTextConfig

from mlc_llm.model.qwen3.qwen3_model import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model

from typing import Optional

class LlamaRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    # fyi config is of type LlamaConfig
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]

        assert self.rope_type == "default", f"Unsupported rope type {self.rope_type}"
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, device)

        self.original_inv_freq = inv_freq

    # fyi config is of type LlamaConfig
    @staticmethod
    def compute_default_rope_parameters(
        config,
        device: Optional[str] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (op.arange(0, dim, 2, dtype="int64").to(device=device, dtype="float32") / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        # TODO: translate from pytorch to tvm
        raise NotImplementedError
        
        # inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # position_ids_expanded = position_ids[:, None, :].float()

        # device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos() * self.attention_scaling
        #     sin = emb.sin() * self.attention_scaling

        # return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextRotaryEmbedding(LlamaRotaryEmbedding):
    inv_freq: Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__(config, device=device)

        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
            
        """

        # TODO: translate from pytorch to tvm
        raise NotImplementedError

        # freqs_t = freqs[0]  # just overwrite the first dimension T
        # for dim, offset in enumerate((1, 2), start=1):  # H, W
        #     length = mrope_section[dim] * 3
        #     idx = slice(offset, length, 3)
        #     freqs_t[..., idx] = freqs[dim, ..., idx]
        # return freqs_t

    def forward(self, x, position_ids):

        # TODO: translate from pytorch to tvm
        raise NotImplementedError
        
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)


        # if position_ids.ndim == 2:
        #     position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        # position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        # device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        #     freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos() * self.attention_scaling
        #     sin = emb.sin() * self.attention_scaling

        # return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextAttention(Qwen3Attention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):

        # TODO - pytorch qwen3attention takes a layer_idx but mlc doesnt, check if we really need it
        # hazardous - qwen3attention expects qwen3config, we are passing qwen3vltextconfig
        super().__init__(config)

        # no sliding window in mlc qwen3attention?
        # del self.sliding_window

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        # TODO: translate from pytorch to tvm
        raise NotImplementedError

        # input_shape = hidden_states.shape[:-1]
        # hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_values is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     **kwargs,
        # )

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # attn_output = self.o_proj(attn_output)
        # return attn_output, attn_weights


class Qwen3VLTextDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        # TODO - pytorch qwen3attention takes a layer_idx but mlc doesnt, check if we really need it
        # hazardous - qwen3attention expects qwen3config, we are passing qwen3vltextconfig
        super().__init__(config)

        # no attention_type in mlc qwen3decoderlayer?
        #del self.attention_type

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # TODO: translate from pytorch to tvm

        # do we even need this class if the forward is just a call to super().forward?
        raise NotImplementedError

        # return super().forward(
        #     hidden_states=hidden_states,
        #     position_embeddings=position_embeddings,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     cache_position=cache_position,
        #     **kwargs,
        # )


class Qwen3VLTextModel(Qwen3Model):
    config: Qwen3VLTextConfig

    def __init__(self, config: Qwen3VLTextConfig):

        # hazardous - qwen3model expects qwen3config, we are passing qwen3vltextconfig
        super().__init__(config)

        # no has_sliding_layers in mlc qwen3model?

        #del self.has_sliding_layers

    def _deepstack_process(
        self, hidden_states: Tensor, visual_pos_masks: Tensor, visual_embeds: Tensor
    ):
        raise NotImplementedError
        
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(self):
        raise NotImplementedError

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     # args for deepstack
    #     visual_pos_masks: Optional[torch.Tensor] = None,
    #     deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )