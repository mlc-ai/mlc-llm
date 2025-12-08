"""
Minimal model for Qwen3-VL.
"""
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .qwen3_vl_config import Qwen3VLConfig, Qwen3VLVisionConfig
from .qwen_2_5_vl import Qwen2_5_VLModel

from .qwen3_vl_vision import Qwen3VLVisionModel
from .qwen3_vl_text import Qwen3VLTextModel

from typing import Optional, Union
from tvm import tir
from mlc_llm.nn import PagedKVCache, RopeMode


class Qwen3VLModel(Qwen2_5_VLModel):
    config: Qwen3VLConfig
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)

    def get_rope_index(
        self,
        input_ids: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # TODO translate pytorch to tvm
        raise NotImplementedError

        # # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        # if video_grid_thw is not None:
        #     video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        #     video_grid_thw[:, 0] = 1

        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        # image_token_id = self.config.image_token_id
        # video_token_id = self.config.video_token_id
        # vision_start_token_id = self.config.vision_start_token_id
        # mrope_position_deltas = []
        # if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        #     total_input_ids = input_ids
        #     if attention_mask is None:
        #         attention_mask = torch.ones_like(total_input_ids)
        #     position_ids = torch.ones(
        #         3,
        #         input_ids.shape[0],
        #         input_ids.shape[1],
        #         dtype=input_ids.dtype,
        #         device=input_ids.device,
        #     )
        #     image_index, video_index = 0, 0
        #     attention_mask = attention_mask.to(total_input_ids.device)
        #     for i, input_ids in enumerate(total_input_ids):
        #         input_ids = input_ids[attention_mask[i] == 1]
        #         image_nums, video_nums = 0, 0
        #         vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
        #         vision_tokens = input_ids[vision_start_indices + 1]
        #         image_nums = (vision_tokens == image_token_id).sum()
        #         video_nums = (vision_tokens == video_token_id).sum()
        #         input_tokens = input_ids.tolist()
        #         llm_pos_ids_list: list = []
        #         st = 0
        #         remain_images, remain_videos = image_nums, video_nums
        #         for _ in range(image_nums + video_nums):
        #             if image_token_id in input_tokens and remain_images > 0:
        #                 ed_image = input_tokens.index(image_token_id, st)
        #             else:
        #                 ed_image = len(input_tokens) + 1
        #             if video_token_id in input_tokens and remain_videos > 0:
        #                 ed_video = input_tokens.index(video_token_id, st)
        #             else:
        #                 ed_video = len(input_tokens) + 1
        #             if ed_image < ed_video:
        #                 t, h, w = (
        #                     image_grid_thw[image_index][0],
        #                     image_grid_thw[image_index][1],
        #                     image_grid_thw[image_index][2],
        #                 )
        #                 image_index += 1
        #                 remain_images -= 1
        #                 ed = ed_image

        #             else:
        #                 t, h, w = (
        #                     video_grid_thw[video_index][0],
        #                     video_grid_thw[video_index][1],
        #                     video_grid_thw[video_index][2],
        #                 )
        #                 video_index += 1
        #                 remain_videos -= 1
        #                 ed = ed_video
        #             llm_grid_t, llm_grid_h, llm_grid_w = (
        #                 t.item(),
        #                 h.item() // spatial_merge_size,
        #                 w.item() // spatial_merge_size,
        #             )
        #             text_len = ed - st

        #             st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        #             llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        #             # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
        #             t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        #             h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        #             w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        #             llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        #             st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        #         if st < len(input_tokens):
        #             st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        #             text_len = len(input_tokens) - st
        #             llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        #         llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        #         position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        #         mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        #     mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        #     return position_ids, mrope_position_deltas
        # else:
        #     if attention_mask is not None:
        #         position_ids = attention_mask.long().cumsum(-1) - 1
        #         position_ids.masked_fill_(attention_mask == 0, 1)
        #         position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        #         max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        #         mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        #     else:
        #         position_ids = (
        #             torch.arange(input_ids.shape[1], device=input_ids.device)
        #             .view(1, 1, -1)
        #             .expand(3, input_ids.shape[0], -1)
        #         )
        #         mrope_position_deltas = torch.zeros(
        #             [input_ids.shape[0], 1],
        #             device=input_ids.device,
        #             dtype=input_ids.dtype,
        #         )

        #     return position_ids, mrope_position_deltas

    def get_image_features(self, pixel_values: Tensor, image_grid_thw: Optional[Tensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        raise NotImplementedError
        # pixel_values = pixel_values.type(self.visual.dtype)
        # image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        # return image_embeds, deepstack_image_embeds

    def get_video_features(
        self, pixel_values_videos: Tensor, video_grid_thw: Optional[Tensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)


    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        pixel_values_videos: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        raise NotImplementedError
        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds = self.get_input_embeddings()(input_ids)

        # image_mask = None
        # video_mask = None

        # if pixel_values is not None:
        #     image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        #     image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        #     image_mask, _ = self.get_placeholder_mask(
        #         input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        #     )
        #     inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # if pixel_values_videos is not None:
        #     video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        #     video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        #     _, video_mask = self.get_placeholder_mask(
        #         input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        #     )
        #     inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # visual_pos_masks = None
        # deepstack_visual_embeds = None
        # if image_mask is not None and video_mask is not None:
        #     # aggregate visual_pos_masks and deepstack_visual_embeds
        #     image_mask = image_mask[..., 0]
        #     video_mask = video_mask[..., 0]
        #     visual_pos_masks = image_mask | video_mask
        #     deepstack_visual_embeds = []
        #     image_mask_joint = image_mask[visual_pos_masks]
        #     video_mask_joint = video_mask[visual_pos_masks]
        #     for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
        #         embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
        #         embed_joint[image_mask_joint, :] = img_embed
        #         embed_joint[video_mask_joint, :] = vid_embed
        #         deepstack_visual_embeds.append(embed_joint)
        # elif image_mask is not None:
        #     image_mask = image_mask[..., 0]
        #     visual_pos_masks = image_mask
        #     deepstack_visual_embeds = deepstack_image_embeds
        # elif video_mask is not None:
        #     video_mask = video_mask[..., 0]
        #     visual_pos_masks = video_mask
        #     deepstack_visual_embeds = deepstack_video_embeds

        # if position_ids is None:
        #     past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        #     if self.rope_deltas is None or past_key_values_length == 0:
        #         position_ids, rope_deltas = self.get_rope_index(
        #             input_ids,
        #             image_grid_thw,
        #             video_grid_thw,
        #             attention_mask=attention_mask,
        #         )
        #         self.rope_deltas = rope_deltas
        #     # then use the prev pre-calculated rope-deltas to get the correct position ids
        #     else:
        #         batch_size, seq_length, _ = inputs_embeds.shape
        #         delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
        #         position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        #         position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        #         if cache_position is not None:  # otherwise `deltas` is an int `0`
        #             delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        #         position_ids = position_ids.add(delta)
        #         position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # outputs = self.language_model(
        #     input_ids=None,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     cache_position=cache_position,
        #     visual_pos_masks=visual_pos_masks,
        #     deepstack_visual_embeds=deepstack_visual_embeds,
        #     **kwargs,
        # )

        # return Qwen3VLModelOutputWithPast(
        #     last_hidden_state=outputs.last_hidden_state,
        #     past_key_values=outputs.past_key_values,
        #     rope_deltas=self.rope_deltas,
        # )


class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        b, s, d = input_embed.shape
        return op.zeros((b, s, self.config.text_config.vocab_size), dtype="float32"), paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        b, s, d = input_embed.shape
        return op.zeros((b, s, self.config.text_config.vocab_size), dtype="float32"), paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.config.text_config.num_hidden_layers,
            num_attention_heads=self.config.text_config.num_attention_heads // self.config.text_config.tensor_parallel_shards,
            num_key_value_heads=self.config.text_config.num_key_value_heads // self.config.text_config.tensor_parallel_shards,
            qk_head_dim=self.config.text_config.head_dim,
            v_head_dim=self.config.text_config.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.config.text_config.rope_theta,
            dtype=self.config.text_config.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.config.text_config.hidden_size], "float32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.config.text_config.hidden_size], "float32"),
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

