from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.op import tensor_to_shape
from tvm import tir
from tvm import relax as rx

from mlc_llm.model.qwen3.qwen3_model import ACT2FN

from .qwen3_vl_config import Qwen3VLVisionConfig
from .qwen2_vl import PatchEmbed, VisionRotaryEmbedding, VisionAttention
from .qwen_2_5_vl import Qwen2_5_VLVisionBlock
from .rot_pos_emb import op_strided_slice, op_power, compute_freq_table_tir, populate_pos_ids_tir

class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(PatchEmbed):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        # TODO - i am assuming tvm has the same conv3d as pytorch
        self.proj = nn.Conv3D(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)


class Qwen3VLVisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # TODO - translate pytorch to tvm
        raise NotImplementedError
        # x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        # x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        # return x


class Qwen3VLVisionAttention(VisionAttention):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:

        # fyi this is weird because the VisionAttention class expects a Qwen2VLVisionConfig param, but hf implementation passes nothing to it?
        super().__init__(config)
        self.dim = config.hidden_size


class Qwen3VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)


class Qwen3VLVisionModel(nn.Module):
    config: Qwen3VLVisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        #super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: Tensor, total_tokens: Tensor) -> Tensor:
        # grid_thw: (N, 3)
        # total_tokens: scalar tensor
        
        # 1. Compute max_hw for rotary embedding precomputation limit
        # Exclude temporal dim (col 0), take max of height (col 1) and width (col 2)
        grid_spatial = op_strided_slice(grid_thw, axes=[1], begin=[1], end=[3])
        max_hw = op.max(grid_spatial) # scalar tensor
        
        # Convert max_hw to PrimExpr
        max_hw_reshaped = op.reshape(max_hw, (1,))
        max_hw_shape = tensor_to_shape(max_hw_reshaped._expr)
        
        m = tir.Var("m", "int64")
        rx.BlockBuilder.current().match_cast(
            max_hw_shape,
            rx.ShapeStructInfo([m])
        )
        
        # 2. Compute frequency table using TIR
        # Calculate inv_freq manually as in VisionRotaryEmbedding
        theta = self.rotary_pos_emb.theta
        dim = self.rotary_pos_emb.dim
        theta_const = Tensor(_expr=rx.const(theta, "float32"))
        inv_freq = op.divide(
            Tensor(_expr=rx.const(1.0, "float32")), 
            op_power(theta_const, (op.arange(0, dim, 2, dtype="float32") / dim))
        )
        
        freq_table = op.tensor_ir_op(
            compute_freq_table_tir,
            "compute_freq_table_tir",
            args=[max_hw, inv_freq], # Pass max_hw tensor directly (scalar)
            out=Tensor.placeholder((m, dim // 2), dtype="float32")
        )
        
        # 3. Populate position IDs using TIR
        merge_size_const = Tensor(_expr=rx.const(self.spatial_merge_size, dtype="int64"))
        
        # Convert total_tokens (scalar Tensor) to PrimExpr for shape
        total_tokens_reshaped = op.reshape(total_tokens, (1,))
        shape_expr = tensor_to_shape(total_tokens_reshaped._expr)
        
        t = tir.Var("t", "int64")
        rx.BlockBuilder.current().match_cast(
            shape_expr,
            rx.ShapeStructInfo([t])
        )

        pos_ids = op.tensor_ir_op(
            populate_pos_ids_tir,
            "populate_pos_ids_tir",
            args=[grid_thw, merge_size_const],
            out=Tensor.placeholder((t, 2), dtype="int64") 
        )
        
        # 4. Gather embeddings
        # op.take with axis=0.
        embeddings = op.take(freq_table, pos_ids, axis=0) # (total_tokens, 2, dim)
        
        # 5. Flatten
        target_dim = self.rotary_pos_emb.dim 
        # We need to reshape embeddings. Shape: (total_tokens, target_dim).
        # We can use the same PrimExpr.
        embeddings = op.reshape(embeddings, (t, target_dim))
        
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        # TODO - translate from pytorch to tvm
        raise NotImplementedError

        # grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        # device = self.pos_embed.weight.device

        # idx_list = [[] for _ in range(4)]
        # weight_list = [[] for _ in range(4)]

        # for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        #     h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        #     w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        #     h_idxs_floor = h_idxs.int()
        #     w_idxs_floor = w_idxs.int()
        #     h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        #     w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        #     dh = h_idxs - h_idxs_floor
        #     dw = w_idxs - w_idxs_floor

        #     base_h = h_idxs_floor * self.num_grid_per_side
        #     base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        #     indices = [
        #         (base_h[None].T + w_idxs_floor[None]).flatten(),
        #         (base_h[None].T + w_idxs_ceil[None]).flatten(),
        #         (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
        #         (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        #     ]

        #     weights = [
        #         ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        #         ((1 - dh)[None].T * dw[None]).flatten(),
        #         (dh[None].T * (1 - dw)[None]).flatten(),
        #         (dh[None].T * dw[None]).flatten(),
        #     ]

        #     for i in range(4):
        #         idx_list[i].extend(indices[i].tolist())
        #         weight_list[i].extend(weights[i].tolist())

        # idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        # weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        # pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        # patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        # patch_pos_embeds_permute = []
        # merge_size = self.config.spatial_merge_size
        # for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        #     pos_embed = pos_embed.repeat(t, 1)
        #     pos_embed = (
        #         pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
        #         .permute(0, 1, 3, 2, 4, 5)
        #         .flatten(0, 4)
        #     )
        #     patch_pos_embeds_permute.append(pos_embed)
        # patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        # return patch_pos_embeds

    def forward(self, hidden_states: Tensor, grid_thw: Tensor, **kwargs) -> Tensor:
        """
        Args:
            hidden_states (`Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # TODO - translate from pytorch to tvm
        raise NotImplementedError

        # hidden_states = self.patch_embed(hidden_states)

        # pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        # hidden_states = hidden_states + pos_embeds

        # rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # seq_len, _ = hidden_states.size()
        # hidden_states = hidden_states.reshape(seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        #     dim=0,
        #     # Select dtype based on the following factors:
        #     #  - FA2 requires that cu_seqlens_q must have dtype int32
        #     #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        #     # See https://github.com/huggingface/transformers/pull/34852 for more information
        #     dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        # )
        # cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # deepstack_feature_lists = []
        # for layer_num, blk in enumerate(self.blocks):
        #     hidden_states = blk(
        #         hidden_states,
        #         cu_seqlens=cu_seqlens,
        #         position_embeddings=position_embeddings,
        #         **kwargs,
        #     )
        #     if layer_num in self.deepstack_visual_indexes:
        #         deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
        #             hidden_states
        #         )
        #         deepstack_feature_lists.append(deepstack_feature)

        # hidden_states = self.merger(hidden_states)

        # return hidden_states, deepstack_feature_lists