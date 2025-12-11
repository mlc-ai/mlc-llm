from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.op import tensor_to_shape
from tvm import tir
from tvm import relax as rx

from mlc_llm.model.qwen3.qwen3_model import ACT2FN

from .qwen3_vl_config import Qwen3VLVisionConfig
from .qwen2_vl import PatchEmbed, VisionRotaryEmbedding, VisionAttention
from .qwen_2_5_vl import Qwen2_5_VLVisionBlock
from .vision_pos_emb import op_strided_slice, op_power, compute_freq_table_tir, populate_pos_ids_tir, fast_pos_embed_interpolate_tir


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

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Input hidden_states: (N, C, T, H, W)
        # Conv3D Output: (N, C, D, H, W)
        hidden_states = self.proj(hidden_states)
        
        # Permute to (N, D, H, W, C) for channel-last flattening
        hidden_states = op.permute_dims(hidden_states, (0, 2, 3, 4, 1))
        
        # Flatten to (N*D*H*W, C)
        hidden_states = op.reshape(hidden_states, (-1, self.embed_dim))
        return hidden_states


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
        if self.use_postshuffle_norm:
            x = op.reshape(x, (-1, self.hidden_size))
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = op.reshape(x, (-1, self.hidden_size))
        
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


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
        self.config = config
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

    def fast_pos_embed_interpolate(self, grid_thw: Tensor, total_tokens: Tensor) -> Tensor:
        # grid_thw: (N, 3)
        # total_tokens: scalar tensor
        
        # Inputs for TIR
        pos_embed_weight = self.pos_embed.weight # (num_pos_emb, hidden_size)
        
        num_grid_per_side_const = Tensor(_expr=rx.const(self.num_grid_per_side, "int64"))
        spatial_merge_size_const = Tensor(_expr=rx.const(self.spatial_merge_size, "int64"))
        
        # Shapes for output
        # total_tokens is scalar, we need to extract PrimExpr
        total_tokens_reshaped = op.reshape(total_tokens, (1,))
        shape_expr = tensor_to_shape(total_tokens_reshaped._expr)
        
        t = tir.Var("t", "int64")
        rx.BlockBuilder.current().match_cast(
            shape_expr,
            rx.ShapeStructInfo([t])
        )
        
        # Call TIR
        patch_pos_embeds = op.tensor_ir_op(
            fast_pos_embed_interpolate_tir,
            "fast_pos_embed_interpolate_tir",
            args=[grid_thw, pos_embed_weight, num_grid_per_side_const, spatial_merge_size_const],
            out=Tensor.placeholder((t, self.config.hidden_size), dtype="float32")
        )
        
        return patch_pos_embeds

    def forward(self, hidden_states: Tensor, grid_thw: Tensor, **kwargs) -> Tensor:
        """
        Args:
            hidden_states: Input features (flattened or spatial)
            grid_thw: (N, 3) tensor
        """
        # 1. Patch Embedding
        hidden_states = self.patch_embed(hidden_states)
        # hidden_states is now (total_tokens, hidden_size)
        
        # Calculate total_tokens from grid
        t = op_strided_slice(grid_thw, axes=[1], begin=[0], end=[1])
        h = op_strided_slice(grid_thw, axes=[1], begin=[1], end=[2])
        w = op_strided_slice(grid_thw, axes=[1], begin=[2], end=[3])
        counts = t * h * w
        counts = op.reshape(counts, (-1,))
        total_tokens = op.sum(counts)
        
        # Bind total_tokens to symbolic var 't_var' to guide shape inference
        total_tokens_reshaped = op.reshape(total_tokens, (1,))
        shape_expr = tensor_to_shape(total_tokens_reshaped._expr)
        t_var = tir.Var("t", "int64")
        rx.BlockBuilder.current().match_cast(shape_expr, rx.ShapeStructInfo([t_var]))
        
        # Enforce shape on hidden_states
        hidden_states = op.reshape(hidden_states, (t_var, self.config.hidden_size))
        
        # 2. Fast Pos Embed Interpolation
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw, total_tokens)
        
        # Cast pos_embeds to (t_var, hidden_size) to match hidden_states symbolically
        pos_embeds = op.reshape(pos_embeds, (t_var, self.config.hidden_size))
        
        hidden_states = hidden_states + pos_embeds
        
        # 3. Rotary Pos Emb
        rotary_pos_emb = self.rot_pos_emb(grid_thw, total_tokens)
        
        # Reshape to (1, total_tokens, hidden_size) for Blocks which expect 3D input
        hidden_states = op.reshape(hidden_states, (1, t_var, self.config.hidden_size))
        
        # 4. cu_seqlens
        # Relax cumsum on axis 0
        cu_seqlens = op.cumsum(counts, axis=0, dtype="int32")
        # Pad with 0. 
        zero_pad = Tensor(_expr=rx.const([0], "int32"))
        cu_seqlens = op.concat([zero_pad, cu_seqlens], dim=0)
        
        # 5. Blocks
        deepstack_feature_lists = []
        
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                # position_embeddings passed as None or omitted? 
                # Qwen2.5 block signature has position_embeddings optional
                **kwargs
            )
            
            if layer_num in self.deepstack_visual_indexes:
                # Find index in deepstack_merger_list
                merger_idx = self.deepstack_visual_indexes.index(layer_num)
                merger = self.deepstack_merger_list[merger_idx]
                deepstack_feature = merger(hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
                
        # 6. Final Merger
        hidden_states = self.merger(hidden_states)
        
        output = [hidden_states]
        output.extend(deepstack_feature_lists)
        
        if len(output) == 1:
            return output[0]
        return output