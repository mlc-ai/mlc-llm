import math
import os
from dataclasses import dataclass

import torch
import tvm
from tvm import relax
from tvm.relax.testing import nn


from ..quantization import ParamQuantKind, QuantizationScheme
from .modules import ModuleList, TransformImage
from .param_manager import ParamManager


@dataclass
class MiniGPTConfig:
    dtype: str = "float16"
    in_chan: int = 4  # represent rgba
    image_size: int = 224
    num_query_token: int = 32
    max_txt_len: int = 160
    vocab_size: int = 32000
    patch_size: int = 14
    word_embed: int = 768
    visual_encoder_embed_dim: int = 1408
    visual_encoder_attn_heads: int = 16
    visual_encoder_attn_hidden_dim: int = 257
    visual_encoder_fc_hidden_dim: int = 6144
    visual_encoder_num_blocks: int = 39
    bert_hidden_layers: int = 12
    bert_num_attn_heads: int = 12
    bert_attn_head_size: int = 64
    bert_interm_query: int = 3072
    llama_proj_size: int = 4096


MODEL_CONFIG = {
    "minigpt4-7b": {},
}


class MiniGPTPatchEmbed(nn.Module):
    def __init__(
        self, image_size, patch_size, embed_dim, dtype: str, in_chans=3, bias=True
    ):
        self.strides = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.out_shape = image_size // patch_size

        bs = 1
        self.cls_token = nn.Parameter((bs, 1, embed_dim), dtype=dtype, name="cls_token")
        self.pos_embed = nn.Parameter(
            (1, self.out_shape * self.out_shape + 1, embed_dim),
            dtype=dtype,
            name="pos_embed",
        )
        self.weight = nn.Parameter(
            (embed_dim, in_chans, patch_size, patch_size),
            dtype=dtype,
            name="patch_embed_weight",
        )
        if bias:
            self.bias = nn.Parameter((embed_dim,), dtype=dtype, name="patch_embed_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        bs = 1
        x = nn.emit(relax.op.nn.conv2d(input, self.weight, self.strides))
        if self.bias:
            bias = relax.op.reshape(self.bias, [1, self.embed_dim, 1, 1])
            x = relax.op.add(x, bias)
        x = relax.op.reshape(x, (bs, self.embed_dim, self.out_shape * self.out_shape))
        x = relax.op.permute_dims(x, [0, 2, 1])
        # concatenate with cls_tokens
        x_concat = relax.op.concat([self.cls_token, x], axis=1)
        # add with pos_embed
        res = relax.op.add(x_concat, self.pos_embed)
        return res


class MiniGPTVisualEncoderAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.embed_dim = config.visual_encoder_embed_dim
        self.num_heads = config.visual_encoder_attn_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)
        self.dtype = config.dtype
        self.N = config.visual_encoder_attn_hidden_dim

        self.q_bias = nn.Parameter((self.embed_dim,), dtype=self.dtype, name="q_bias")
        self.v_bias = nn.Parameter((self.embed_dim,), dtype=self.dtype, name="v_bias")
        self.qkv_weight = nn.Parameter(
            (self.embed_dim * 3, self.embed_dim), dtype=self.dtype, name="qkv_weight"
        )
        self.proj_weight = nn.Parameter(
            (self.embed_dim, self.embed_dim), dtype=self.dtype, name="proj_weight"
        )
        self.proj_bias = nn.Parameter(
            (self.embed_dim,), dtype=self.dtype, name="proj_bias"
        )

    def forward(self, input: relax.Expr):
        from tvm.relax.op import (
            concat,
            linear,
            matmul,
            permute_dims,
            reshape,
            squeeze,
            strided_slice,
            zeros,
        )

        bs = 1
        k_bias = zeros((self.embed_dim,), self.dtype)
        qkv_bias = concat([self.q_bias, k_bias, self.v_bias], axis=0)
        x = linear(input, self.qkv_weight, qkv_bias)
        x = reshape(x, (bs, self.N, 3, self.num_heads, self.head_dim))
        x = permute_dims(x, [2, 0, 3, 1, 4])
        q = squeeze(strided_slice(x, axes=[0], begin=[0], end=[1]), [0])
        k = squeeze(strided_slice(x, axes=[0], begin=[1], end=[2]), [0])
        v = squeeze(strided_slice(x, axes=[0], begin=[2], end=[3]), [0])
        q = q * relax.const(self.scale, self.dtype)
        attn = matmul(q, permute_dims(k, [0, 1, 3, 2]))
        attn = relax.op.nn.softmax(attn, -1)
        res = permute_dims(matmul(attn, v), [0, 2, 1, 3])
        res = reshape(res, (bs, self.N, self.embed_dim))
        res = linear(res, self.proj_weight, self.proj_bias)
        return res


class MiniGPTMLP(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.hidden_dim = config.visual_encoder_fc_hidden_dim
        self.embed_dim = config.visual_encoder_embed_dim
        self.dtype = config.dtype

        self.fc1_weight = nn.Parameter(
            (self.hidden_dim, self.embed_dim), dtype=self.dtype, name="fc1_weight"
        )
        self.fc1_bias = nn.Parameter(
            (self.hidden_dim,), dtype=self.dtype, name="fc1_bias"
        )
        self.fc2_weight = nn.Parameter(
            (self.embed_dim, self.hidden_dim), dtype=self.dtype, name="fc2_weight"
        )
        self.fc2_bias = nn.Parameter(
            (self.embed_dim,), dtype=self.dtype, name="fc2_bias"
        )

    def forward(self, input: relax.Expr):
        res = relax.op.linear(input, self.fc1_weight, self.fc1_bias)
        res = relax.op.nn.gelu(res)
        res = relax.op.linear(res, self.fc2_weight, self.fc2_bias)
        return res


class MiniGPTVisualEncoderBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        embed_dim = config.visual_encoder_embed_dim
        dtype = config.dtype
        self.norm1_weight = nn.Parameter((embed_dim,), dtype=dtype, name="norm1_weight")
        self.norm1_bias = nn.Parameter((embed_dim,), dtype=dtype, name="norm1_bias")
        self.attn = MiniGPTVisualEncoderAttention(config)
        self.norm2_weight = nn.Parameter((embed_dim,), dtype=dtype, name="norm2_weight")
        self.norm2_bias = nn.Parameter((embed_dim,), dtype=dtype, name="norm2_bias")
        self.mlp = MiniGPTMLP(config)

    def forward(self, input: relax.Expr):
        x = relax.op.nn.layer_norm(input, self.norm1_weight, self.norm1_bias, axes=[-1])
        proj = self.attn(x)
        proj = relax.op.add(input, proj)
        res = relax.op.nn.layer_norm(
            proj, self.norm2_weight, self.norm2_bias, axes=[-1]
        )
        res = self.mlp(res)
        res = relax.op.add(proj, res)
        return res


class MiniGPTVisualEncoder(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.embed_dim = config.visual_encoder_embed_dim
        self.dtype = config.dtype
        self.transform = TransformImage(config.dtype, config.in_chan)
        self.patch_embed = MiniGPTPatchEmbed(
            config.image_size,
            config.patch_size,
            config.visual_encoder_embed_dim,
            config.dtype,
        )
        self.num_blocks = config.visual_encoder_num_blocks
        self.blocks = ModuleList(
            [MiniGPTVisualEncoderBlock(config) for _ in range(self.num_blocks)]
        )

        self.ln_vision_weight = nn.Parameter(
            (self.embed_dim,), dtype=self.dtype, name="ln_vision_weight"
        )
        self.ln_vision_bias = nn.Parameter(
            (self.embed_dim,), dtype=self.dtype, name="ln_vision_bias"
        )

    def forward(self, input_image: relax.Expr):
        res = self.transform(input_image)
        res = self.patch_embed(res)
        for block in self.blocks:
            res = block(res)
        res = relax.op.nn.layer_norm(
            res, self.ln_vision_weight, self.ln_vision_bias, axes=[-1]
        )
        return res


class MiniGPTEmbedding(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.word_embed = config.word_embed
        self.dtype = config.dtype
        self.eps = 1e-12

        self.norm_weight = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_weight"
        )
        self.norm_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_bias"
        )

    def forward(self, embedding: relax.Expr):
        res = relax.op.nn.layer_norm(
            embedding, self.norm_weight, self.norm_bias, axes=[-1], epsilon=self.eps
        )
        return res


class MiniGPTBertAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig, hidden_dim: int):
        self.word_embed = config.word_embed
        self.num_query_token = config.num_query_token
        self.num_attn_heads = config.bert_num_attn_heads
        self.attn_head_size = config.bert_attn_head_size
        self.visual_encoder_attn_hidden_dim = config.visual_encoder_attn_hidden_dim
        self.dtype = config.dtype
        self.eps = 1e-12

        self.query_weight = nn.Parameter(
            (self.word_embed, self.word_embed), dtype=self.dtype, name="query_weight"
        )
        self.query_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="query_bias"
        )
        self.key_weight = nn.Parameter(
            (self.word_embed, hidden_dim), dtype=self.dtype, name="key_weight"
        )
        self.key_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="key_bias"
        )
        self.value_weight = nn.Parameter(
            (self.word_embed, hidden_dim), dtype=self.dtype, name="value_weight"
        )
        self.value_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="value_bias"
        )
        self.dense_weight = nn.Parameter(
            (self.word_embed, self.word_embed), dtype=self.dtype, name="dense_weight"
        )
        self.dense_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="dense_bias"
        )
        self.norm_weight = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_weight"
        )
        self.norm_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_bias"
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        attention_mask: relax.Expr,
        encoder_hidden_states=None,
        encoder_extend_attention_mask=None,
    ):
        from tvm.relax.op import add, linear, matmul, permute_dims, reshape

        bs = 1
        states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        mask = (
            encoder_extend_attention_mask
            if encoder_extend_attention_mask is not None
            else attention_mask
        )
        hidden_dim = (
            self.visual_encoder_attn_hidden_dim
            if encoder_hidden_states is not None
            else self.num_query_token
        )
        key = linear(states, self.key_weight, self.key_bias)
        value = linear(states, self.value_weight, self.value_bias)
        key = reshape(key, [bs, hidden_dim, self.num_attn_heads, self.attn_head_size])
        key = permute_dims(key, [0, 2, 1, 3])
        value = reshape(
            value, [bs, hidden_dim, self.num_attn_heads, self.attn_head_size]
        )
        value = permute_dims(value, [0, 2, 1, 3])
        query = linear(hidden_states, self.query_weight, self.query_bias)
        query = reshape(
            query, [bs, self.num_query_token, self.num_attn_heads, self.attn_head_size]
        )
        query = permute_dims(query, [0, 2, 1, 3])
        scores = matmul(query, permute_dims(key, [0, 1, 3, 2]))
        scores = scores / relax.const(math.sqrt(self.attn_head_size), dtype=self.dtype)
        scores = add(scores, mask)
        probs = relax.op.nn.softmax(scores, axis=-1)
        context = matmul(probs, value)
        context = permute_dims(context, [0, 2, 1, 3])
        context = reshape(context, [bs, self.num_query_token, self.word_embed])
        # calculate the output
        context = linear(context, self.dense_weight, self.dense_bias)
        context = add(context, hidden_states)
        res = relax.op.nn.layer_norm(
            context, self.norm_weight, self.norm_bias, axes=[-1], epsilon=self.eps
        )
        return res, key, value


class MiniGPTBertLayer(nn.Module):
    def __init__(self, config: MiniGPTConfig, use_cross_attention=False):
        self.word_embed = config.word_embed
        self.embed_dim = config.visual_encoder_embed_dim
        self.interm_query = config.bert_interm_query
        self.dtype = config.dtype
        self.eps = 1e-12

        self.attention = MiniGPTBertAttention(config, self.word_embed)
        if use_cross_attention:
            self.cross_attention = MiniGPTBertAttention(config, self.embed_dim)
        else:
            self.cross_attention = None
        self.interm_query_weight = nn.Parameter(
            (self.interm_query, self.word_embed),
            dtype=self.dtype,
            name="interm_query_weight",
        )
        self.interm_query_bias = nn.Parameter(
            (self.interm_query,), dtype=self.dtype, name="interm_query_bias"
        )
        self.output_query_weight = nn.Parameter(
            (self.word_embed, self.interm_query),
            dtype=self.dtype,
            name="output_query_weight",
        )
        self.output_query_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="output_query_bias"
        )
        self.norm_weight = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_weight"
        )
        self.norm_bias = nn.Parameter(
            (self.word_embed,), dtype=self.dtype, name="norm_bias"
        )

    def forward(
        self,
        embedding: relax.Expr,
        extend_attention_mask: relax.Expr,
        encoder_hidden_states: relax.Expr,
        encoder_extend_attention_mask: relax.Expr,
    ):
        attn_output, key, value = self.attention(embedding, extend_attention_mask)
        if self.cross_attention:
            attn_output, _, _ = self.cross_attention(
                attn_output,
                extend_attention_mask,
                encoder_hidden_states,
                encoder_extend_attention_mask,
            )
        res = relax.op.linear(
            attn_output, self.interm_query_weight, self.interm_query_bias
        )
        res = relax.op.nn.gelu(res)
        res = relax.op.linear(res, self.output_query_weight, self.output_query_bias)
        res = relax.op.add(res, attn_output)
        res = relax.op.nn.layer_norm(
            res, self.norm_weight, self.norm_bias, axes=[-1], epsilon=self.eps
        )
        return res, key, value


class MiniGPTQFormer(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.N = config.visual_encoder_attn_hidden_dim
        self.num_query_token = config.num_query_token
        self.word_embed = config.word_embed
        self.num_layers = config.bert_hidden_layers
        self.dtype = config.dtype

        bs = 1
        self.query_tokens = nn.Parameter(
            (bs, self.num_query_token, self.word_embed),
            dtype=self.dtype,
            name="query_tokens",
        )
        self.embedding = MiniGPTEmbedding(config)
        self.bert_layers = ModuleList(
            [MiniGPTBertLayer(config, i % 2 == 0) for i in range(self.num_layers)]
        )

    def forward(self, image_embeds: relax.Expr):
        from tvm.relax.op import expand_dims, ones

        bs = 1
        image_attns = ones((bs, self.N), self.dtype)
        embedding = self.embedding(self.query_tokens)
        attention_mask = ones((bs, self.num_query_token), self.dtype)
        extend_attention_mask = expand_dims(attention_mask, [1, 2])
        extend_attention_mask = (
            relax.const(1.0, self.dtype) - extend_attention_mask
        ) * relax.const(-10000.0, self.dtype)
        encoder_extend_attention_mask = expand_dims(image_attns, [1, 2])
        encoder_extend_attention_mask = (
            relax.const(1.0, self.dtype) - encoder_extend_attention_mask
        )
        for layer in self.bert_layers:
            embedding, _, _ = layer(
                embedding,
                extend_attention_mask,
                image_embeds,
                encoder_extend_attention_mask,
            )
        return embedding


class MiniGPTLLaMAProj(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.proj_size = config.llama_proj_size
        self.word_embed = config.word_embed
        self.dtype = config.dtype

        self.weight = nn.Parameter(
            (self.proj_size, self.word_embed), dtype=self.dtype, name="weight"
        )
        self.bias = nn.Parameter((self.proj_size,), dtype=self.dtype, name="bias")

    def forward(self, embedding: relax.Expr):
        return relax.op.linear(embedding, self.weight, self.bias)


class MiniGPTModel(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        self.visual_encoder = MiniGPTVisualEncoder(config)
        self.q_former = MiniGPTQFormer(config)
        self.llama_proj = MiniGPTLLaMAProj(config)

    def forward(self, input_image: relax.Expr):
        output = self.visual_encoder(input_image)
        output = self.q_former(output)
        output = self.llama_proj(output)
        return output


def get_param_quant_kind(
    name: str, param_info: relax.TensorStructInfo
) -> ParamQuantKind:
    """No quantization for MiniGPT. Use q0f16 or q0f32 when building it."""
    return ParamQuantKind.others


def create_embed_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: MiniGPTConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bs = 1
    with bb.function(func_name):
        model = MiniGPTModel(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_image = nn.Placeholder(
            (bs, config.image_size, config.image_size, config.in_chan),
            dtype="uint8",
            name="input_image",
        )
        with bb.dataflow():
            output = model(input_image)
            params = [input_image] + model.parameters()
            gv = bb.emit_output(output)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def get_model(args, _config):
    model_name = args.model
    model_path = args.model_path

    if model_name.startswith("minigpt"):
        config = MiniGPTConfig(**MODEL_CONFIG[model_name])
        config.dtype = args.quantization.model_dtype
        # build the relax model
        param_manager = ParamManager()
        bb = relax.BlockBuilder()
        create_embed_func(bb, param_manager, config, args.quantization)
        mod = bb.get()

        if args.build_model_only:
            return mod, param_manager, None, config

        param_manager.set_param_loading_func(
            args.model_path, args.use_safetensors, no_lazy_param_loading=True
        )

        # load visual encoder weights
        visual_encoder_url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
        visual_encoder_cached_file = download_cached_file(
            visual_encoder_url, check_hash=False, progress=True
        )
        visual_encoder_state_dict = torch.load(
            visual_encoder_cached_file, map_location="cpu"
        )

        # load QFormer weights
        q_former_url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        q_former_cached_file = download_cached_file(
            q_former_url, check_hash=False, progress=True
        )
        q_former_state_dict = torch.load(q_former_cached_file, map_location="cpu")[
            "model"
        ]

        # load llama and llama proj weights
        if os.path.isdir(model_path):
            raise ValueError(
                "MiniGPT model path should be a single file instead of a directory."
            )
        llama_state_dict = torch.load(model_path + ".pth", map_location="cpu")["model"]

        param_list = []
        device = tvm.cpu()
        visual_encoder_key_list = list(visual_encoder_state_dict.keys())[
            : 4 + 13 * config.visual_encoder_num_blocks
        ]
        for key in visual_encoder_key_list:
            param_list.append(
                tvm.nd.array(
                    visual_encoder_state_dict[key].numpy().astype(config.dtype), device
                )
            )
        q_former_key_list = (
            list(q_former_state_dict.keys())[1:3]
            + [list(q_former_state_dict.keys())[0]]
            + list(q_former_state_dict.keys())[
                6 : 8 + (26 + 16) * config.bert_hidden_layers // 2
            ]
        )
        for key in q_former_key_list:
            param_list.append(
                tvm.nd.array(
                    q_former_state_dict[key].numpy().astype(config.dtype), device
                )
            )
        llama_key_list = list(llama_state_dict.keys())[-2:]
        for key in llama_key_list:
            param_list.append(
                tvm.nd.array(llama_state_dict[key].numpy().astype(config.dtype), device)
            )

        return mod, param_manager, param_list, config

    raise ValueError(f"Unsupported model: {model_name}")


# helper functions for distributed download of model weights from URL
# source: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/common/dist_utils.py (originally credit to Salesforce)


def download_cached_file(url, check_hash=True, progress=False):
    import timm.models.hub as timm_hub
    import torch.distributed as dist

    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_rank():
        if not is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process():
        return get_rank() == 0

    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()
