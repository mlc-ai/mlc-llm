import dataclasses
from typing import Tuple, Optional

import tvm
from tvm import te, tir
from tvm.script import relax as R
import tvm.relax as relax
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn import op
from tvm.relax.frontend.nn.modules import (
    Conv2D,
    Parameter,
    Embedding,
    ModuleList,
    LayerNorm,
    Linear
)
from tvm.relax.frontend.nn import(
    Tensor,
    Module
)
from tvm.relax.frontend.nn.op import(
    reshape,
    permute_dims,
    repeat,
    broadcast_to,
    concat,
    softmax,
    matmul,
    _wrap_nested,
    zeros
)
import tvm.relax.frontend.nn.spec as spec

import dataclasses
import logging
import math
from typing import Any, Dict, Optional

from tvm import te, tir

from ....support.config import ConfigBase
from ....support.style import bold


from ..llama.llama_model import LlamaModel,LlamaConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class LlavaConfig(LlamaModel):
    vision_image_size: int = 224
    vision_num_channels: int = 3
    vision_hidden_size: int = 1024
    vision_patch_size: int = 14
    vision_grid_size: int = 16
    vision_intermediate_size: int = 4096
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_layer_norm_eps: float = 1e-5
    vision_image_path:str = "https://llava-vl.github.io/static/images/view.jpg"
    mm_hidden_size: int = 1024
    dtype: str = "float16"    
    hidden_size: int =  5120
    intermediate_size: int = 13824
    num_attention_heads: int = 40
    num_hidden_layers: int = 40
    rms_norm_eps: float = 1e-05
    vocab_size: int = 32000
    position_embedding_base: int = 0
    context_window_size: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    prefill_chunk_size: int = 0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size

        if self.prefill_chunk_size == 0:
            # chunk size same as context window size by default
            self.prefill_chunk_size = self.context_window_size



class CLIPVisionEmbeddings(Module):
    def __init__(self, config:LlavaConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        self.vision_image_size = config.vision_image_size
        self.vision_patch_size = config.vision_patch_size

        self.num_patches = (self.vision_image_size // self.vision_patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = Parameter((self.embed_dim,),dtype=config.dtype)
        self.position_ids = Parameter(shape=(1,self.num_positions),dtype="int64")
        self.patch_embedding = Conv2D(
            in_channels=config.vision_num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.vision_patch_size,
            stride=self.vision_patch_size,
            bias=False,
            dtype=config.dtype
        )
        
        
        self.position_embedding = Embedding(num=self.num_positions,dim=self.embed_dim,dtype=config.dtype)
        

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = reshape(patch_embeds,shape=(batch_size,self.embed_dim,-1))
        patch_embeds = permute_dims(patch_embeds,axes=(0,2,1)) # shape = [batch,grid*grid,embed_dim]
        class_embeds = broadcast_to(self.class_embedding,shape=(batch_size,1,self.embed_dim)) # shape of (batch,1,embed_dim)
        embeddings  = concat([class_embeds,patch_embeds],dim=1)
        import numpy as np
        posi_ids = reshape(Tensor.from_const(np.arange(self.num_positions)), shape=(1,-1))
        batch_position_embedding = broadcast_to(self.position_embedding(posi_ids),shape=(batch_size,self.num_positions,self.embed_dim))
        embeddings = embeddings +  batch_position_embedding
        return embeddings
    
   

def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    """Sigmoid of a Tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Sigmoid result.
    """
    return _wrap_nested(relax.op.sigmoid(x._expr), name)



class LlavaQuickGELU(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        pass
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)


class CLIPMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = LlavaQuickGELU(config=config)
        self.fc1 = Linear(config.vision_hidden_size, config.vision_intermediate_size,dtype=config.dtype)
        self.fc2 = Linear(config.vision_intermediate_size, config.vision_hidden_size,dtype=config.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(Module):
    def __init__(self,config:LlavaConfig):
        super().__init__()
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = Linear(self.embed_dim, self.embed_dim,dtype=config.dtype)
        self.v_proj = Linear(self.embed_dim, self.embed_dim,dtype=config.dtype)
        self.q_proj = Linear(self.embed_dim, self.embed_dim,dtype=config.dtype)
        self.out_proj = Linear(self.embed_dim, self.embed_dim,dtype=config.dtype)
    
    
    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        reshape_tensor = reshape(tensor,shape=(bsz, seq_len, self.num_heads, self.head_dim))
        permute_tensor = permute_dims(reshape_tensor,axes=(0,2,1,3))
        return permute_tensor

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) ->Tensor:
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states) * self.scale,tgt_len,bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim) # shape of (batch*num_heads, seq_len,head_dim)

        query_states = reshape(query_states,shape=proj_shape)
        key_states = reshape(key_states,shape=proj_shape)
        value_states = reshape(value_states,shape=proj_shape)

        trans_key_states = permute_dims(key_states,axes=(0,2,1))
        attn_weights = matmul(query_states, trans_key_states)

        if attn_weights.shape != [bsz * self.num_heads, tgt_len, tgt_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, tgt_len)}, but is"
                f" {attn_weights.shape}"
            )

        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)
        if attn_output.shape != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )
        attn_output = reshape(attn_output,shape=(bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = permute_dims(attn_output,axes=(0,2,1,3))
        attn_output = reshape(attn_output,shape=(bsz,tgt_len,embed_dim))
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class CLIPEncoderLayer(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.embed_dim = config.vision_hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(normalized_shape=self.embed_dim, eps=config.vision_layer_norm_eps,dtype=config.dtype)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(normalized_shape=self.embed_dim, eps=config.vision_layer_norm_eps,dtype=config.dtype)
    
    def forward(self,hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs
    


class CLIPEncoder(Module):
    def __init__(self, config:LlavaConfig):
        super().__init__()
        self.layers = ModuleList([CLIPEncoderLayer(config) for _ in range(config.vision_num_hidden_layers)])
    def forward(self,inputs_embeds):
        hidden_states = inputs_embeds
        encoder_states=()
        for idx,encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states
            )
            hidden_states = layer_outputs[0]
        encoder_states = encoder_states + (hidden_states,)
        return encoder_states
    


class CLIPVisionTransformer(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        embed_dim = config.vision_hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(embed_dim, eps=config.vision_layer_norm_eps,dtype=config.dtype)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = LayerNorm(embed_dim, eps=config.vision_layer_norm_eps,dtype=config.dtype) 

    def forward(self,pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs

class CLIPVisionModel(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)
    def forward(self,pixel_values)->Tensor:
        return self.vision_model(pixel_values)[-2]
    
    

class LlavaForCasualLM(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = CLIPVisionModel(config)
        self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.llama_model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = "float16"
    def forward(self,pixel_values, inputs: Tensor, total_seq_len: tir.Var, attention_mask: Tensor):
        image_features = self.vision_tower.forward(pixel_values)
        image_features = reshape(image_features,shape=(-1,1,self.config.vision_hidden_size))
        print("shape of image features",image_features.shape)

        
        hidden_states = self.llama_model.embed_tokens(inputs)
        hidden_states = concat([image_features,hidden_states], dim=1)

    
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len + 1)
        hidden_states = self.norm(hidden_states)

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def test(self):
        import numpy as np
        x = Tensor.from_const(np.zeros(shape=(1,3,224,224),dtype="float16"))
        y = self.forward(x)
        return y
    
    def prefill(self,pixel_values, inputs: Tensor, total_seq_len: tir.Var):
        def _attention_mask(batch_size, seq_len, total_seq_len):
            return te.compute(
                (batch_size, 1, seq_len, total_seq_len),
                lambda b, _, i, j: tir.if_then_else(
                    i < j - (total_seq_len - seq_len),
                    tir.min_value(self.dtype),
                    tir.max_value(self.dtype),
                ),
                name="attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask = op.tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[batch_size, seq_len, total_seq_len],
        )
        return self.forward(pixel_values,inputs, total_seq_len, attention_mask)

    def decode(self,pixel_values, inputs: Tensor, total_seq_len: tir.Var):
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, total_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(pixel_values, inputs, total_seq_len, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)




