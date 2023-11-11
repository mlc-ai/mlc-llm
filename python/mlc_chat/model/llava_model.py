import dataclasses
from typing import Tuple, Optional

import tvm
from tvm import te, tir
from tvm.script import relax as R
import tvm.relax as relax
import tvm.relax.frontend.nn as nn
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

from ...support.config import ConfigBase
from ...support.style import bold

from ..loader import QuantizeMapping
from ..quantization import AWQQuantize, GroupQuantize

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class LlavaConfig(ConfigBase):
    image_size: int = 224
    num_channels: int = 3
    hidden_size: int = 1024
    projection_dim: int = 768
    patch_size: int = 14
    grid_size: int = 16
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-5
    dtype: str = "float16"

    
    max_sequence_length=2048
    head_dim = 0
    num_key_value_heads=0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        pass


# done
class CLIPVisionEmbeddings(Module):
    def __init__(self, config:LlavaConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = Parameter((self.embed_dim,),dtype=config.dtype)
        self.position_ids = Parameter(shape=(1,self.num_positions),dtype="int64") # actuallly not used, just for naming
        self.patch_embedding = Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
            dtype=config.dtype
        )
        
        # for paramatering naming problem, still use Embedding here
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


# done
class LlavaQuickGELU(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        pass
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)

# done
class CLIPMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = LlavaQuickGELU(config=config)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size,dtype=config.dtype)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size,dtype=config.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# done
class CLIPAttention(Module):
    def __init__(self,config:LlavaConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
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
    
    # return a tensor of shape(batch,num_heads,seq_len,head_dim)
    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        reshape_tensor = reshape(tensor,shape=(bsz, seq_len, self.num_heads, self.head_dim))
        permute_tensor = permute_dims(reshape_tensor,axes=(0,2,1,3))
        return permute_tensor
        # return reshape(tensor,shape=(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2).contiguous()

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

#done
class CLIPEncoderLayer(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps,dtype=config.dtype)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps,dtype=config.dtype)
    
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
    
    
        
# done
class CLIPEncoder(Module):
    def __init__(self, config:LlavaConfig):
        super().__init__()
        self.layers = ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(embed_dim, eps=config.layer_norm_eps,dtype=config.dtype)
        self.encoder = CLIPEncoder(config)

        # Even it is not used for output, it still need to be here for paramaters importing
        self.post_layernorm = LayerNorm(embed_dim, eps=config.layer_norm_eps,dtype=config.dtype) 

    def forward(self,pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs

class CLIPVisionModel(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)
    def forward(self,pixel_values):
        return self.vision_model(pixel_values)[-2] # as in the original llava implementation
    
    

class LLavaModel(Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_tower = CLIPVisionModel(config)
    def forward(self,pixel_values):
        return self.vision_tower.forward(pixel_values)
    def test(self):
        import numpy as np
        x = Tensor.from_const(np.zeros(shape=(1,3,224,224),dtype="float16"))
        y = self.forward(x)
        return y
    
    def get_default_spec(self):
        batch_size = 1
        mod_spec = {
            "forward":{
                "pixel_values":spec.Tensor([batch_size, 3, 224, 224], "float16")
            },
            "test":{}
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)



# define the quantization function for llava here

def group_quant(
    model_config: LlavaConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model using group quantization."""
    model: nn.Module = LLavaModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: LlavaConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = LLavaModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map




def main():
    import sys
    import numpy as np
    # Add the path to the sys.path list
    sys.path.append('/home/zhiruiw/LLaVA')

    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

    from llava.model.builder import load_pretrained_model

    def load_image(image_file):
        import requests
        from PIL import Image
        from io import BytesIO
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    model_path = '/ssd1/zhiruiw/llava-llama-2-13b-chat-lightning-preview'
    model_name = get_model_name_from_path(model_path)
    model_base = None

    llava_model = load_pretrained_model(model_path, None, model_name,False,False)
    image_path ="https://llava-vl.github.io/static/images/view.jpg"
    image = load_image(image_path)
    tokenizer, model, image_processor, context_len = llava_model
    model = model
    vision_tower = model.get_model().get_vision_tower()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    llava = LLavaModel(LlavaConfig)

    dev = tvm.cuda()
    target = tvm.target.Target(
        {
            "kind": "cuda",
            # "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            # "max_threads_per_block": dev.max_threads_per_block,
            # "thread_warp_size": dev.warp_size,
            # "registers_per_block": 65536,
            # "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
        }
    )

    mod_spec = {
        "forward":{
            "pixel_values":spec.Tensor([1, 3, 224, 224], "float16")
        },
        "test":{}
    }

    for name, param in llava.state_dict().items():
        param.data = vision_tower.state_dict()[name]
    
    mod, named_params = llava.export_tvm(spec=mod_spec)
    for g_var, func in mod.functions_items():
        if isinstance(func, relax.Function):
            print("this is a func",g_var)
    print("this is the name_params\n",isinstance(named_params[0][1],Parameter))
    
    # # for name, param in llava.state_dict().items():
    # #     param.data = vision_tower.state_dict()[name]

    with target:
        mod = relax.get_pipeline()(mod)
    
    # # llava = llava.jit(spec=mod_spec, target=target, device="cuda", out_format="torch", debug=True)
    # # output_tvm = llava['forward'](image_tensor)
    # ex = relax.build(mod, target)
    # dev = tvm.cuda()
    # vm = relax.VirtualMachine(ex, dev)
    # output_tvm = vm['forward'](image_tensor)

    # output_groud_truth = vision_tower(image_tensor)
    # print(output_tvm==output_groud_truth)

