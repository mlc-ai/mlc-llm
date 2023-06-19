# MPT-7b-instruct

There is brief description of mpt-7b-instruct model. It is needed for correct Relax implementation of the model and weights mapping.
MPT-7b-instruct is decoder-like kv_cache free model using flash attention.
Data type is brain float16 by default. But numpy used inside scripts and TVM do not support this type. Due to this to compile MPT-like model use following script:
```bash
python3 bfloat16_to_float16.py
```
It is saved converted model in `dist/models/<model-name>-float16` directory.
**Note:** After conversion to float16, only weights and config will be saved. Transfer other files (like tokenizer vocab) from the original directory.

The list of Tensor name - tensor size for the original (pytorch) model can be found in mpt_topology.txt file.
The original config for the model:
{
  "architectures": [
    "MPTForCausalLM"
  ],
  "attn_config": {
    "alibi": true,
    "alibi_bias_max": 8,
    "attn_impl": "torch",
    "attn_pdrop": 0,
    "attn_type": "multihead_attention",
    "attn_uses_sequence_id": false,
    "clip_qkv": null,
    "prefix_lm": false,
    "qk_ln": false,
    "softmax_scale": null
  },
  "auto_map": {
    "AutoConfig": "configuration_mpt.MPTConfig",
    "AutoModelForCausalLM": "modeling_mpt.MPTForCausalLM"
  },
  "d_model": 4096,
  "emb_pdrop": 0,
  "embedding_fraction": 1.0,
  "expansion_ratio": 4,
  "init_config": {
    "emb_init_std": null,
    "emb_init_uniform_lim": null,
    "fan_mode": "fan_in",
    "init_div_is_residual": true,
    "init_gain": 0,
    "init_nonlinearity": "relu",
    "init_std": 0.02,
    "name": "kaiming_normal_",
    "verbose": 0
  },
  "init_device": "cpu",
  "learned_pos_emb": true,
  "logit_scale": null,
  "max_seq_len": 2048,
  "model_type": "mpt",
  "n_heads": 32,
  "n_layers": 32,
  "no_bias": true,
  "norm_type": "low_precision_layernorm",
  "resid_pdrop": 0,
  "tokenizer_name": "EleutherAI/gpt-neox-20b",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.28.1",
  "use_cache": false,
  "verbose": 0,
  "vocab_size": 50432
}

This config wraps default one. It should highlight two defaults parameters:
"is_encoder_decoder": false,
"use_cache": false,
