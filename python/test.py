from mlc_llm.model.cohere.cohere_model import CohereConfig, CohereForCausalLM
config_dict = {
  "architectures": [
    "CohereForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 5,
  "eos_token_id": 255001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "layer_norm_eps": 1e-05,
  "logit_scale": 0.0625,
  "max_position_embeddings": 8192,
  "model_type": "cohere",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "rope_theta": 10000,
  "vocab_size": 256000
}

config = CohereConfig.from_dict(config_dict)
model = CohereForCausalLM(config)
mod, named_params = model.export_tvm(
    spec=model.get_default_spec(),
)

# Uncomment the following line to show the model in Tensor IR
# mod.show(black_format=False)

for name, param in named_params:
    print(name, param.shape, param.dtype)