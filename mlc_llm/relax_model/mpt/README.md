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
  **"use_cache": false,**
  "verbose": 0,
  "vocab_size": 50432
}

This config wraps default one (see below). It should highlight two defaults parameters:
"is_encoder_decoder": false,
"use_cache": false,

Default config parameters (PretrainedConfig):
"return_dict": True
"output_hidden_states": False
"output_attentions": False
"torchscript": False
"torch_dtype": None
"use_bfloat16": False
"tf_legacy_loss": False
"pruned_heads": {}
"tie_word_embeddings": True

**"is_encoder_decoder": False**
"is_decoder": False
"cross_attention_hidden_size": None
"add_cross_attention": False
"tie_encoder_decoder": False

"max_length": 20
"min_length": 0
"do_sample": False
"early_stopping": False
"num_beams": 1
"num_beam_groups": 1
"diversity_penalty": 0.0
"temperature": 1.0
"top_k": 50
"top_p": 1.0
"typical_p": 1.0
"repetition_penalty": 1.0
"length_penalty": 1.0
"no_repeat_ngram_size": 0
"encoder_no_repeat_ngram_size": 0
"bad_words_ids": None
"num_return_sequences": 1
"chunk_size_feed_forward": 0
"output_scores": False
"return_dict_in_generate": False
"forced_bos_token_id": None
"forced_eos_token_id": None
"remove_invalid_values": False
"exponential_decay_length_penalty": None
"suppress_tokens": None
"begin_suppress_tokens": None

"architectures": None
"finetuning_task": None
"id2label": None
"label2id": None
if self.id2label is not None:
    "num_labels": None
    id2label = dict((int(key), value) for key, value in id2label.items())
else:
    "num_labels": 2

"tokenizer_class": None
"prefix": None
"bos_token_id": None
"pad_token_id": None
"eos_token_id": None
"sep_token_id": None

"decoder_start_token_id": None

"task_specific_params": None


Refactored greedy_search method for MPT-7b-instruct:
```python
def greedy_search(...):
  # init values
  logits_processor = LogitsProcessorList()
  stopping_criteria = stopping_criteria # max_length and max_time criteria
  pad_token_id = None
  eos_token_id = None
  output_scores = False
  output_attentions = False
  output_hidden_states = False
  return_dict_in_generate = False

  # init attention / hidden states / scores tuples
  scores = None
  decoder_attentions = None
  decoder_hidden_states = None

  # keep track of which sequences are already finished
  unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

  while True:
    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution. Due to logits_processor is empty next_tokens_scores = next_token_logits
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # argmax
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)

    # update generated ids, model inputs, and length for next step
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    )

    # stop when each sentence is finished, or if we exceed the maximum length
    if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
        break
```