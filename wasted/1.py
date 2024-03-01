#  pylint: skip-file
{
    "model_type": "gpt_neox",
    "quantization": "q4f16_1",
    "context_window_size": 2048,
    "prefill_chunk_size": 2048,
    "sliding_window_size": -1,
    "attention_sink_size": -1,
    "tensor_parallel_shards": 2,
    "params": [
        {
            "name": "gpt_neox.embed_in.q_weight",
            "shape": [50432, 320],
            "dtype": "uint32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.embed_in.q_scale",
            "shape": [50432, 80],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.0.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.1.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.2.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.3.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.4.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.5.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.6.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.7.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.8.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.9.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.10.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.11.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.12.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.13.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.14.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.15.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.16.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.17.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.18.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.19.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.20.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.21.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.22.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.23.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.24.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.25.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.26.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.27.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.28.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.29.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.30.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.input_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.input_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.post_attention_layernorm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.post_attention_layernorm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.attention.query_key_value.q_weight",
            "shape": [3840, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_weight",
                    "out_shape": [2, 3840, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.attention.query_key_value.q_scale",
            "shape": [3840, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_qkv_q_scale",
                    "out_shape": [2, 3840, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.attention.query_key_value.bias",
            "shape": [3840],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.attention.dense.q_weight",
            "shape": [2560, 160],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_weight",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.attention.dense.q_scale",
            "shape": [2560, 40],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_q_scale",
                    "out_shape": [2, 2560, 40],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.attention.dense.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_h_to_4h.q_weight",
            "shape": [5120, 320],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_weight",
                    "out_shape": [2, 5120, 320],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_h_to_4h.q_scale",
            "shape": [5120, 80],
            "dtype": "float16",
            "preprocs": [
                {
                    "func": "_shard_dense_h_to_4h_q_scale",
                    "out_shape": [2, 5120, 80],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_h_to_4h.bias",
            "shape": [5120],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_4h_to_h.q_weight",
            "shape": [2560, 640],
            "dtype": "uint32",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_weight",
                    "out_shape": [2, 2560, 640],
                    "out_dtype": "uint32",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_4h_to_h.q_scale",
            "shape": [2560, 160],
            "dtype": "float16",
            "preprocs": [
                {
                    "func_name": "_shard_dense_4h_to_h_q_scale",
                    "out_shape": [2, 2560, 160],
                    "out_dtype": "float16",
                }
            ],
        },
        {
            "name": "gpt_neox.layers.31.mlp.dense_4h_to_h.bias",
            "shape": [2560],
            "dtype": "float32",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.final_layer_norm.weight",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {
            "name": "gpt_neox.final_layer_norm.bias",
            "shape": [2560],
            "dtype": "float16",
            "preprocs": [],
        },
        {"name": "embed_out.q_weight", "shape": [50432, 320], "dtype": "uint32", "preprocs": []},
        {"name": "embed_out.q_scale", "shape": [50432, 80], "dtype": "float16", "preprocs": []},
    ],
    "memory_usage": {
        "_initialize_effect": 0,
        "decode": 16155140,
        "prefill": 460539904,
        "softmax_with_temperature": 0,
    },
}
