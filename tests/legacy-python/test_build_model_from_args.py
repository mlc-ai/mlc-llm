import argparse
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from mlc_llm import utils
from mlc_llm.core import build_model_from_args


class MockMkdir(object):
    def __init__(self):
        self.received_args = None

    def __call__(self, *args):
        self.received_args = args


class BuildModelTest(unittest.TestCase):
    def setUp(self):
        self._orig_mkdir = os.mkdir
        os.mkdir = MockMkdir()

        self.mock_args = argparse.Namespace()
        self.mock_args.quantization = utils.quantization_schemes["q8f16_1"]
        self.mock_args.debug_dump = False
        self.mock_args.use_cache = False
        self.mock_args.sep_embed = False
        self.mock_args.build_model_only = True
        self.mock_args.use_safetensors = False
        self.mock_args.convert_weights_only = False
        self.mock_args.no_cutlass_attn = True
        self.mock_args.no_cutlass_norm = True
        self.mock_args.reuse_lib = True
        self.mock_args.artifact_path = "/tmp/"
        self.mock_args.model_path = "/tmp/"
        self.mock_args.model = "/tmp/"
        self.mock_args.target_kind = "cuda"
        self.mock_args.max_seq_len = 2048

    def tearDown(self):
        os.mkdir = self._orig_mkdir

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", MagicMock(side_effect=[{}]))
    def test_llama_model(self, mock_file):
        self.mock_args.model_category = "llama"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch(
        "json.load",
        MagicMock(
            side_effect=[
                {
                    "use_parallel_residual": False,
                    "hidden_size": 32,
                    "intermediate_size": 32,
                    "num_attention_heads": 32,
                    "num_hidden_layers": 28,
                    "vocab_size": 1024,
                    "rotary_pct": 1,
                    "rotary_emb_base": 1,
                    "layer_norm_eps": 1,
                }
            ]
        ),
    )
    def test_gpt_neox_model(self, mock_file):
        self.mock_args.model_category = "gpt_neox"
        self.mock_args.model = "dolly-test"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", MagicMock(side_effect=[{}]))
    def test_gpt_bigcode_model(self, mock_file):
        self.mock_args.model_category = "gpt_bigcode"
        self.mock_args.model = "gpt_bigcode"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", MagicMock(side_effect=[{}]))
    def test_minigpt_model(self, mock_file):
        self.mock_args.model_category = "minigpt"
        self.mock_args.model = "minigpt4-7b"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch(
        "json.load",
        MagicMock(
            side_effect=[
                {
                    "vocab_size": 1024,
                    "n_embd": 32,
                    "n_inner": 32,
                    "n_head": 32,
                    "n_layer": 28,
                    "bos_token_id": 28,
                    "eos_token_id": 1,
                    "rotary_dim": 1,
                    "tie_word_embeddings": 1,
                }
            ]
        ),
    )
    def test_gptj_model(self, mock_file):
        self.mock_args.model_category = "gptj"
        self.mock_args.model = "gpt-j-"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch(
        "json.load",
        MagicMock(
            side_effect=[
                {
                    "num_hidden_layers": 16,
                    "vocab_size": 1024,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                }
            ]
        ),
    )
    def test_rwkv_model(self, mock_file):
        self.mock_args.model_category = "rwkv"
        self.mock_args.model = "rwkv-"

        build_model_from_args(self.mock_args)

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", MagicMock(side_effect=[{}]))
    def test_chatglm_model(self, mock_file):
        self.mock_args.model_category = "chatglm"
        self.mock_args.model = "chatglm2"

        build_model_from_args(self.mock_args)
