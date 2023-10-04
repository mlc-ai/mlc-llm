import unittest
from unittest.mock import MagicMock, patch

from mlc_chat.chat_module import ChatConfig, ChatModule, ConvConfig

class UpdateConfigTest(unittest.TestCase):

    
    @patch("mlc_chat.chat_module.ChatModule.__init__")
    def setUp(self, mock_init):
        mock_init.return_value = None
        self.cm_under_test = ChatModule("test")
        default_conv_config = {
            "prefix_tokens": [],
            "role_empty_sep": "",
            "role_msg_sep": "",
            "seps": [""],
            "stop_tokens": [2],
            "offset": 0,
            "separator_style": 1,
            "messages": [],
            "stop_str": "<\/s>",
            "roles": ["Prompt", "Code"],
            "system": "",
            "add_bos": True,
            "name": "codellama_completion"
        }
        default_chat_config = {
            'model_lib': 'default_model_lib', 
            'local_id': 'default_local_id', 
            'conv_template': 'codellama_completion', 
            'temperature': 0.7, 
            'repetition_penalty': 1.0, 
            'top_p': 0.95, 
            'mean_gen_len': 128, 
            'max_gen_len': 512, 
            'shift_fill_factor': 0.3, 
            'tokenizer_files': ['tokenizer.json', 'tokenizer.model'], 
            'conv_config': None, 
            'model_category': 'llama', 
            'model_name': 'default_model_name'
        }
        self.cm_under_test.default_chat_config = default_chat_config
        self.cm_under_test.default_conv_config = default_conv_config
        self.cm_under_test._load_json_override_func = MagicMock()
    
    def test_update_config(self):
        expected_value = '{"model_lib": "default_model_lib", "local_id": "default_local_id", "conv_template": "codellama_completion", "temperature": 0.5, "repetition_penalty": 1.0, "top_p": 0.95, "mean_gen_len": 128, "max_gen_len": 512, "shift_fill_factor": 0.3, "tokenizer_files": ["tokenizer.json", "tokenizer.model"], "conv_config": {"prefix_tokens": [], "role_empty_sep": "", "role_msg_sep": "", "seps": [""], "stop_tokens": [2], "offset": 0, "separator_style": 1, "messages": [], "stop_str": "}", "roles": ["Prompt", "Code"], "system": "", "add_bos": true, "name": "codellama_completion"}, "model_category": "llama", "model_name": "default_model_name"}'

        conv_config = ConvConfig(
            system=None,
            roles=None,
            messages=None,
            offset=None,
            separator_style=None,
            seps=None,
            role_msg_sep=None,
            role_empty_sep=None,
            stop_str="}",
            stop_tokens=None,
            add_bos=None,
        )

        chat_config = ChatConfig(
            temperature=0.5,
            repetition_penalty=None,
            top_p=None,
            mean_gen_len=None,
            max_gen_len=None,
            conv_config=conv_config,
        )

        self.cm_under_test.update_chat_config(chat_config)
        self.cm_under_test._load_json_override_func.assert_called_once_with(expected_value.replace('\n', '').replace('\t', ''), True)

    def test_update_config_none_conv_config(self):
        expected_value = '{"model_lib": "default_model_lib", "local_id": "default_local_id", "conv_template": "codellama_completion", "temperature": 0.5, "repetition_penalty": 1.0, "top_p": 0.95, "mean_gen_len": 128, "max_gen_len": 512, "shift_fill_factor": 0.3, "tokenizer_files": ["tokenizer.json", "tokenizer.model"], "conv_config": null, "model_category": "llama", "model_name": "default_model_name"}'

        chat_config = ChatConfig(
            temperature=0.5,
            repetition_penalty=None,
            top_p=None,
            mean_gen_len=None,
            max_gen_len=None,
        )

        self.cm_under_test.update_chat_config(chat_config)
        self.cm_under_test._load_json_override_func.assert_called_once_with(expected_value.replace('\n', '').replace('\t', ''), True)
    