from transformers import AutoTokenizer
import os
from ..engine import ChatMessage

class Tokenizer:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.eos_token_id = self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class ConversationTemplate:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def apply(self, messages: list[ChatMessage]) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


class HfTokenizerModule:
    def __init__(self, model_artifact_path: str):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_artifact_path, "model"),
            trust_remote_code=False,
        )
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.conversation_template = ConversationTemplate(hf_tokenizer)
