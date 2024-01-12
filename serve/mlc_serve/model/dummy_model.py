from typing import Union, List

from mlc_serve.engine import (
    ChatMessage,
    RequestState,
    RequestId,
    get_engine_config
)
from mlc_serve.model.base import ModelArtifactConfig
from mlc_serve.engine.model_module import (
    DecodeRequest,
    KVCache,
    PrefillRequest,
    SequenceId,
    TextGenerationResult,
)

class DummyTokenizer:
    is_fast = True
    @property
    def eos_token_id(self):
        return 1000

    def encode(self, text: str, **kwargs) -> list[int]:
        return [1] * len(text.split())

    def decode(self, tokens: list[int], **kwargs) -> str:
        return "test " * len(tokens)

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        r = []
        for i in token_ids:
            r.append(f"_test{i}")
        return r

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ret = ""
        for a in tokens:
            ret += a + " "
        return ret


class DummyConversationTemplate:
    def apply(self, messages: list[ChatMessage]) -> str:
        return " ".join(m.content for m in messages if m.content is not None)


class DummyCache:
    def __init__(self, max_cached_tokens: int):
        self.max_cached_tokens = max_cached_tokens
        self.cached_requests = dict[SequenceId, int]()


class DummyCacheManager:
    def __init__(self, max_cached_tokens: int):
        self.cache = DummyCache(max_cached_tokens)

    def get_cache(self) -> KVCache:
        return self.cache

    def allocate(self, request_id: RequestId, num_tokens: int, num_sequences: int) -> bool:
        seq_id = SequenceId(request_id, 0)
        self.cache.cached_requests[seq_id] = num_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")
        return True

    def extend(self, sequence_id: SequenceId, new_tokens: int) -> bool:
        if sequence_id.sequence_index > 0:
            raise RuntimeError("Multiple generated sequences not supported")
        self.cache.cached_requests[sequence_id] += new_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")
        return True

    def free(self, sequence_id: SequenceId):
        if sequence_id.sequence_index > 0:
            raise RuntimeError("Multiple generated sequences not supported")
        del self.cache.cached_requests[sequence_id]

    def free_request(self, state: RequestState):
        for gen_seq in state.generation_sequences:
            self.free(gen_seq.seq_id)

    def get_kv_cache_size(self) -> int:
        return self.cache.max_cached_tokens

    def get_free_space(self) -> int:
        return self.cache.max_cached_tokens - sum(self.cache.cached_requests.values())

    def get_max_new_tokens(self) -> int:
        if not self.cache.cached_requests:
            return self.get_kv_cache_size()
        return self.get_free_space() // len(self.cache.cached_requests)


class DummyTextGenerator:
    def generate(
        self,
        requests: list[Union[PrefillRequest, DecodeRequest]],
        kv_cache: DummyCache,
    ) -> list[TextGenerationResult]:
        result = []
        for req in requests:
            if isinstance(req, DecodeRequest):
                seq_id = req.sequence_id
                request_id = req.sequence_id.request_id
                if req.sequence_id.sequence_index > 0:
                    raise RuntimeError("Multiple generated sequences not supported")
            else:
                seq_id = SequenceId(req.request_id, 0)
                request_id = req.request_id

            if len(req.token_ids) > kv_cache.cached_requests[seq_id]:
                raise RuntimeError(f"Cache out of space for request {request_id}")

            result.append(
                TextGenerationResult(
                    sequence_id=SequenceId(
                        request_id=request_id,
                        sequence_index=0,
                    ),
                    generated_tokens=[req.token_ids[-1] + 1],
                    # generated_tokens=[1],
                    error=None,
                )
            )
        return result


class DummyModelModule:
    def __init__(self, max_cached_tokens: int, max_input_len = 512, max_num_sequences = 8):
        self.tokenizer = DummyTokenizer()
        self.conversation_template = DummyConversationTemplate()
        self.text_generator = DummyTextGenerator()
        self.cache_manager = DummyCacheManager(max_cached_tokens)
        self.model_artifact_config = ModelArtifactConfig._from_json({
            "max_context_length": 1024,
        })
        self.engine_config = get_engine_config({
            "max_decode_steps": 2,
            "min_decode_steps": 1,
            "use_staging_engine" : False,
            "max_input_len": max_input_len,
            "max_num_sequences": max_num_sequences
        })


class DummyTokenizerModule:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.conversation_template = DummyConversationTemplate()
