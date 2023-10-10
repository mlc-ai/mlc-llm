from typing import Dict
import random
from .types import (InferenceEngine, Request, InferenceStepResult, TextGenerationOutput,
                    ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig)

class DummyLLMEngine (InferenceEngine):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.requests: Dict[str, Request] = {}
        self.tokens_left: Dict[str, int] = {}

    def add(self, requests: list[Request]) -> list[str]:
        if len(requests):
            output = []
            for r in requests:
                self.requests[r.request_id] = r
                self.tokens_left[r.request_id] = random.randint(3,6)
                output.append(r.request_id)
        return output

    def cancel(self, request_id: str):
        del self.requests[request_id]
        del self.tokens_left[request_id]

    def step(self) -> InferenceStepResult:
        a = InferenceStepResult([],[])
        if len(self.tokens_left):
            a.outputs = []
            for i  in self.tokens_left:
                self.tokens_left[i] = self.tokens_left[i] - 1
                r = TextGenerationOutput(i, " " + self.requests[i].prompt)
                if self.tokens_left[i] == 0:
                    r.finish_reason = "stop"
                a.outputs.append(r)
        return a
