from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from ..modeling import BaseGPTQForCausalLM
from ..utils.data_utils import get_dataloader


class BaseTask:
    def __init__(
        self,
        model: Union[BaseGPTQForCausalLM, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        data_name_or_path: str,
        prompt_col_name: str,
        label_col_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.dl = get_dataloader(
            data_name_or_path,
            prompt_col_name=prompt_col_name,
            label_col_name=label_col_name,
            tokenizer=tokenizer,
            **kwargs
        )

        self.device = device
        if not self.device:
            self.device = self.model.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    @abstractmethod
    def _predict(self, batch_data: Dict[str, Any], **kwargs) -> List[Any]:
        pass

    @abstractmethod
    def _parse_labels(self, label_ids: torch.LongTensor) -> List[Any]:
        pass

    @abstractmethod
    def _metric(self, pred: List[Any], label: List[Any]) -> Dict[str, float]:
        pass

    def run(self, **predict_kwargs) -> Dict[str, float]:
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            predictions = []
            labels = []
            for batch_data in self.dl:
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(self.device)
                labels += self._parse_labels(batch_data["labels"])
                predictions += self._predict(batch_data, **predict_kwargs)

        return self._metric(predictions, labels)
