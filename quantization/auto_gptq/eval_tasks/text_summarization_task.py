from typing import Any, Dict, List, Optional

import rouge
from torch import LongTensor
from transformers import GenerationConfig

from ._base import BaseTask
from ._utils.generation_utils import postprocess_generation_ids


class TextSummarizationTask(BaseTask):
    def __init__(
        self,
        model,
        tokenizer,
        data_name_or_path: str,
        prompt_col_name: str,
        label_col_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        kwargs["merge_prompt_label"] = False
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_name_or_path=data_name_or_path,
            prompt_col_name=prompt_col_name,
            label_col_name=label_col_name,
            device=device,
            **kwargs
        )

    def _predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[str]:
        generation_config = kwargs["generation_config"]
        output_ids = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            generation_config=generation_config
        )
        return [
            each[0].lower().strip() for each in postprocess_generation_ids(
                input_ids=batch_data["input_ids"],
                output_ids=output_ids,
                num_return_sequences=generation_config.num_return_sequences,
                tokenizer=self.tokenizer
            )
        ]

    def _parse_labels(self, label_ids: LongTensor) -> List[str]:
        labels = []
        for one_label_ids in label_ids:
            one_label_ids = one_label_ids[(one_label_ids == -100).sum():]
            label = self.tokenizer.decode(one_label_ids).lower().strip()
            labels.append(label)

        return labels

    def _metric(self, pred: List[Any], label: List[Any]) -> Dict[str, Dict[str, float]]:
        metric = rouge.Rouge()
        return metric.get_scores(hyps=pred, refs=label, avg=True)

    def run(self, generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        if not generation_config:
            generation_config = GenerationConfig(
                num_beams=1,
                do_sample=False,
                max_new_tokens=128
            )
        generation_config.num_return_sequences = 1
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        return super().run(generation_config=generation_config)


__all__ = ["TextSummarizationTask"]
