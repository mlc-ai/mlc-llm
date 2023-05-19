from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from torch import LongTensor
from transformers import PreTrainedTokenizer, GenerationConfig

from ._base import BaseTask
from ._utils.generation_utils import postprocess_generation_ids
from ._utils.classification_utils import get_closest_label


def get_predictions(
    input_ids: LongTensor,
    output_ids: LongTensor,
    num_return_sequences: int,
    tokenizer: PreTrainedTokenizer,
    classes: List[str]
) -> List[int]:
    predictions = []
    generated_texts = postprocess_generation_ids(
        input_ids=input_ids,
        output_ids=output_ids,
        num_return_sequences=num_return_sequences,
        tokenizer=tokenizer
    )
    for sub_generated_texts in generated_texts:
        sub_predictions = []
        for gen_text in sub_generated_texts:
            sub_predictions.append(get_closest_label(gen_text.lower().strip(), classes))
        predictions.append(Counter(sub_predictions).most_common(1)[0][0])
    return predictions


class SequenceClassificationTask(BaseTask):
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        classes: List[str],
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
        self.classes = [each.lower().strip() for each in classes]
        classes_ids = self.tokenizer(classes)
        self.max_new_tokens = max([len(each) for each in classes_ids])

    def _predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[int]:
        generation_config = kwargs["generation_config"]
        output_ids = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            generation_config=generation_config
        )
        return get_predictions(
            batch_data["input_ids"],
            output_ids,
            generation_config.num_return_sequences,
            self.tokenizer,
            self.classes
        )

    def _parse_labels(self, label_ids: LongTensor) -> List[int]:
        labels = []
        for one_label_ids in label_ids:
            one_label_ids = one_label_ids[(one_label_ids == -100).sum():]
            label = self.tokenizer.decode(one_label_ids, clean_up_tokenization_spaces=True).lower().strip()
            label = get_closest_label(label, self.classes)
            labels.append(label)

        return labels

    def _metric(self, pred: List[int], label: List[int]) -> Dict[str, float]:
        pred = np.array(pred)
        label = np.array(label)

        acc = (pred == label).mean()

        return {"acc": acc}

    def run(self, generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        if not generation_config:
            generation_config = GenerationConfig(
                num_beams=1,
                do_sample=False,
                num_return_sequences=1
            )
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        return super().run(generation_config=generation_config)


__all__ = ["SequenceClassificationTask"]
