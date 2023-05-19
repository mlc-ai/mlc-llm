import math
from typing import Any, Dict, List, Optional

from torch import LongTensor

from ._base import BaseTask


class LanguageModelingTask(BaseTask):
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
        kwargs["merge_prompt_label"] = True
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_name_or_path=data_name_or_path,
            prompt_col_name=prompt_col_name,
            label_col_name=label_col_name,
            device=device,
            **kwargs
        )

    def _predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[float]:
        outputs = self.model(**batch_data)
        loss = outputs.loss.cpu().item()

        return [loss]

    def _parse_labels(self, label_ids: LongTensor) -> List[Any]:
        return []

    def _metric(self, pred: List[Any], label: List[Any]) -> Dict[str, float]:
        return {"ppl": math.exp(sum(pred) / len(pred))}

    def run(self) -> Dict[str, float]:
        return super().run()


__all__ = ["LanguageModelingTask"]
