from typing import List, Optional, Union

from torch import LongTensor
from transformers import PreTrainedTokenizer


def postprocess_generation_ids(
    input_ids: LongTensor,
    output_ids: LongTensor,
    num_return_sequences: int,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    pad_token_ids: Optional[int] = None,
) -> List[List[Union[str, List[int]]]]:
    outputs = []
    for idx, start in enumerate(range(0, len(output_ids), num_return_sequences)):
        sub_output_ids = output_ids[start: start + num_return_sequences]
        sub_generated_ids = sub_output_ids[..., input_ids[idx].size(0):]
        if tokenizer:
            outputs.append(
                [
                    generated_text for generated_text in tokenizer.batch_decode(
                        sub_generated_ids,
                        clean_up_tokenization_spaces=True
                    )
                ]
            )
        else:
            sub_generated_ids = sub_output_ids.cpu().numpy().tolist()
            for i, one_sub_generated_ids in enumerate(sub_generated_ids):
                if pad_token_ids is not None and pad_token_ids in one_sub_generated_ids:
                    one_sub_generated_ids = one_sub_generated_ids[: one_sub_generated_ids.index(pad_token_ids)]
                sub_generated_ids[i] = one_sub_generated_ids
            outputs.append(sub_generated_ids)

    return outputs


__all__ = ["postprocess_generation_ids"]
