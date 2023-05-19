import copy
import random
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict, IterableDatasetDict
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def make_data_block(
    samples: Dict[str, List[str]],
    prompt_col_name: str,
    label_col_name: str,
    tokenizer: PreTrainedTokenizer,
    preprocess_fn: Optional[Callable] = None,
    sample_max_len: int = 1024,
    block_max_len: int = 2048,
    add_eos_token: bool = False,
    truncate_prompt: bool = True,
    merge_prompt_label: bool = False
) -> Dict[str, List[LongTensor]]:
    """A simple implementation of text generation oriented smart batching to maximize VRAM usage when evaluation

    :param samples: Dict[str, List[str]], samples that used to make data blocks
    :param prompt_col_name: str, name of the key in samples whose value stores prompt
    :param label_col_name: str, name of the key in samples whose value stores label
    :param tokenizer: transformers.PretrainedTokenizer, tokenizer that used to tokenize samples
    :param preprocess_fn: Optional[Callable], optional function that used to preprocess samples such as
        refactor the data structure of samples, note the output of this function must be a dict whose keys
        at least contains `prompt_col_name` and `label_col_name`
    :param sample_max_len: int, defaults to 1024, max tokens number of each sample (before padding)
    :param block_max_len: int, defaults to 2048, max tokens number of each data block (after padding)
    :param add_eos_token: bool, defaults to False, whether add eos_token or not to the label
    :param truncate_prompt: bool, defaults to True, whether to truncate prompt if the sample's total tokens
        number exceeds `sample_max_len`, if not, will truncate label and drop this sample when all tokens
        in label are truncated
    :param merge_prompt_label: bool, defaults to False, will merge label into prompt if set to True, usually
        this only required when doing language modeling task
    :return: Dict[str, List[torch.LongTensor]], a dict whose keys are `input_ids`, `attention_mask` and
        `label` and values are a list of torch.LongTensor
    """
    if preprocess_fn:
        samples = preprocess_fn(samples)

    prompts = samples[prompt_col_name]
    labels = samples[label_col_name]

    # tokenize samples
    tokenized_prompts = tokenizer(prompts, truncation=False)["input_ids"]
    tokenized_labels = tokenizer(labels, truncation=False)["input_ids"]

    # filter tokenized samples by length
    dropped_indices = []
    for idx, (tokenized_prompt, tokenized_label) in enumerate(zip(tokenized_prompts, tokenized_labels)):
        if add_eos_token:
            tokenized_label += [tokenizer.eos_token_id]
        len_prompt = len(tokenized_prompt)
        len_label = len(tokenized_label)
        exceed_len = len_prompt + len_label - sample_max_len
        if exceed_len > 0:
            if truncate_prompt:
                tokenized_prompt = tokenized_prompt[exceed_len:]
            else:
                tokenized_label = tokenized_label[: -exceed_len]
        tokenized_prompts[idx] = tokenized_prompt
        tokenized_labels[idx] = tokenized_label
        if not tokenized_label:
            dropped_indices.append(idx)

    # make data blocks of samples
    tokenized_samples = sorted(
        [
            (p, l) for idx, (p, l) in enumerate(zip(tokenized_prompts, tokenized_labels))
            if idx not in dropped_indices
        ],
        key=lambda x: (len(x[0]) + len(x[1])) if merge_prompt_label else len(x[0])
    )
    sample_blocks = []
    sample_block = []
    blk_max_len = 0
    blk_total_len = 0
    for tokenized_sample in tokenized_samples:
        prompt_ids, label_ids = tokenized_sample
        ori_sample_len = len(prompt_ids)
        if merge_prompt_label:
            ori_sample_len += len(label_ids)
        if ori_sample_len <= blk_max_len:
            additional_len = blk_max_len
            sample_len = blk_max_len
        else:
            additional_len = len(sample_block) * (ori_sample_len - blk_max_len) + ori_sample_len
            sample_len = ori_sample_len

        if blk_total_len + additional_len > block_max_len:
            sample_blocks.append((copy.copy(sample_block), blk_max_len))
            sample_block = []
            blk_max_len = 0
            blk_total_len = 0
            sample_len = ori_sample_len
            additional_len = ori_sample_len

        sample_block.append(tokenized_sample)
        blk_max_len = max(blk_max_len, sample_len)
        blk_total_len += additional_len

    if sample_block:
        sample_blocks.append((copy.copy(sample_block), blk_max_len))
    del sample_block
    del blk_max_len
    del blk_total_len

    new_samples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    # padding each data block internally
    for block, blk_max_len in sample_blocks:
        input_ids = []
        attention_mask = []
        label_ids = []
        label_max_len = max([len(sample[1]) for sample in block])

        for sample in block:
            tokenized_prompt, tokenized_label = sample
            sample_len = len(tokenized_prompt)
            if merge_prompt_label:
                sample_len += len(tokenized_label)
            pad_num = blk_max_len - sample_len
            if merge_prompt_label:
                input_ids.append([tokenizer.pad_token_id] * pad_num + tokenized_prompt + tokenized_label)
                label_ids.append([-100] * (pad_num + len(tokenized_prompt)) + tokenized_label)
            else:
                input_ids.append([tokenizer.pad_token_id] * pad_num + tokenized_prompt)
                label_ids.append([-100] * (label_max_len - len(tokenized_label)) + tokenized_label)
            attention_mask.append([0] * pad_num + [1] * sample_len)

        new_samples["input_ids"].append(input_ids)
        new_samples["attention_mask"].append(attention_mask)
        new_samples["labels"].append(label_ids)

    return new_samples


def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [LongTensor(block["labels"]) for block in blocks]

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    return {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long()
    }


def get_dataloader(
    data_path_or_name: str,
    prompt_col_name: str,
    label_col_name: str,
    tokenizer: PreTrainedTokenizer,
    load_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
    num_samples: int = 128,
    sample_max_len: int = 1024,
    block_max_len: int = 2048,
    add_eos_token: bool = False,
    truncate_prompt: bool = True,
    merge_prompt_label: bool = False,
    load_fn_kwargs: Optional[dict] = None,
    preprocess_fn_kwargs: Optional[dict] = None,
    **kwargs
) -> DataLoader:
    """load dataset and build dataloader

    :param data_path_or_name: str, dataset name in hf-hub or local file path
    :param prompt_col_name: str, see `make_data_block`
    :param label_col_name: str, see `make_data_block`
    :param tokenizer: str, see `make_data_block`
    :param load_fn: Optional[Callable], defaults to None, function used to load dataset, if not specified,
        use `datasets.load_dataset`
    :param preprocess_fn: Optional[Callable], see `make_data_block`
    :param num_samples: int, defaults to 128, total samples used to evaluation
    :param sample_max_len: int, see `make_data_block`
    :param block_max_len: int, see `make_data_block`
    :param add_eos_token: bool, see `make_data_block`
    :param truncate_prompt: bool, see `make_data_block`
    :param merge_prompt_label: bool, see `make_data_block`
    :param load_fn_kwargs: Optional[dict], defaults to None, keyword arguments used
        for `load_fn` or `datasets.load_dataset`
    :param preprocess_fn_kwargs: Optional[dict], defaults to None, keyword arguments used
        for `preprocess_fn`
    :param kwargs: additional keyword arguments will be passed to torch's `DataLoader` initialization,
        note values of `batch_size`, `shuffle` and `collate_fn` will always be overridden to fixed value
    :return: torch.utils.data.DataLoader
    """

    if not load_fn_kwargs:
        load_fn_kwargs = dict()
    if not preprocess_fn_kwargs:
        preprocess_fn_kwargs = dict()

    if load_fn:
        ds = load_fn(data_path_or_name, **load_fn_kwargs)
    else:
        ds = load_dataset(data_path_or_name, **load_fn_kwargs)
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        if "evaluation" in ds:
            ds = ds["evaluation"]
        elif "test" in ds:
            ds = ds["test"]
        else:
            ds = ds["train"]

    ds = ds.select(indices=random.sample(range(len(ds)), min(len(ds), num_samples)), keep_in_memory=True)
    ds = ds.map(
        make_data_block,
        batched=True,
        batch_size=len(ds),
        num_proc=1,
        remove_columns=ds.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": prompt_col_name,
            "label_col_name": label_col_name,
            "tokenizer": tokenizer,
            "preprocess_fn": partial(preprocess_fn, **preprocess_fn_kwargs),
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": truncate_prompt,
            "merge_prompt_label": merge_prompt_label
        }
    )

    # override some arguments' values in kwargs despite user specified
    kwargs["batch_size"] = 1
    kwargs["shuffle"] = False
    kwargs["collate_fn"] = partial(collate_data, pad_token_id=tokenizer.pad_token_id)
    dl = DataLoader(ds, **kwargs)

    return dl


__all__ = ["make_data_block", "collate_data", "get_dataloader"]
