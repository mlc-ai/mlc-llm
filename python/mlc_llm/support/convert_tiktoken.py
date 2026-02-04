"""
Adapted from https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
Generator of mlc-chat-config.json and tokenizer configuration.
"""

# pylint: disable=import-error
# isort: off
import json
import os
from typing import Dict, List, Optional


def bpe(
    mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None
) -> List[bytes]:
    """Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960"""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def generate_vocab_and_merges(encoder, mergeable_ranks):
    """Generate vocab and merges in huggingface tokenizers format"""

    from transformers.models.gpt2.tokenization_gpt2 import (  # pylint: disable=import-outside-toplevel
        bytes_to_unicode,
    )

    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(b):
        """Convert a token from bytes to a string"""
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(merged) == 2

        merges.append(" ".join(map(token_bytes_to_string, merged)))

    # Also add special tokens
    vocab.update(encoder._special_tokens)  # pylint: disable=protected-access

    return vocab, merges


def convert_tiktoken(model_path, output_dir, context_window_size=None):
    """Convert tiktoken tokenizers to huggingface tokenizers style"""
    try:
        from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(  # pylint: disable=raise-missing-from
            'Converting tiktoken tokenizer requires the "transformers" package.'
            'Please install the "transformers" package to convert toktoken tokenizer'
        )

    tiktoken_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = tiktoken_tokenizer.tokenizer

    vocab, merges = generate_vocab_and_merges(encoder, tiktoken_tokenizer.get_vocab())

    added_tokens = [
        {
            "id": id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, id in encoder._special_tokens.items()  # pylint: disable=protected-access
    ]

    tokenizer_template = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    tokenizer_config_template = {
        "add_prefix_space": False,
        "bos_token": "<|endoftext|>",
        "clean_up_tokenization_spaces": True,
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }

    tokenizer_name = type(tiktoken_tokenizer).__name__

    tokenizer_config_template["tokenizer_class"] = tokenizer_name
    if context_window_size:
        tokenizer_config_template["model_max_length"] = context_window_size
    tokenizer_config_template = dict(sorted(tokenizer_config_template.items(), key=lambda x: x[0]))

    os.makedirs(output_dir, exist_ok=True)

    # Save to files
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as fp:
        json.dump(vocab, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as fp:
        json.dump(tokenizer_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as fp:
        json.dump(tokenizer_config_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as fp:
        json.dump(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as fp:
        fp.write("#version: 0.2\n")
        fp.write("\n".join(merges))
