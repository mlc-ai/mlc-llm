# pylint: disable=invalid-name,missing-docstring
from typing import Any, Tuple

import torch
import tvm
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration as hf_Whisper
from transformers import WhisperProcessor
from tvm.relax.frontend.nn import spec

from mlc_chat.model.whisper.whisper_model import (
    WhisperConfig,
    WhisperForConditionalGeneration,
)


def load_whisper_from_hf() -> Tuple[hf_Whisper, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    hf_model = hf_Whisper.from_pretrained("openai/whisper-medium")
    hf_model = hf_model.eval().to("cuda")
    return hf_model, processor


def load_data(processor: WhisperProcessor, test_idx: int) -> Tuple[torch.Tensor, str]:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[test_idx]["audio"]
    text = ds[test_idx]["text"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features.to("cuda")

    return input_features, text


def pipe(model: Any, config: WhisperConfig, input_features) -> torch.Tensor:
    # encode
    encode_output = model["encode"](input_features)

    # decode start token
    input_ids = torch.tensor([[config.decoder_start_token_id]], dtype=torch.int32).to("cuda")
    generated_tokens = [config.decoder_start_token_id]

    while True:
        if len(generated_tokens) == 1:
            outputs, encode_kv_cache = model["decode"](
                input_ids, len(generated_tokens), encode_output
            )
        else:
            outputs = model["prefill"](input_ids, len(generated_tokens), encode_kv_cache)

        outputs_logits = outputs
        next_token_logits = outputs_logits[:, 0, :]

        # suppress tokens
        next_tokens_scores = next_token_logits
        next_tokens_scores[:, config.suppress_tokens] = -float("inf")

        # suppress tokens at begin
        if len(generated_tokens) == 1 + config.forced_decoder_ids[-1][0]:
            next_tokens_scores[:, config.begin_suppress_tokens] = -float("inf")

        # force tokens at sepcific position
        generation_idx = len(generated_tokens)
        current_token = dict(config.forced_decoder_ids).get(generation_idx, None)
        if current_token is not None:
            next_tokens_scores[:, :] = -float("inf")
            next_tokens_scores[:, current_token] = 0

        # argmax
        next_token = torch.argmax(next_tokens_scores, dim=-1)[0]
        input_ids[0][0] = next_token

        generated_tokens.append(next_token)

        # stop when we meet eos_token_id or exceed the maximum length
        if (
            next_token == config.eos_token_id
            or len(generated_tokens) == config.max_target_positions
        ):
            break

    return generated_tokens


def main():
    # Set the device and target
    dev = tvm.cuda()
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
        }
    )

    # load model from transformers
    hf_model, processor = load_whisper_from_hf()

    # Define the model config
    config = WhisperConfig(**hf_model.config.to_dict())
    bsz, encode_input_ndim = 1, 16000 * 30 // 160

    # Define the model
    model = WhisperForConditionalGeneration(config=config)

    mod_spec = {
        "encode": {
            "input_features": spec.Tensor([bsz, config.num_mel_bins, encode_input_ndim], "float32"),
        },
        "decode": {
            "input_ids": spec.Tensor([bsz, "seq_len"], "int32"),
            "total_seq_len": int,
            "encoder_hidden_states": spec.Tensor(
                [bsz, config.max_source_positions, config.d_model], "float32"
            ),
        },
        "prefill": {
            "input_ids": spec.Tensor([bsz, 1], "int32"),
            "total_seq_len": int,
            "cached_encoder_key_value": tuple(
                tuple(
                    spec.Tensor(
                        [
                            1,
                            config.max_source_positions,
                            config.decoder_attention_heads,
                            config.d_model // config.decoder_attention_heads,
                        ],
                        "float32",
                    )
                    for i2 in range(2)
                )
                for i1 in range(config.num_hidden_layers)
            ),
        },
    }

    # Usercase1, export it to TVM's IRModule, use `mod.show()` to print the IRModule
    mod, _ = model.export_tvm(spec=mod_spec)

    # Usercase2, JIT compile a model
    for name, param in model.state_dict().items():
        param.data = hf_model.state_dict()[name]

    model = model.jit(spec=mod_spec, target=target, device="cuda", out_format="torch", debug=True)

    # Test on librispeech_asr_dummy
    input_features, text = load_data(processor, test_idx=0)
    generated_tokens = pipe(model, config, input_features)

    # compare with hf whisper output
    hf_predicted_ids = hf_model.generate(input_features).to("cpu")
    assert torch.equal(torch.tensor([generated_tokens], dtype=torch.long), hf_predicted_ids)

    # decode token ids to text
    output = processor.decode(generated_tokens, skip_special_tokens=True)
    assert (
        output
        == " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
    )

    print("Transcription:\n", output)


if __name__ == "__main__":
    main()
