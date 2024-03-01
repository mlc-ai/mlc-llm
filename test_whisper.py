# pylint: disable=invalid-name,missing-docstring
import json
import time
from typing import Any, Tuple

import numpy as np
import torch
import tvm
from datasets import load_dataset
from scipy.io import wavfile
from transformers import WhisperForConditionalGeneration as hf_Whisper
from transformers import WhisperProcessor
from tvm import relax

from mlc_chat.model.whisper.whisper_model import (
    WhisperConfig,
    WhisperForConditionalGeneration,
)


def load_params(artifact_path: str, device, param_names):
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(artifact_path, device)
    plist = []
    size = meta["ParamSize"]
    for pname in param_names:
        plist.append(params[pname])
    return plist


def get_param_list(vm):
    x = vm["_metadata"]()
    metadata = eval(x)
    params = metadata["params"]
    param_names = []
    for param in params:
        param_names.append(param["name"])
    return param_names


def get_param_list000(config, dtype, submodel_name):
    model = WhisperForConditionalGeneration(config)
    # model.to(dtype)
    encode_params = []
    decode_params = []
    prefill_params = []
    for param, _ in model.named_parameters():
        if param.startswith("model.encoder"):
            encode_params.append(param)
        else:
            encode_params.append(param)
            decode_params.append(param)
            if "encoder_attn_k_proj" in param:
                continue
            if "encoder_attn_v_proj" in param:
                continue
            prefill_params.append(param)

    return encode_params, decode_params, prefill_params


def load_whisper_from_hf(model_name="openai/whisper-medium") -> Tuple[hf_Whisper, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(model_name)
    # hf_model = hf_Whisper.from_pretrained(model_name)
    # hf_model = hf_model.eval().to("cuda")
    hf_model = None
    return hf_model, processor


def load_data(processor: WhisperProcessor, test_idx: int) -> Tuple[torch.Tensor, str]:
    samplerate, data = wavfile.read("../librispeech_dummy.wav")
    # print(samplerate)
    # print(len(data))
    input_features = processor(
        data, sampling_rate=samplerate, return_tensors="pt"
    ).input_features.to("cuda")
    # print("input_features: ", input_features.shape)
    return input_features, None


def pipe(
    model: Any, const_params: Any, config: WhisperConfig, input_features, device
) -> torch.Tensor:
    kv_caches = model["_initialize_effect"]()
    # encode
    encode_start = time.time()
    encode_output, kv_caches = model["encode"](input_features, kv_caches, const_params)
    device.sync()
    encode_end = time.time()
    print(f"encode {(encode_end - encode_start) * 1000:.2f}")

    # decode start token
    input_ids = torch.tensor([[config.decoder_start_token_id]], dtype=torch.int32)  # .to("cuda")
    generated_tokens = [config.decoder_start_token_id]

    input_ids = tvm.nd.array(input_ids, tvm.cuda())
    while True:
        decode_start = time.time()
        if len(generated_tokens) == 1:
            (outputs, encode_kv_cache), kv_caches = model["decode"](
                input_ids,
                tvm.runtime.ShapeTuple([len(generated_tokens)]),
                encode_output,
                kv_caches,
                const_params,
            )
        else:
            outputs, kv_caches = model["prefill"](
                input_ids,
                tvm.runtime.ShapeTuple([len(generated_tokens)]),
                encode_kv_cache,
                kv_caches,
                const_params,
            )
        device.sync()
        decode_end = time.time()
        print(f"{(decode_end - decode_start) * 1000:.2f}")

        outputs_logits = outputs.numpy()
        next_token_logits = outputs_logits[:, 0, :]

        # suppress tokens
        next_tokens_scores = next_token_logits
        next_tokens_scores[:, config.suppress_tokens] = -float("inf")

        # suppress tokens at begin
        if len(generated_tokens) == 1:  # + config.forced_decoder_ids[-1][0]:
            next_tokens_scores[:, config.begin_suppress_tokens] = -float("inf")

        # force tokens at sepcific position
        generation_idx = len(generated_tokens)
        current_token = dict(config.forced_decoder_ids).get(generation_idx, None)
        if current_token is not None:
            next_tokens_scores[:, :] = -float("inf")
            next_tokens_scores[:, current_token] = 0

        # argmax
        next_token = np.argmax(next_tokens_scores, axis=-1)[0]
        # next_token = torch.argmax(next_tokens_scores, dim=-1)[0]
        # input_ids[0][0] = next_token
        input_ids = tvm.nd.array(np.array([[next_token]], dtype=np.int32), tvm.cuda())

        generated_tokens.append(next_token)

        # stop when we meet eos_token_id or exceed the maximum length
        if (
            next_token == config.eos_token_id
            or len(generated_tokens) == config.max_target_positions
        ):
            break

    return generated_tokens


def main():
    with open("../models/whisper-large-v3/config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config = WhisperConfig.from_dict(config)

    model_dir = "dist/whisper-q0f16"
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
    ex = tvm.runtime.load_module(f"{model_dir}/model.so")
    vm = relax.VirtualMachine(ex, dev)
    param_list = get_param_list(vm)
    const_params = load_params(f"{model_dir}/params", dev, param_list)

    # load model from transformers
    hf_model, processor = load_whisper_from_hf("../models/whisper-large-v3/")

    # Test on librispeech_asr_dummy
    input_features, text = load_data(processor, test_idx=0)
    input_features = input_features.to("cpu").to(torch.float16)
    input_features = tvm.nd.array(input_features, device=dev)

    # const_params = [encode_params, decode_params, prefill_params]
    generated_tokens = pipe(vm, const_params, config, input_features, dev)

    # # compare with hf whisper output
    # hf_predicted_ids = hf_model.generate(input_features).to("cpu")
    # assert torch.equal(torch.tensor([generated_tokens], dtype=torch.long), hf_predicted_ids)

    # decode token ids to text
    output = processor.decode(generated_tokens, skip_special_tokens=True)
    assert (
        output
        == " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
    )

    print("Transcription:\n", output)


if __name__ == "__main__":
    main()
