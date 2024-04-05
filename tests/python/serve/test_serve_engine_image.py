from mlc_llm.serve import Engine, GenerationConfig, KVCacheConfig, data
from mlc_llm.serve.engine import ModelInfo
from mlc_llm.serve.entrypoints.entrypoint_utils import get_image_from_url


def get_test_image():
    return get_image_from_url("https://llava-vl.github.io/static/images/view.jpg")


def test_engine_generate():
    # Initialize model loading info and KV cache config
    model = ModelInfo(
        "dist/llava-1.5-7b-hf-q4f16_1-MLC/params",
        model_lib_path="dist/llava-1.5-7b-hf-q4f16_1-MLC/llava-1.5-7b-hf-q4f16_1-MLC.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16, max_total_sequence_length=4096)
    # Create engine
    engine = Engine(model, kv_cache_config)

    max_tokens = 256

    prompts = [
        [
            data.TextData("USER: "),
            data.ImageData(get_test_image(), 576),
            data.TextData("\nWhat does this image represent? ASSISTANT:"),
        ],
        [
            data.TextData("USER: "),
            data.ImageData(get_test_image(), 576),
            data.TextData("\nIs there a dog in this image? ASSISTANT:"),
        ],
        [data.TextData("USER: What is the meaning of life? ASSISTANT:")],
    ]

    output_texts, _ = engine.generate(
        prompts, GenerationConfig(max_tokens=max_tokens, stop_token_ids=[2])
    )

    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


if __name__ == "__main__":
    test_engine_generate()
