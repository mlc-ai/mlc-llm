# The procedure is based on https://huggingface.co/transformers/v1.2.0/serialization.html#serialization-best-practices
from pathlib import Path
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME


def load_bf16_model(dir_path, tokenizer_name):
  model = AutoModelForCausalLM.from_pretrained(
    dir_path,
    trust_remote_code=True
  )
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  return model, tokenizer


def save_fp16_model(dir_path, model, tokenizer):
  model_to_save = model.module if hasattr(model, 'module') else model

  output_model_file = Path.joinpath(dir_path, WEIGHTS_NAME)
  output_config_file = Path.joinpath(dir_path, CONFIG_NAME)

  torch.save(model_to_save.state_dict(), output_model_file)
  model_to_save.config.to_json_file(output_config_file)
  tokenizer.save_vocabulary(dir_path)


def main(args):
  model_root_dir = Path(args.model_path)
  new_name = model_root_dir.name + "-float16"
  out_path = model_root_dir.parent.joinpath(new_name)

  model, tokenizer = load_bf16_model(model_root_dir, args.tokenizer)
  model.to(dtype=torch.float16)
  model.save_pretrained(out_path, from_pt=True)
  # save_fp16_model(out_path, model, tokenizer)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model_path', type=str, default="../../../dist/models/mpt-7b-instruct",
                      help="The path to directory with bfloat16 model")
  parser.add_argument('-t', '--tokenizer', type=str, default="EleutherAI/gpt-neox-20b",
                      help="Tag for transformers to upload correct tokenizer")

  args = parser.parse_args()
  main(args)
