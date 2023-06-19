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


def save_fp16_model(dir_path, model, tokenizer, manually=False):
  new_name = dir_path.name + "-float16"
  out_path = dir_path.parent.joinpath(new_name)

  if manually:
    # Manual saving
    output_model_file = Path.joinpath(out_path, WEIGHTS_NAME)
    output_config_file = Path.joinpath(out_path, CONFIG_NAME)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(out_path)
  else:
    # Use transformer API
    model.save_pretrained(out_path, from_pt=True)


def main(args):
  model_root_dir = Path(args.model_path)

  # Load original model (bfloat16)
  model, tokenizer = load_bf16_model(model_root_dir, args.tokenizer)
  # Convert data type to float 16
  model.to(dtype=torch.float16)
  # Save converted model
  save_fp16_model(model_root_dir, model, tokenizer)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model_path', type=str, default="../../../dist/models/mpt-7b-instruct",
                      help="The path to directory with bfloat16 model")
  parser.add_argument('-t', '--tokenizer', type=str, default="EleutherAI/gpt-neox-20b",
                      help="Tag for transformers to upload correct tokenizer")

  args = parser.parse_args()
  main(args)
