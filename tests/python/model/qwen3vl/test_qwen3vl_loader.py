import os
import subprocess

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def verify():
    local_model_path = "../mlc-models/Qwen3-VL-2B-Instruct/"
    mlc_model_path = "../mlc-models/mlc-qwen/"
    quantization = "q0f16"
    conv_template = "qwen3_vl"
    
    # ensure output dir exists
    if not os.path.exists(mlc_model_path):
        os.makedirs(mlc_model_path)
    
    # 1. Gen Config
    cmd_gen_config = f"python -m mlc_llm gen_config {local_model_path} --quantization {quantization} --conv-template {conv_template} -o {mlc_model_path}"
    if run_command(cmd_gen_config) != 0:
        print("Gen Config Failed")
        return

    # 2. Convert Weight
    cmd_convert = f"mlc_llm convert_weight {local_model_path} --quantization {quantization} -o {mlc_model_path}"
    if run_command(cmd_convert) != 0:
        print("Convert Weight Failed")
        return

    print("Verification Successful!")

if __name__ == "__main__":
    verify()
