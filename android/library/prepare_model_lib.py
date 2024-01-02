import json
import os
from tvm.contrib import ndk


def main():
    app_config = json.load(open("src/main/assets/app-config.json", "r"))
    artifact_path = os.path.abspath(os.path.join("../..", "dist"))
    tar_list = []

    for model_data in app_config["model_list"]:
        path = os.path.join(artifact_path, model_data["model_lib_path"])
        if not os.path.isfile(path):
            raise RuntimeError(f"Cannot find android library {path}")
        tar_list.append(path)

    ndk.create_staticlib(os.path.join("build", "model_lib", "libmodel_android.a"), tar_list)
    print(f"Creating lib from {tar_list}..")


if __name__ == "__main__":
    main()
