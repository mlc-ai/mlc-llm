import json
import os
from tvm.contrib import cc


def main():
    app_config = json.load(open("MLCChat/app-config.json", "r"))
    artifact_path = os.path.abspath(os.path.join("..", "dist"))

    tar_list = []

    for model, model_lib_path in app_config["model_lib_path_for_prepare_libs"].items():
        paths = [
            os.path.join(artifact_path, model_lib_path),
            os.path.join(artifact_path, "prebuilt", model_lib_path),
            os.path.join(model_lib_path),
        ]
        valid_paths = [p for p in paths if os.path.isfile(p)]
        if not valid_paths:
            raise RuntimeError(
                f"Cannot find iOS lib for {model} from the following candidate paths: {paths}"
            )
        tar_list.append(valid_paths[0])

    cc.create_staticlib(os.path.join("build", "lib", "libmodel_iphone.a"), tar_list)
    print(f"Creating lib from {tar_list}..")


if __name__ == "__main__":
    main()
