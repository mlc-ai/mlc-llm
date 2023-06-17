import json
import os
from tvm.contrib import cc

def main():
    app_config = json.load(open("MLCChat/app-config.json", "r"))
    target = "iphone"
    artifact_path = os.path.abspath(os.path.join("..", "dist"))

    tar_list = []

    for local_id in app_config["model_libs"]:
        paths = [
            os.path.join(artifact_path, local_id, f"{local_id}-{target}.tar"),
            os.path.join(artifact_path, "prebuilt", "lib", f"{local_id}-{target}.tar")
        ]
        valid_paths = [p for p in paths if os.path.isfile(p)]
        if not valid_paths:
            raise RuntimeError(
                f"Cannot find lib for {local_id} in the following candidate path: {paths}"
            )
        tar_list.append(valid_paths[0])

    cc.create_staticlib(os.path.join("build", "lib", "libmodel_iphone.a"), tar_list)
    print(f"Creating lib from {tar_list}..")


if __name__ == "__main__":
    main()
