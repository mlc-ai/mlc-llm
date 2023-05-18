import json
import os
from tvm.contrib import cc

def main():
    app_config = json.load(open("app-config.json", "r"))
    target = "iphone"
    artifact_path = os.path.abspath(os.path.join("..", "dist"))

    tar_list = []

    for local_id in app_config["model_libs"]:
        path = os.path.join(artifact_path, local_id, f"{local_id}-{target}.tar")
        if not os.path.isfile(path):
            raise RuntimeError(f"Cannot find {path}")
        tar_list.append(path)

    cc.create_staticlib(os.path.join("build", "lib", "libmodel_iphone.a"), tar_list)
    print(f"Creating lib from {tar_list}..")


if __name__ == "__main__":
    main()
