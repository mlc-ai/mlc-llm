import json
import os
import sys
from tvm.contrib import cc


def get_model_libs(lib_path):
    global_symbol_map = cc.get_global_symbol_section_map(lib_path)
    libs = []
    suffix = "___tvm_dev_mblob"
    for name in global_symbol_map.keys():
        if name.endswith(suffix):
            model_lib = name[: -len(suffix)]
            if model_lib.startswith("_"):
                model_lib = model_lib[1:]
            libs.append(model_lib)
    return libs


def main():
    app_config_path = "MLCChat/app-config.json"
    app_config = json.load(open(app_config_path, "r"))
    artifact_path = os.path.abspath(os.path.join("..", "dist"))

    tar_list = []
    model_set = set()

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
        model_set.add(model)

    lib_path = os.path.join("build", "lib", "libmodel_iphone.a")

    cc.create_staticlib(lib_path, tar_list)
    available_model_libs = get_model_libs(lib_path)
    print(f"Creating lib from {tar_list}..")
    print(f"Validating the library {lib_path}...")
    print(
        f"List of available model libs packaged: {available_model_libs},"
        " if we have '-' in the model_lib string, it will be turned into '_'"
    )
    global_symbol_map = cc.get_global_symbol_section_map(lib_path)
    error_happened = False
    for item in app_config["model_list"]:
        model_lib = item["model_lib"]
        model_id = item["model_id"]
        if model_lib not in model_set:
            print(
                f"ValidationError: model_lib={model_lib} specified for model_id={model_id} "
                "is not included in model_lib_path_for_prepare_libs field, "
                "This will cause the specific model not being able to load, "
                f"please check {app_config_path}."
            )
            error_happened = True

        model_prefix_pattern = model_lib.replace("-", "_") + "___tvm_dev_mblob"
        if (
            model_prefix_pattern not in global_symbol_map
            and "_" + model_prefix_pattern not in global_symbol_map
        ):
            model_lib_path = app_config["model_lib_path_for_prepare_libs"][model_lib]
            print(
                "ValidationError:\n"
                f"\tmodel_lib {model_lib} requested in {app_config_path} is not found in {lib_path}\n"
                f"\tspecifically the model_lib for {model_lib_path} in model_lib_path_for_prepare_libs.\n"
                f"\tcurrent available model_libs in {lib_path}: {available_model_libs}"
            )
            error_happened = True

    if not error_happened:
        print("Validation pass")
    else:
        print("Validation failed")
        exit(255)


if __name__ == "__main__":
    main()
