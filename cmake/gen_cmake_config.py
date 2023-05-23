from collections import namedtuple

Backend = namedtuple("Backend", ["name", "cmake_config_name", "prompt_str"])

if __name__ == "__main__":
    cmake_config_str = "set(CMAKE_BUILD_TYPE RelWithDebInfo)\n"
    backends = [
        Backend("CUDA", "USE_CUDA", "Use CUDA? (y/n): "),
        Backend("Vulkan", "USE_VULKAN", "Use Vulkan? (y/n): "),
        Backend(
            "Metal",
            "USE_METAL",
            "Use Metal (Apple M1/M2 GPU) ? (y/n): ",
        ),
    ]

    for backend in backends:
        while True:
            use_backend = input(backend.prompt_str)
            if use_backend in ["yes", "Y", "y"]:
                cmake_config_str += "set({} ON)\n".format(backend.cmake_config_name)
                break
            elif use_backend in ["no", "N", "n"]:
                cmake_config_str += "set({} OFF)\n".format(backend.cmake_config_name)
                break
            else:
                print("Invalid input: {}. Please input again.".format(use_backend))
    print(cmake_config_str)
    print("Writing configuration to config.cmake...")

    with open("config.cmake", "w") as f:
        f.write(cmake_config_str)
