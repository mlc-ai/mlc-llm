from collections import namedtuple

Backend = namedtuple("Backend", ["name", "cmake_config_name", "prompt_str"])

if __name__ == "__main__":
    tvm_home = ""

    tvm_home = input(
        "Enter TVM_HOME in absolute path. If not specified, 3rdparty/tvm will be used by default: "
    )
    if len(tvm_home) == 0:
        tvm_home = "3rdparty/tvm"

    cmake_config_str = "set(TVM_HOME {})\n".format(tvm_home)
    cmake_config_str += "set(CMAKE_BUILD_TYPE RelWithDebInfo)\n"
    backends = [
        Backend("CUDA", "USE_CUDA", "Use CUDA? (y/n): "),
        Backend("CUTLASS", "USE_CUTLASS", "Use CUTLASS? (y/n): "),
        Backend("CUBLAS", "USE_CUBLAS", "Use CUBLAS? (y/n): "),
        Backend("ROCm", "USE_ROCM", "Use ROCm? (y/n): "),
        Backend("Vulkan", "USE_VULKAN", "Use Vulkan? (y/n): "),
        Backend(
            "Metal",
            "USE_METAL",
            "Use Metal (Apple M1/M2 GPU) ? (y/n): ",
        ),
        Backend(
            "OpenCL",
            "USE_OPENCL",
            "Use OpenCL? (y/n) ",
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

    print("\nWriting the following configuration to config.cmake...")
    print(cmake_config_str)

    with open("config.cmake", "w") as f:
        f.write(cmake_config_str)
