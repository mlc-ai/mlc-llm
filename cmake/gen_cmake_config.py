from collections import namedtuple

Backend = namedtuple("Backend", ["name", "cmake_config_name", "prompt_str", "parent"])

if __name__ == "__main__":
    tvm_home = ""  # pylint: disable=invalid-name

    tvm_home = input(
        "Enter TVM_SOURCE_DIR in absolute path. If not specified, 3rdparty/tvm will be used by default: "
    )
    if len(tvm_home) == 0:
        tvm_home = "3rdparty/tvm"  # pylint: disable=invalid-name

    cmake_config_str = f"set(TVM_SOURCE_DIR {tvm_home})\n"
    cmake_config_str += "set(CMAKE_BUILD_TYPE RelWithDebInfo)\n"
    cuda_backend = Backend("CUDA", "USE_CUDA", "Use CUDA? (y/n): ", None)
    opencl_backend = Backend("OpenCL", "USE_OPENCL", "Use OpenCL? (y/n) ", None)
    backends = [
        cuda_backend,
        Backend("CUTLASS", "USE_CUTLASS", "Use CUTLASS? (y/n): ", cuda_backend),
        Backend("CUBLAS", "USE_CUBLAS", "Use CUBLAS? (y/n): ", cuda_backend),
        Backend("ROCm", "USE_ROCM", "Use ROCm? (y/n): ", None),
        Backend("Vulkan", "USE_VULKAN", "Use Vulkan? (y/n): ", None),
        Backend("Metal", "USE_METAL", "Use Metal (Apple M1/M2 GPU) ? (y/n): ", None),
        opencl_backend,
        Backend(
            "OpenCLHostPtr",
            "USE_OPENCL_ENABLE_HOST_PTR",
            "Use OpenCLHostPtr? (y/n): ",
            opencl_backend,
        ),
    ]

    enabled_backends = set()

    for backend in backends:
        if backend.parent is not None and backend.parent.name not in enabled_backends:
            cmake_config_str += f"set({backend.cmake_config_name} OFF)\n"
        else:
            while True:
                use_backend = input(backend.prompt_str)
                if use_backend in ["yes", "Y", "y"]:
                    cmake_config_str += f"set({backend.cmake_config_name} ON)\n"
                    enabled_backends.add(backend.name)
                    break
                elif use_backend in ["no", "N", "n"]:
                    cmake_config_str += f"set({backend.cmake_config_name} OFF)\n"
                    break
                else:
                    print(f"Invalid input: {use_backend}. Please input again.")

    if "CUDA" in enabled_backends:
        cmake_config_str += f"set(USE_THRUST ON)\n"

    print("\nWriting the following configuration to config.cmake...")
    print(cmake_config_str)

    with open("config.cmake", "w") as f:
        f.write(cmake_config_str)
