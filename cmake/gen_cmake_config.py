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

    # FlashInfer related
    use_flashInfer = False  # pylint: disable=invalid-name
    if "CUDA" in enabled_backends:
        while True:
            user_input = input(
                "Use FlashInfer? (need CUDA w/ compute capability 80;86;89;90) (y/n): "
            )
            if user_input in ["yes", "Y", "y"]:
                cmake_config_str += "set(USE_FLASHINFER ON)\n"
                cmake_config_str += "set(FLASHINFER_ENABLE_FP8 OFF)\n"
                cmake_config_str += "set(FLASHINFER_ENABLE_BF16 OFF)\n"
                cmake_config_str += "set(FLASHINFER_GEN_GROUP_SIZES 1 4 6 8)\n"
                cmake_config_str += "set(FLASHINFER_GEN_PAGE_SIZES 16)\n"
                cmake_config_str += "set(FLASHINFER_GEN_HEAD_DIMS 128)\n"
                cmake_config_str += "set(FLASHINFER_GEN_KV_LAYOUTS 0 1)\n"
                cmake_config_str += "set(FLASHINFER_GEN_POS_ENCODING_MODES 0 1)\n"
                cmake_config_str += 'set(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS "false")\n'
                cmake_config_str += 'set(FLASHINFER_GEN_CASUALS "false" "true")\n'
                use_flashInfer = True  # pylint: disable=invalid-name
                break
            elif user_input in ["no", "N", "n"]:
                cmake_config_str += "set(USE_FLASHINFER OFF)\n"
                break
            else:
                print(f"Invalid input: {use_flashInfer}. Please input again.")
    else:
        cmake_config_str += "set(USE_FLASHINFER OFF)\n"

    if use_flashInfer:
        while True:
            user_input = input("Enter your CUDA compute capability: ")
            if user_input in ["80", "86", "89", "90", "100", "120"]:
                cmake_config_str += f"set(FLASHINFER_CUDA_ARCHITECTURES {user_input})\n"
                cmake_config_str += f"set(CMAKE_CUDA_ARCHITECTURES {user_input})\n"
                break
            else:
                print(
                    f"Invalid input: {user_input}. FlashInfer requires 80, 86, 89, 90, 100 or 120"
                )

    print("\nWriting the following configuration to config.cmake...")
    print(cmake_config_str)

    with open("config.cmake", "w") as f:
        f.write(cmake_config_str)
