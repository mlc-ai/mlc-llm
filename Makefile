#for dylan with broken python :( 
PY:=python3

.PHONY: init
init:
# git submodule update --init --recursive
#install rust, i had issues with brew version - script from site/rustup worked
	mkdir -p build
	cd build && $(PY) ../cmake/gen_cmake_config.py
#we can just make our own gen cfg script
#set(TVM_SOURCE_DIR 3rdparty/tvm)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(USE_CUDA OFF)
#set(USE_CUTLASS OFF)
#set(USE_CUBLAS OFF)
#set(USE_ROCM OFF)
#set(USE_VULKAN OFF)
#set(USE_METAL ON)
#set(USE_OPENCL OFF)
#set(USE_OPENCL_ENABLE_HOST_PTR OFF)
#set(USE_FLASHINFER OFF)


.PHONY: shared_lib
shared_lib:
	mkdir -p build
	cd build && cmake .. && cmake --build . --parallel $(nproc)