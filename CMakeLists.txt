cmake_minimum_required(VERSION 3.26)
project(mlc_llm C CXX)

include(CheckCXXCompilerFlag)
if(MSVC)
  set(CMAKE_CXX_FLAGS "/fp:fast ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-ffast-math ${CMAKE_CXX_FLAGS}")
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Build type" FORCE)
  message(STATUS "Setting default build type to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

option(MLC_HIDE_PRIVATE_SYMBOLS "Hide private symbols" ON)
option(MLC_LLM_BUILD_PYTHON_MODULE "Build Python module with scikit-build-core"
       OFF)

set(MLC_VISIBILITY_FLAG "")
if(MLC_HIDE_PRIVATE_SYMBOLS)
  set(HIDE_PRIVATE_SYMBOLS ON)
  if(NOT MSVC)
    set(MLC_VISIBILITY_FLAG "-fvisibility=hidden")
  endif()
  message(STATUS "Hide private symbols")
endif()

option(BUILD_CPP_TEST "Build cpp unittests" OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the existing MLC build options when configuring TVM and TIRx.
foreach(_option LLVM CUDA ROCM METAL VULKAN OPENCL OPENCL_ENABLE_HOST_PTR
                OPENCL_EXTN_QCOM OPENMP THREADS CUSTOM_LOGGING)
  if(DEFINED USE_${_option})
    set(TVM_TIRX_USE_${_option} "${USE_${_option}}")
  endif()
endforeach()
if(NOT DEFINED TVM_TIRX_USE_LLVM)
  set(TVM_TIRX_USE_LLVM OFF)
endif()
foreach(_option CUBLAS CUDNN NCCL RCCL NVSHMEM CUTLASS THRUST CURAND NVTX)
  if(DEFINED USE_${_option})
    set(TVM_USE_${_option} "${USE_${_option}}")
  endif()
endforeach()
set(TVM_USE_RPC OFF)
set(TVM_USE_GTEST OFF)
if(NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  else()
    set(TVM_SOURCE_DIR 3rdparty/tvm)
  endif(DEFINED ENV{TVM_SOURCE_DIR})
endif(NOT DEFINED TVM_SOURCE_DIR)
message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
get_directory_property(MLC_LLM_RUNTIME_BACKEND_LIBS
                       DIRECTORY ${TVM_SOURCE_DIR}
                       DEFINITION TVM_RUNTIME_BACKEND_LIBS)
if(MLC_LLM_INSTALL_STATIC_LIB)
  include(cmake/StaticRuntime.cmake)
endif()

set(MLC_LLM_RUNTIME_LINKER_LIB "")
set(TOKENZIER_CPP_PATH 3rdparty/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

set(XGRAMMAR_PATH 3rdparty/xgrammar)
file(GLOB_RECURSE MLC_LLM_SRCS CONFIGURE_DEPENDS cpp/*.cc)
file(GLOB_RECURSE XGRAMMAR_SRCS CONFIGURE_DEPENDS ${XGRAMMAR_PATH}/cpp/*.cc)
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX "${XGRAMMAR_PATH}/cpp/pybind/.*\\.cc")
list(APPEND MLC_LLM_SRCS ${XGRAMMAR_SRCS})
add_library(mlc_llm_objs OBJECT ${MLC_LLM_SRCS})

set(MLC_LLM_INCLUDES ${TVM_SOURCE_DIR}/include)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS} __STDC_FORMAT_MACROS=1)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS} XGRAMMAR_ENABLE_LOG_DEBUG=0)

target_compile_definitions(mlc_llm_objs PRIVATE ${MLC_LLM_COMPILE_DEFS})
target_compile_definitions(mlc_llm_objs PRIVATE -DMLC_LLM_EXPORTS)
target_include_directories(mlc_llm_objs PRIVATE ${MLC_LLM_INCLUDES})
target_include_directories(mlc_llm_objs PRIVATE 3rdparty/stb)
target_include_directories(mlc_llm_objs PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(mlc_llm_objs PRIVATE ${XGRAMMAR_PATH}/include)
# xgrammar still depends on picojson - use its bundled copy
target_include_directories(mlc_llm_objs
                           PRIVATE ${XGRAMMAR_PATH}/3rdparty/picojson)
target_link_libraries(mlc_llm_objs PRIVATE tvm_tirx::header tvm_ffi::header)
target_compile_definitions(mlc_llm_objs PRIVATE
  "$<TARGET_PROPERTY:tvm_runtime,INTERFACE_COMPILE_DEFINITIONS>")
if(TVM_USE_NVTX)
  get_directory_property(MLC_CUDA_INCLUDE_DIRS
                         DIRECTORY ${TVM_SOURCE_DIR}
                         DEFINITION CUDA_INCLUDE_DIRS)
  target_include_directories(mlc_llm_objs PRIVATE ${MLC_CUDA_INCLUDE_DIRS})
  target_compile_definitions(mlc_llm_objs PRIVATE MLC_NVTX_ENABLED=1)
endif()

add_library(mlc_llm SHARED $<TARGET_OBJECTS:mlc_llm_objs>)
add_library(mlc_llm_static STATIC $<TARGET_OBJECTS:mlc_llm_objs>)
add_dependencies(mlc_llm_static tokenizers_cpp sentencepiece-static tokenizers_c)
set_target_properties(mlc_llm_static PROPERTIES OUTPUT_NAME mlc_llm)
if(MLC_LLM_INSTALL_STATIC_LIB)
  add_dependencies(mlc_llm_static mlc_tvm_runtime_static tvm_ffi_static)
  # FFI registers the enum metadata used by the runtime's static initializers.
  target_link_libraries(mlc_llm_static INTERFACE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,tvm_ffi_static,mlc_tvm_runtime_static>")
else()
  add_dependencies(mlc_llm_static tvm_runtime ${MLC_LLM_RUNTIME_BACKEND_LIBS})
endif()

target_link_libraries(mlc_llm PUBLIC tvm_runtime)
target_link_libraries(mlc_llm PRIVATE ${MLC_LLM_RUNTIME_BACKEND_LIBS})
target_link_libraries(mlc_llm PRIVATE tokenizers_cpp)

find_library(FLASH_ATTN_LIBRARY flash_attn
             HINTS ${TVM_SOURCE_DIR}/*/3rdparty/libflash_attn/src)

if(FLASH_ATTN_LIBRARY STREQUAL "FLASH_ATTN_LIBRARY-NOTFOUND")
  message(
    WARNING
      "Cannot find libflash_attn. The model must not have been built with --use-flash-attn-mqa option."
  )
else()
  target_link_libraries(mlc_llm PUBLIC -Wl,--no-as-needed ${FLASH_ATTN_LIBRARY})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(mlc_llm PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(mlc_llm_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(mlc_llm_static PRIVATE "TVM_LOG_DEBUG")
endif()

if(BUILD_CPP_TEST)
  message(STATUS "Building cpp unittests")
  add_subdirectory(3rdparty/googletest)
  file(GLOB_RECURSE MLC_LLM_TEST_SRCS
       ${PROJECT_SOURCE_DIR}/tests/cpp/*unittest.cc)
  add_executable(mlc_llm_cpp_tests ${MLC_LLM_TEST_SRCS})
  target_include_directories(mlc_llm_cpp_tests PRIVATE ${MLC_LLM_INCLUDES})
  target_include_directories(mlc_llm_cpp_tests
                             PRIVATE ${PROJECT_SOURCE_DIR}/cpp
                                     ${TOKENZIER_CPP_PATH}/include
                                     ${XGRAMMAR_PATH}/include)
  target_include_directories(
    mlc_llm_cpp_tests PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(mlc_llm_cpp_tests PUBLIC mlc_llm gtest gtest_main)
endif(BUILD_CPP_TEST)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(mlc_llm PRIVATE log)
  target_link_libraries(tokenizers_cpp PRIVATE log)
endif()

add_library(mlc_llm_module SHARED $<TARGET_OBJECTS:mlc_llm_objs>)
target_link_libraries(mlc_llm_module PUBLIC tvm_runtime)
target_link_libraries(mlc_llm_module PRIVATE ${MLC_LLM_RUNTIME_BACKEND_LIBS})
target_link_libraries(mlc_llm_module PRIVATE tokenizers_cpp)

set_property(
  TARGET mlc_llm_module
  APPEND
  PROPERTY LINK_OPTIONS "${MLC_VISIBILITY_FLAG}")
set_property(
  TARGET mlc_llm
  APPEND
  PROPERTY LINK_OPTIONS "${MLC_VISIBILITY_FLAG}")

find_program(CARGO_EXECUTABLE cargo)

if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR "Cargo is not found! Please install cargo.")
endif()

# when this option is on, we install all static lib deps into lib
if(MLC_LLM_INSTALL_STATIC_LIB)
  install(TARGETS mlc_llm_static tokenizers_cpp sentencepiece-static
                  mlc_tvm_runtime_static tvm_ffi_static
          ARCHIVE DESTINATION lib${LIB_SUFFIX})
  # tokenizers need special handling as it builds from rust
  if(MSVC)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/libtokenizers_c.lib
            DESTINATION lib${LIB_SUFFIX})
  else()
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/libtokenizers_c.a
            DESTINATION lib${LIB_SUFFIX})
  endif()
else()
  install(
    TARGETS tvm_runtime
            ${MLC_LLM_RUNTIME_BACKEND_LIBS}
            mlc_llm
            mlc_llm_module
            mlc_llm_static
            tokenizers_cpp
            sentencepiece-static
            RUNTIME_DEPENDENCY_SET
            tokenizers_c
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX})
endif()

# Python package installation configuration This section ensures that all
# necessary files are installed for the Python wheel
if(MLC_LLM_BUILD_PYTHON_MODULE)
  message(STATUS "Configuring Python package installation")

  # Set RPATH for mlc_llm and mlc_llm_module to find other libraries relatively
  if(APPLE)
    # macOS uses @loader_path
    set_target_properties(mlc_llm PROPERTIES INSTALL_RPATH "@loader_path")
    set_target_properties(mlc_llm_module PROPERTIES INSTALL_RPATH
                                                    "@loader_path")
  elseif(LINUX)
    # Linux uses $ORIGIN
    set_target_properties(mlc_llm PROPERTIES INSTALL_RPATH "\$ORIGIN")
    set_target_properties(mlc_llm_module PROPERTIES INSTALL_RPATH "\$ORIGIN")
  endif()

  # Install compiled shared libraries
  install(TARGETS mlc_llm DESTINATION ".")
  install(TARGETS mlc_llm_module DESTINATION ".")
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/" DESTINATION "cpp/")
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/web/" DESTINATION "web/")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
                "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
                "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE" DESTINATION ".")

  message(STATUS "Python package installation configured")
endif()
