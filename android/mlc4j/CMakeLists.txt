cmake_minimum_required(VERSION 3.18)

project(mlc-chat C CXX)

set(ANDROID_DIR ${CMAKE_CURRENT_LIST_DIR})
set(ANDROID_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(MLC_LLM_DIR ${ANDROID_DIR}/../..)
set(MLC_LLM_BINARY_DIR mlc_llm)
set(MLC_LLM_COMPILE_DEFS TVM_LOG_CUSTOMIZE=1)
add_subdirectory(${MLC_LLM_DIR} ${MLC_LLM_BINARY_DIR} EXCLUDE_FROM_ALL)

if(NOT DEFINED TVM_SOURCE_DIR)
  set(TVM_SOURCE_DIR ${MLC_LLM_DIR}/3rdparty/tvm)
endif(NOT DEFINED TVM_SOURCE_DIR)
message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")

find_package(Java REQUIRED)
include(UseJava)

find_package(JNI)
if(JNI_FOUND)
  message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
else()
  message(STATUS "Try to find jni directly from android env")
  # try to find JNI_LIBRARY
  find_path(JNI_INCLUDE_DIRS NAMES "jni.h")
  message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

file(GLOB_RECURSE javasources
     ${TVM_SOURCE_DIR}/jvm/core/src/main/java/org/apache/tvm/*.java
     ${ANDROID_DIR}/src/java/*.java)
set(JNI_HEADER ${CMAKE_BINARY_DIR}/jni_header)
add_jar(tvm4j_core ${javasources} GENERATE_NATIVE_HEADERS tvm4jheaders
        DESTINATION ${JNI_HEADER})

add_custom_command(
  TARGET tvm4j_core
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${JNI_HEADER}/org_apache_tvm_LibInfo.h
          ${JNI_HEADER}/org_apache_tvm_native_c_api.h)

add_library(model_android STATIC IMPORTED)
set_target_properties(
  model_android PROPERTIES IMPORTED_LOCATION
                           ${ANDROID_BIN_DIR}/lib/libmodel_android.a)

add_library(
  tvm4j_runtime_packed SHARED
  ${TVM_SOURCE_DIR}/jvm/native/src/main/native/org_apache_tvm_native_c_api.cc)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS}
                         TVM_SOURCE_DIR=${TVM_SOURCE_DIR})

target_include_directories(
  tvm4j_runtime_packed
  PUBLIC ${JNI_INCLUDE_DIRS}
         ${JNI_HEADER}
         ${ANDROID_DIR}/src/cpp
         ${TVM_SOURCE_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include
         ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
         ${TVM_SOURCE_DIR}/3rdparty/OpenCL-Headers
         ${TVM_SOURCE_DIR}/3rdparty/picojson
         ${TVM_SOURCE_DIR}/include
         ${TVM_SOURCE_DIR}/src
         ${TVM_SOURCE_DIR}/3rdparty/tvm-ffi/include
         ${TVM_SOURCE_DIR}/3rdparty/tvm-ffi/src)
target_compile_definitions(tvm4j_runtime_packed PUBLIC ${MLC_LLM_COMPILE_DEFS})
target_compile_definitions(
  tvm4j_runtime_packed
  PUBLIC TVM_VM_ENABLE_PROFILER=0
  PUBLIC TVM_FFI_USE_LIBBACKTRACE=0
  PUBLIC TVM_FFI_BACKTRACE_ON_SEGFAULT=0)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF)
target_link_libraries(
  tvm4j_runtime_packed
  tokenizers_c
  tokenizers_cpp
  log
  -Wl,--whole-archive
  mlc_llm_static
  model_android
  -Wl,--no-whole-archive)

target_compile_definitions(tvm4j_runtime_packed PUBLIC TVM4J_ANDROID)
add_dependencies(tvm4j_runtime_packed tvm4j_core)

target_compile_definitions(mlc_llm_objs PUBLIC MLC_SINGLE_GPU_ONLY)

install_jar(tvm4j_core output)
install(TARGETS tvm4j_runtime_packed LIBRARY DESTINATION output/${ANDROID_ABI})
