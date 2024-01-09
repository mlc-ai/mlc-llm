cmake_minimum_required(VERSION 3.18)

project(mlc-chat C CXX)

set(ANDROID_DIR ${CMAKE_CURRENT_LIST_DIR})
set(ANDROID_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(MLC_LLM_DIR ${ANDROID_DIR}/../..)
set(MLC_LLM_BINARY_DIR mlc_llm)
set(MLC_LLM_COMPILE_DEFS TVM_LOG_CUSTOMIZE=1)
add_subdirectory(${MLC_LLM_DIR} ${MLC_LLM_BINARY_DIR} EXCLUDE_FROM_ALL)

if (NOT DEFINED TVM_HOME)
  set(TVM_HOME ${MLC_LLM_DIR}/3rdparty/tvm)
endif (NOT DEFINED TVM_HOME)
message(STATUS "TVM_HOME: ${TVM_HOME}")

find_package(Java REQUIRED)
find_package(JNI REQUIRED)
if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()
include(UseJava)


file(GLOB_RECURSE javasources
    ${TVM_HOME}/jvm/core/src/main/java/org/apache/tvm/*.java
    ${ANDROID_DIR}/src/java/*.java
)
set(JNI_HEADER ${CMAKE_BINARY_DIR}/jni_header)
add_jar(tvm4j_core ${javasources} GENERATE_NATIVE_HEADERS tvm4jheaders DESTINATION ${JNI_HEADER})

add_custom_command(
  TARGET tvm4j_core POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${JNI_HEADER}/org_apache_tvm_LibInfo.h ${JNI_HEADER}/org_apache_tvm_native_c_api.h
)

add_library(model_android STATIC IMPORTED)
set_target_properties(model_android PROPERTIES IMPORTED_LOCATION ${ANDROID_BIN_DIR}/model_lib/libmodel_android.a)

add_library(tvm4j_runtime_packed SHARED ${TVM_HOME}/jvm/native/src/main/native/org_apache_tvm_native_c_api.cc)

target_include_directories(tvm4j_runtime_packed PUBLIC
  ${JNI_INCLUDE_DIRS}
  ${JNI_HEADER}
  ${ANDROID_DIR}/src/cpp
  ${TVM_HOME}/3rdparty/dlpack/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
  ${TVM_HOME}/include
)

target_link_libraries(tvm4j_runtime_packed
  sentencepiece-static
  tokenizers_c
  tokenizers_cpp
  log
  -Wl,--whole-archive
  tvm_runtime
  mlc_llm_static
  model_android
  -Wl,--no-whole-archive
)

target_compile_definitions(tvm4j_runtime_packed PUBLIC TVM4J_ANDROID)
add_dependencies(tvm4j_runtime_packed tvm4j_core)

install_jar(tvm4j_core output)
install(TARGETS tvm4j_runtime_packed LIBRARY DESTINATION output/${ANDROID_ABI})
