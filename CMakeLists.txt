cmake_minimum_required(VERSION 3.18)
project(mlc_llm C CXX)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE
  )
  message(STATUS "Setting default build type to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

option(MLC_LLM_INSTALL_STATIC_LIB "Install static lib instead of cli" OFF)

if (MLC_LLM_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# tvm runtime config: minimize runtime components
set(USE_RPC OFF)
set(USE_MICRO OFF)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_GRAPH_EXECUTOR_DEBUG OFF)
set(USE_AOT_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_GTEST OFF)
set(USE_LIBBACKTRACE OFF)
add_subdirectory($ENV{TVM_HOME} tvm EXCLUDE_FROM_ALL)

# sentencepiece config
option(SPM_ENABLE_SHARED "override sentence piece config" OFF)
option(SPM_ENABLE_TCMALLOC "" OFF)


set(MLC_LLM_RUNTIME_LINKER_LIB "")
# provide macro if it does not exist in cmake system
# it is OK to skip those since we do not provide these apps in the ios
# instead just link to the sentencepiece directly
if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  macro (set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
      set_property (TARGET ${TARGET} PROPERTY
          XCODE_ATTRIBUTE_${XCODE_PROPERTY} ${XCODE_VALUE})
  endmacro (set_xcode_property)
endif()

add_subdirectory(3rdparty/sentencepiece-js/sentencepiece sentencepiece EXCLUDE_FROM_ALL)
add_subdirectory(3rdparty/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)


tvm_file_glob(GLOB_RECURSE MLC_LLM_SRCS cpp/*.cc)
tvm_file_glob(GLOB_RECURSE MLC_CLI_SRCS cpp/cli_main.cc)
list(REMOVE_ITEM MLC_LLM_SRCS ${MLC_CLI_SRCS})

add_library(mlc_llm_objs OBJECT ${MLC_LLM_SRCS})
add_library(mlc_cli_objs OBJECT ${MLC_CLI_SRCS})

set(
  MLC_LLM_INCLUDES
  $ENV{TVM_HOME}/include
  $ENV{TVM_HOME}/3rdparty/dlpack/include
  $ENV{TVM_HOME}/3rdparty/dmlc-core/include
)

set(MLC_LLM_COMPILE_DEFS DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

target_include_directories(mlc_llm_objs PRIVATE ${MLC_LLM_INCLUDES})
target_compile_definitions(mlc_llm_objs PRIVATE ${MLC_LLM_COMPILE_DEFS})
target_include_directories(mlc_llm_objs PRIVATE 3rdparty/sentencepiece-js/sentencepiece/src)
target_include_directories(mlc_llm_objs PRIVATE 3rdparty/tokenizers-cpp)
target_compile_definitions(mlc_llm_objs PRIVATE -DMLC_LLM_EXPORTS)

add_library(mlc_llm SHARED $<TARGET_OBJECTS:mlc_llm_objs>)
add_library(mlc_llm_static STATIC $<TARGET_OBJECTS:mlc_llm_objs>)
add_dependencies(mlc_llm_static sentencepiece-static tokenizers tvm_runtime)
set_target_properties(mlc_llm_static PROPERTIES OUTPUT_NAME mlc_llm)

target_link_libraries(mlc_llm PUBLIC tvm_runtime)
target_link_libraries(mlc_llm PRIVATE sentencepiece)
# note: somehow cmake needs to specify the dependency twice
target_link_libraries(mlc_llm PRIVATE tokenizers)

# Exmaple app that may depends on mlc_llm
add_executable(mlc_chat_cli $<TARGET_OBJECTS:mlc_cli_objs>)
target_include_directories(mlc_cli_objs PRIVATE ${MLC_LLM_INCLUDES})
target_include_directories(mlc_cli_objs PRIVATE 3rdparty/argparse/include)
target_compile_definitions(mlc_cli_objs PRIVATE ${MLC_LLM_COMPILE_DEFS})

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(mlc_llm PRIVATE log)
  target_link_libraries(mlc_chat_cli PRIVATE log)
endif()

if (MLC_LLM_INSTALL_STATIC_LIB)
  target_link_libraries(
    mlc_chat_cli PRIVATE mlc_llm_static sentencepiece-static tokenizers)
  target_link_libraries(
    mlc_chat_cli PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,tvm_runtime>")
else()
  target_link_libraries(mlc_chat_cli PUBLIC mlc_llm)
endif()

# when this option is on,
# we install all static lib deps into lib
if (MLC_LLM_INSTALL_STATIC_LIB)
  install(TARGETS
    mlc_llm_static
    sentencepiece-static
    tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  # tokenizers need special handling as it builds from rust
  if(MSVC)
    install(FILES ${CMAKE_BINARY_DIR}/tokenizers/libtokenizers_cpp.lib
      DESTINATION lib${LIB_SUFFIX}
      )
  else()
    install(FILES ${CMAKE_BINARY_DIR}/tokenizers/libtokenizers_cpp.a
      DESTINATION lib${LIB_SUFFIX}
      )
  endif()
else()
  install(TARGETS mlc_chat_cli tvm_runtime mlc_llm
    mlc_llm_static
    sentencepiece-static
    RUNTIME_DEPENDENCY_SET tokenizers
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
endif()
