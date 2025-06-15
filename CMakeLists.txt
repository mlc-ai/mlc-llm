cmake_minimum_required(VERSION 3.18)
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

if(MLC_LLM_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

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
set(CMAKE_CXX_STANDARD 17)
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
set(BUILD_DUMMY_LIBTVM ON)
if(NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  else()
    set(TVM_SOURCE_DIR 3rdparty/tvm)
  endif(DEFINED ENV{TVM_SOURCE_DIR})
endif(NOT DEFINED TVM_SOURCE_DIR)
message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)

set(MLC_LLM_RUNTIME_LINKER_LIB "")
set(TOKENZIER_CPP_PATH 3rdparty/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

set(XGRAMMAR_PATH 3rdparty/xgrammar)
tvm_file_glob(GLOB_RECURSE MLC_LLM_SRCS cpp/*.cc)
tvm_file_glob(GLOB_RECURSE XGRAMMAR_SRCS ${XGRAMMAR_PATH}/cpp/*.cc)
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX "${XGRAMMAR_PATH}/cpp/pybind/.*\\.cc")
list(APPEND MLC_LLM_SRCS ${XGRAMMAR_SRCS})
add_library(mlc_llm_objs OBJECT ${MLC_LLM_SRCS})

set(MLC_LLM_INCLUDES
    ${TVM_SOURCE_DIR}/include ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
    ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
    ${TVM_SOURCE_DIR}/3rdparty/picojson)

set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS}
                         DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS} __STDC_FORMAT_MACROS=1)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS} PICOJSON_USE_INT64)
set(MLC_LLM_COMPILE_DEFS ${MLC_LLM_COMPILE_DEFS} XGRAMMAR_ENABLE_LOG_DEBUG=0)

target_compile_definitions(mlc_llm_objs PRIVATE ${MLC_LLM_COMPILE_DEFS})
target_compile_definitions(mlc_llm_objs PRIVATE -DMLC_LLM_EXPORTS)
target_include_directories(mlc_llm_objs PRIVATE ${MLC_LLM_INCLUDES})
target_include_directories(mlc_llm_objs PRIVATE 3rdparty/stb)
target_include_directories(mlc_llm_objs PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(mlc_llm_objs PRIVATE ${XGRAMMAR_PATH}/include)
target_link_libraries(mlc_llm_objs PRIVATE tvm_ffi_header)

add_library(mlc_llm SHARED $<TARGET_OBJECTS:mlc_llm_objs>)
add_library(mlc_llm_static STATIC $<TARGET_OBJECTS:mlc_llm_objs>)
add_dependencies(mlc_llm_static tokenizers_cpp sentencepiece-static
                 tokenizers_c tvm_runtime)
set_target_properties(mlc_llm_static PROPERTIES OUTPUT_NAME mlc_llm)

target_link_libraries(mlc_llm PUBLIC tvm_runtime)
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
                             PRIVATE ${PROJECT_SOURCE_DIR}/cpp)
  target_include_directories(
    mlc_llm_cpp_tests PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(mlc_llm_cpp_tests PUBLIC mlc_llm gtest gtest_main)
endif(BUILD_CPP_TEST)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(mlc_llm PRIVATE log)
  target_link_libraries(tokenizers_cpp PRIVATE log)
endif()

add_library(mlc_llm_module SHARED $<TARGET_OBJECTS:mlc_llm_objs>)
target_link_libraries(mlc_llm_module PUBLIC tvm)
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
  install(TARGETS mlc_llm_static tokenizers_cpp sentencepiece-static tvm_runtime
          LIBRARY DESTINATION lib${LIB_SUFFIX})
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
