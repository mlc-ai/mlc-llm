cmake_policy(SET CMP0069 NEW) # suppress cmake warning about IPO

set(MLC_CLI_SOURCES
  mlc_cli_chat.cc
  chat_state.cc
  engine.cc
)
set(MLC_CLI_LINKER_LIBS "")

set(
  MLC_CLI_CHAT_INCLUDES
  ../../3rdparty/tvm/include
  ../../3rdparty/tvm/3rdparty/dlpack/include
  ../../3rdparty/tvm/3rdparty/dmlc-core/include
  ../../3rdparty/tvm/3rdparty/picojson
  ../../3rdparty/tokenizers-cpp/include
  ../..//3rdparty/xgrammar/include
)

add_executable(mlc_cli_chat ${MLC_CLI_SOURCES})
target_include_directories(mlc_cli_chat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MLC_CLI_CHAT_INCLUDES}  ${PROJECT_SOURCE_DIR}/cpp)
target_link_libraries(mlc_cli_chat PUBLIC  mlc_llm ${TVM_RUNTIME_LINKER_LIBS})

if(USE_CUDA)
  include(../../3rdparty/tvm/cmake/utils/Utils.cmake)
  include(../../3rdparty/tvm/cmake/utils/FindCUDA.cmake)
  find_cuda(${USE_CUDA} ${USE_CUDNN})
  target_link_libraries(mlc_cli_chat PUBLIC ${CUDA_NVRTC_LIBRARY})
  target_link_libraries(mlc_cli_chat PUBLIC ${CUDA_CUDART_LIBRARY})
  target_link_libraries(mlc_cli_chat PUBLIC ${CUDA_CUDA_LIBRARY})
endif()
