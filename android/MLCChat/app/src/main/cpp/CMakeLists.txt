cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_CXX_STANDARD 14)

set(LOCAL_PATH ${CMAKE_CURRENT_LIST_DIR})
set(MY_PATH ${LOCAL_PATH})
set(LIB_PATH ${MY_PATH}/../../../../../build/lib)
set(TVM_HOME ${MY_PATH}/../../../../../build/tvm_home)


#####################################################

add_library(local_mlc_llm STATIC IMPORTED)
set_target_properties(local_mlc_llm PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libmlc_llm.a)

#####################################################

add_library(local_sentencepiece STATIC IMPORTED)
set_target_properties(local_sentencepiece PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libsentencepiece.a)

#####################################################

add_library(local_tokenizers_c STATIC IMPORTED)
set_target_properties(local_tokenizers_c PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libtokenizers_c.a)

#####################################################

add_library(local_tokenizers_cpp STATIC IMPORTED)
set_target_properties(local_tokenizers_cpp PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libtokenizers_cpp.a)

#####################################################

add_library(local_tvm_runtime STATIC IMPORTED)
set_target_properties(local_tvm_runtime PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libtvm_runtime.a)

#####################################################

add_library(local_model_android STATIC IMPORTED)
set_target_properties(local_model_android PROPERTIES IMPORTED_LOCATION ${LIB_PATH}/libmodel_android.a)

#####################################################

add_library(tvm4j_runtime_packed SHARED org_apache_tvm_native_c_api.cc)

target_include_directories(tvm4j_runtime_packed PUBLIC
        ${TVM_HOME}/include
        ${TVM_HOME}/3rdparty/dlpack/include
        ${TVM_HOME}/3rdparty/dmlc-core/include
        ${MY_PATH}
        )

target_link_libraries(tvm4j_runtime_packed
        local_sentencepiece
        local_tokenizers_c
        local_tokenizers_cpp
        -Wl,--whole-archive
        local_mlc_llm
        local_tvm_runtime
        local_model_android
        -Wl,--no-whole-archive
        )

#####################################################

# Add other dependencies and compile flags as needed

#####################################################
