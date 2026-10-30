# Keep libtvm_runtime.a usable by the iOS and Android static builds after
# TVM split the VM and device backends into separate shared libraries.
if(TVM_TIRX_USE_THREADS)
  find_package(Threads REQUIRED)
endif()
set_property(TARGET tvm_runtime_extra_defs PROPERTY INTERFACE_LINK_LIBRARIES
             tvm_tirx::header tvm_ffi::header)
set(_mlc_runtime_sources "")
set(_mlc_runtime_links "")
foreach(_target tvm_runtime ${MLC_LLM_RUNTIME_BACKEND_LIBS})
  get_target_property(_source_dir ${_target} SOURCE_DIR)
  get_target_property(_sources ${_target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(IS_ABSOLUTE "${_source}" OR "${_source}" MATCHES "^\\$<")
      list(APPEND _mlc_runtime_sources "${_source}")
    else()
      list(APPEND _mlc_runtime_sources "${_source_dir}/${_source}")
    endif()
  endforeach()
  get_target_property(_links ${_target} LINK_LIBRARIES)
  foreach(_link IN LISTS _links)
    if(TARGET "${_link}")
      get_target_property(_type "${_link}" TYPE)
      if(_type STREQUAL "OBJECT_LIBRARY")
        list(APPEND _mlc_runtime_sources "$<TARGET_OBJECTS:${_link}>")
      elseif(NOT "${_link}" MATCHES "^tvm_")
        list(APPEND _mlc_runtime_links "${_link}")
      endif()
    elseif(_link)
      list(APPEND _mlc_runtime_links "${_link}")
    endif()
  endforeach()
endforeach()
list(REMOVE_DUPLICATES _mlc_runtime_sources)
list(REMOVE_DUPLICATES _mlc_runtime_links)
add_library(mlc_tvm_runtime_static STATIC ${_mlc_runtime_sources})
set_target_properties(mlc_tvm_runtime_static PROPERTIES
  OUTPUT_NAME tvm_runtime CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(mlc_tvm_runtime_static PUBLIC tvm_tirx::header tvm_ffi::header
                      ${_mlc_runtime_links})
target_include_directories(mlc_tvm_runtime_static PRIVATE
  "$<TARGET_PROPERTY:tvm_runtime_extra,INCLUDE_DIRECTORIES>"
  "$<TARGET_PROPERTY:tvm_runtime_extra_defs,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_definitions(mlc_tvm_runtime_static PRIVATE
  "$<TARGET_PROPERTY:tvm_runtime_extra,COMPILE_DEFINITIONS>"
  "$<TARGET_PROPERTY:tvm_runtime_extra_defs,INTERFACE_COMPILE_DEFINITIONS>"
  "$<TARGET_PROPERTY:tvm_runtime,INTERFACE_COMPILE_DEFINITIONS>")
