add_library(mlc_llm_serve_objects OBJECT
    // ... existing code ...
    lora.cc
    lora_manager.cc
)

# LoRA loader dependencies
target_include_directories(mlc_llm_serve_objects
    PRIVATE
        ${CMAKE_SOURCE_DIR}/3rdparty
)

# zlib is required for the mini cnpy header (<zlib.h>).  We only include the
# headers and do not link against the library because the minimal ZIP reader
# avoids any zlib symbols.  Still, add the library if available so future
# extensions (e.g. DEFLATE support) can rely on it.
find_package(ZLIB)
if(ZLIB_FOUND)
  target_include_directories(mlc_llm_serve_objects PRIVATE ${ZLIB_INCLUDE_DIRS})
  target_link_libraries(mlc_llm_serve_objects PRIVATE ${ZLIB_LIBRARIES})
endif() 