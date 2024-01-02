/*!
 * \file model.h
 * \brief Metadata stored in model lib
 */
#ifndef MLC_LLM_CPP_MODEL_METADATA_H_
#define MLC_LLM_CPP_MODEL_METADATA_H_

#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>

#include <unordered_map>

// Forward decalre picojson's value, object and array
namespace picojson {
class value;
using object = std::unordered_map<std::string, value>;
using array = std::vector<value>;
}  // namespace picojson

namespace mlc {
namespace llm {

struct ModelMetadata {
  struct Param {
    struct Preproc {
      tvm::runtime::String func_name;
      tvm::runtime::ShapeTuple out_shape;
      tvm::runtime::DataType out_dtype;
      static Preproc FromJSON(const picojson::object& js);
    };

    tvm::runtime::String name;
    tvm::runtime::ShapeTuple shape;
    tvm::runtime::DataType dtype;
    std::vector<Preproc> preprocs;
    static Param FromJSON(const picojson::object& param_obj);
  };

  std::string model_type;
  std::string quantization;
  int64_t context_window_size;
  int64_t prefill_chunk_size;
  int64_t sliding_window_size;
  int64_t tensor_parallel_shards;
  int64_t attention_sink_size;
  std::vector<Param> params;
  std::unordered_map<std::string, int64_t> memory_usage;

  static ModelMetadata FromJSON(const picojson::object& json_str);
  static ModelMetadata FromModule(tvm::runtime::Module module);
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_MODEL_METADATA_H_
