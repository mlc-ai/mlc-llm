/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** GenerationConfig ******************/

/*! \brief The generation configuration of a request. */
class GenerationConfigNode : public Object {
 public:
  double temperature = 0.8;
  double top_p = 0.95;
  double frequency_penalty = 0.0;
  double presence_penalty = 0.0;
  double repetition_penalty = 1.0;

  int max_tokens = 128;
  Array<String> stop_strs;
  std::vector<int> stop_token_ids;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.GenerationConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GenerationConfigNode, Object);
};

class GenerationConfig : public ObjectRef {
 public:
  explicit GenerationConfig(String config_json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(GenerationConfig, ObjectRef, GenerationConfigNode);
};

/****************** KV Cache config ******************/

/*! \brief The configuration of paged KV cache. */
class KVCacheConfigNode : public Object {
 public:
  int page_size;
  int max_num_sequence;
  int max_total_sequence_length;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.KVCacheConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(KVCacheConfigNode, Object);
};

class KVCacheConfig : public ObjectRef {
 public:
  explicit KVCacheConfig(int page_size, int max_num_sequence, int max_total_sequence_length);

  explicit KVCacheConfig(const std::string& config_str, int max_single_sequence_length);

  TVM_DEFINE_OBJECT_REF_METHODS(KVCacheConfig, ObjectRef, KVCacheConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
