/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <optional>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm;
using namespace tvm::runtime;

/****************** GenerationConfig ******************/

/*! \brief The response format of a request. */
struct ResponseFormat {
  String type = "text";
  Optional<String> schema = NullOpt;
};

/*! \brief The generation configuration of a request. */
class GenerationConfigNode : public Object {
 public:
  int n = 1;
  double temperature = 0.8;
  double top_p = 0.95;
  double frequency_penalty = 0.0;
  double presence_penalty = 0.0;
  double repetition_penalty = 1.0;
  bool logprobs = false;
  int top_logprobs = 0;
  std::vector<std::pair<int, float>> logit_bias;
  int seed;
  bool ignore_eos = false;

  int max_tokens = 128;
  Array<String> stop_strs;
  std::vector<int> stop_token_ids;

  ResponseFormat response_format;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.GenerationConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GenerationConfigNode, Object);
};

class GenerationConfig : public ObjectRef {
 public:
  explicit GenerationConfig(String config_json_str);

  /*!
   * \brief Parse the generation config from the given JSON string.
   * When parsing fails, errors are dumped to the input error string, and NullOpt is returned.
   */
  static Optional<GenerationConfig> FromJSON(const std::string& json_str, std::string* err);

  TVM_DEFINE_OBJECT_REF_METHODS(GenerationConfig, ObjectRef, GenerationConfigNode);
};

/****************** KV Cache config ******************/

/*! \brief The configuration of paged KV cache. */
class KVCacheConfigNode : public Object {
 public:
  int page_size;
  int max_num_sequence;
  int max_total_sequence_length;
  int prefill_chunk_size;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.KVCacheConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(KVCacheConfigNode, Object);
};

class KVCacheConfig : public ObjectRef {
 public:
  explicit KVCacheConfig(int page_size, int max_num_sequence, int max_total_sequence_length,
                         int prefill_chunk_size);

  explicit KVCacheConfig(const std::string& config_str, int max_single_sequence_length);

  TVM_DEFINE_OBJECT_REF_METHODS(KVCacheConfig, ObjectRef, KVCacheConfigNode);
};

/****************** Engine Mode ******************/

/*! \brief The speculative mode. */
enum class SpeculativeMode : int {
  kDisable = 0,
  kSmallDraft = 1,
  kEagle = 2,
};

/*! \brief The configuration of engine execution config. */
class EngineConfigNode : public Object {
 public:
  /* The number of tokens to generate in speculative proposal (draft) */
  int spec_draft_length;
  /* The speculative mode. */
  SpeculativeMode speculative_mode;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.EngineConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EngineConfigNode, Object);
};

class EngineConfig : public ObjectRef {
 public:
  explicit EngineConfig(int spec_draft_length, int speculative_mode);

  explicit EngineConfig(const std::string& config_str);

  TVM_DEFINE_OBJECT_REF_METHODS(EngineConfig, ObjectRef, EngineConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
