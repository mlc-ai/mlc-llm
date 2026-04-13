/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/data.h
 */
#ifndef MLC_LLM_SERVE_DATA_H_
#define MLC_LLM_SERVE_DATA_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/node/cast.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/tensor.h>

#include <atomic>
#include <optional>

#include "../tokenizers/tokenizers.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;
using tvm::ffi::Optional;

class Model;

/****************** DataNode ******************/

/*! \brief The base class of multi-modality data (text, tokens, embedding, etc). */
class DataNode : public Object {
 public:
  /*! \brief Get the length (equivalent number of tokens) of the data. */
  virtual int GetLength() const = 0;

  /*!
   * \brief Compute the embedding of this data with regard to the input model.
   * When the input destination pointer is not nullptr, it in-place writes the
   * embedding into the input destination array at the given offset.
   * Otherwise, the embeddings will be directly returned back.
   * \param model The model to take embeddings from.
   * \param dst The destination array of the embedding lookup.
   * \param offset The token offset where the computed embeddings will be written
   * into the destination array.
   * \return The updated destination embedding array or the computed embeddings.
   * \note When `dst` is nullptr, we require `offset` to be 0.
   */
  virtual ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataNode>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const uint32_t _type_child_slots = 3;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.Data", DataNode, Object);
};

class Data : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Data, ObjectRef, DataNode);
};

/*! \brief Split the given data array into two arrays at the "split_pos" position. */
std::pair<Array<Data>, Array<Data>> SplitData(const Array<Data>& original_data, int total_length,
                                              int split_pos);

/****************** TextDataNode ******************/

/*! \brief The class of text data, containing a text string. */
class TextDataNode : public DataNode {
 public:
  /*! \brief The text string. */
  tvm::ffi::String text;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TextDataNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mlc.serve.TextData", TextDataNode, DataNode);
};

class TextData : public Data {
 public:
  explicit TextData(String text);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TextData, Data, TextDataNode);
};

/****************** TokenDataNode ******************/

/*! \brief The class of token data, containing a list of token ids. */
class TokenDataNode : public DataNode {
 public:
  /*! \brief The token ids. */
  IntTuple token_ids;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TokenDataNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mlc.serve.TokenData", TokenDataNode, DataNode);
};

class TokenData : public Data {
 public:
  explicit TokenData(IntTuple token_ids);

  explicit TokenData(std::vector<int32_t> token_ids);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TokenData, Data, TokenDataNode);
};

/****************** ImageDataNode ******************/

/*! \brief The class of image data, containing a 3D array of pixel values. */
class ImageDataNode : public DataNode {
 public:
  /*! \brief The pixel values. */
  Tensor image;
  int embed_size;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ImageDataNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mlc.serve.ImageData", ImageDataNode, DataNode);
};

class ImageData : public Data {
 public:
  explicit ImageData(Tensor image, int embed_size);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ImageData, Data, ImageDataNode);
};

/****************** SampleResult ******************/

// The pair of a token id and its probability in sampling.
using TokenProbPair = std::pair<int32_t, float>;

/*!
 * \brief The class of sampler's sampling result.
 * It's not a TVM object since it will not be used directly on Python side.
 */
struct SampleResult {
  /*! \brief The token id and probability of the sampled token. */
  TokenProbPair sampled_token_id;
  /*! \brief The token id and probability of the tokens with top probabilities. */
  std::vector<TokenProbPair> top_prob_tokens;

  /*! \brief Get the sampled token id. */
  int32_t GetTokenId() const;

  /*!
   * \brief Get the logprob JSON string of this token with regard
   * to OpenAI API at https://platform.openai.com/docs/api-reference/chat/object.
   * \param tokenizer The tokenizer for token table lookup.
   * \param logprob A boolean indicating if need to return log probability.
   * \return A JSON string that conforms to the logprob spec in OpenAI API.
   */
  std::string GetLogProbJSON(const Tokenizer& tokenizer, bool logprob) const;
};

/****************** RequestStreamOutput ******************/

/*!
 * \brief The generated delta request output that is streamed back
 * through callback stream function.
 *
 * \note: This output object corresponds to parallel generated outputs when n != 1.
 *
 * For example, if n=2, then group_delta_token_ids[0] matches to the output stream 0
 * and group_delta_token_ids[1] matches to the output stream 1
 */
class RequestStreamOutputObj : public Object {
 public:
  /*! \brief The id of the request that the function is invoked for. */
  String request_id;
  /*!
   * \brief The new generated token ids since the last callback invocation
   * for the input request.
   */
  std::vector<std::vector<int64_t>> group_delta_token_ids;
  /*! \brief The logprobs JSON strings of the new generated tokens since last invocation. */
  std::optional<std::vector<std::vector<String>>> group_delta_logprob_json_strs;
  /*!
   * \brief The finish reason of the request when it is finished,
   * of None if the request has not finished yet.
   */
  std::vector<Optional<String>> group_finish_reason;
  /*!
   * \brief The usage field of the response, this is global to all streams.
   */
  Optional<String> request_final_usage_json_str;

  /*!
   * \brief The extra prefix string of all requests.
   */
  std::vector<String> group_extra_prefix_string;

  std::atomic<bool> unpacked = false;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RequestStreamOutputObj>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.RequestStreamOutput", RequestStreamOutputObj, Object);
};

/*!
 * \brief Managed reference to RequestStreamOutputObj.
 * \sa RequestStreamOutputObj
 */
class RequestStreamOutput : public ObjectRef {
 public:
  explicit RequestStreamOutput(
      String request_id, std::vector<std::vector<int64_t>> group_delta_token_ids,
      std::optional<std::vector<std::vector<String>>> group_delta_logprob_json_strs,
      std::vector<Optional<String>> group_finish_reason,
      std::vector<String> group_extra_prefix_string);

  static RequestStreamOutput Usage(String request_id, String request_final_usage_json_str);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(RequestStreamOutput, ObjectRef,
                                             RequestStreamOutputObj);
};

/****************** Embedding Types ******************/

/*! \brief The pooling strategy for encoder embedding. */
enum class PoolingStrategy : int {
  kCLS = 0,
  kMean = 1,
  kLast = 2,
};

/*!
 * \brief A single embedding item within an embedding request.
 * Each item is a canonicalized sequence of token IDs
 * (already includes CLS/SEP, already truncated).
 */
struct EmbeddingItem {
  /*! \brief The token ids for this item (canonicalized by Python). */
  std::vector<int32_t> token_ids;
  /*! \brief The original index of this item in the request. */
  int item_index;
};

/*!
 * \brief An embedding request containing one or more items.
 * The C++ side only sees canonicalized token-id items.
 */
struct EmbeddingRequestNode : public Object {
  /*! \brief The unique identifier of the request. */
  String id;
  /*! \brief The items to embed. */
  std::vector<EmbeddingItem> items;
  /*! \brief The pooling strategy. */
  PoolingStrategy pooling_strategy = PoolingStrategy::kCLS;
  /*! \brief Whether to L2-normalize the output embeddings. */
  bool normalize = true;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EmbeddingRequestNode>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.EmbeddingRequest", EmbeddingRequestNode, Object);
};

class EmbeddingRequest : public ObjectRef {
 public:
  explicit EmbeddingRequest(String id, std::vector<EmbeddingItem> items,
                            PoolingStrategy pooling_strategy, bool normalize);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EmbeddingRequest, ObjectRef, EmbeddingRequestNode);
};

/*!
 * \brief The result of an embedding request, carrying
 * request-level aggregated embeddings on CPU.
 */
class EmbeddingResultObj : public Object {
 public:
  /*! \brief The request id. */
  String request_id;
  /*!
   * \brief The pooled embeddings as a CPU NDArray of shape [num_items, hidden_dim].
   * Lifetime is owned by the result; Python can read it directly.
   */
  Tensor embeddings;
  /*! \brief Total number of prompt tokens across all items. */
  int prompt_tokens = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EmbeddingResultObj>();
  }

  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.EmbeddingResult", EmbeddingResultObj, Object);
};

class EmbeddingResult : public ObjectRef {
 public:
  explicit EmbeddingResult(String request_id, Tensor embeddings, int prompt_tokens);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EmbeddingResult, ObjectRef, EmbeddingResultObj);
};

/*!
 * \brief Internal state for tracking an in-flight embedding request
 * inside the engine. Holds per-item completion status and the
 * result buffer that items write into.
 */
struct EmbeddingRequestState {
  /*! \brief The embedding request. */
  EmbeddingRequest request;
  /*! \brief Number of items that have been completed so far. */
  int completed_items = 0;
  /*! \brief The CPU result buffer [num_items, hidden_dim], pre-allocated. */
  Tensor result_buffer;
  /*! \brief Total prompt tokens across all items. */
  int prompt_tokens = 0;

  explicit EmbeddingRequestState(EmbeddingRequest request, Tensor result_buffer)
      : request(std::move(request)), result_buffer(std::move(result_buffer)) {}
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_DATA_H_
