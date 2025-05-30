/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/data.h
 */
#ifndef MLC_LLM_SERVE_DATA_H_
#define MLC_LLM_SERVE_DATA_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

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

  static constexpr const char* _type_key = "mlc.serve.Data";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(DataNode, Object);
};

class Data : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Data, ObjectRef, DataNode);
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

  static constexpr const char* _type_key = "mlc.serve.TextData";
  TVM_DECLARE_BASE_OBJECT_INFO(TextDataNode, DataNode);
};

class TextData : public Data {
 public:
  explicit TextData(String text);

  TVM_DEFINE_OBJECT_REF_METHODS(TextData, Data, TextDataNode);
};

/****************** TokenDataNode ******************/

/*! \brief The class of token data, containing a list of token ids. */
class TokenDataNode : public DataNode {
 public:
  /*! \brief The token ids. */
  IntTuple token_ids;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static constexpr const char* _type_key = "mlc.serve.TokenData";
  TVM_DECLARE_BASE_OBJECT_INFO(TokenDataNode, DataNode);
};

class TokenData : public Data {
 public:
  explicit TokenData(IntTuple token_ids);

  explicit TokenData(std::vector<int32_t> token_ids);

  TVM_DEFINE_OBJECT_REF_METHODS(TokenData, Data, TokenDataNode);
};

/****************** ImageDataNode ******************/

/*! \brief The class of image data, containing a 3D array of pixel values. */
class ImageDataNode : public DataNode {
 public:
  /*! \brief The pixel values. */
  NDArray image;
  int embed_size;

  int GetLength() const final;
  ObjectRef GetEmbedding(Model model, ObjectRef* dst = nullptr, int offset = 0) const final;

  static constexpr const char* _type_key = "mlc.serve.ImageData";
  TVM_DECLARE_BASE_OBJECT_INFO(ImageDataNode, DataNode);
};

class ImageData : public Data {
 public:
  explicit ImageData(NDArray image, int embed_size);

  TVM_DEFINE_OBJECT_REF_METHODS(ImageData, Data, ImageDataNode);
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

  static constexpr const char* _type_key = "mlc.serve.RequestStreamOutput";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(RequestStreamOutputObj, Object);
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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RequestStreamOutput, ObjectRef, RequestStreamOutputObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_DATA_H_
