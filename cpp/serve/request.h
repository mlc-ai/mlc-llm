/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/request.h
 * \brief Implementation of llm chat.
 */
#ifndef MLC_LLM_SERVE_REQUEST_H_
#define MLC_LLM_SERVE_REQUEST_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/object.h>

#include "../tokenizers/tokenizers.h"
#include "config.h"
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** Request ******************/

/*!
 * \brief The user submitted text-generation request, which contains
 * a unique request id, a list of multi-modal inputs, a set of
 * generation configuration parameters.
 * \note Request is immutable and can be re-dispatched to another
 * node and restart the request handling on the new one.
 */
class RequestNode : public Object {
 public:
  /*!
   * \brief The unique identifier of the request.
   * Different requests should have different ids.
   */
  String id;
  /*!
   * \brief The user inputs of a request. Input may have multi-modality.
   * \sa data.h
   */
  Array<Data> inputs;
  /*!
   * \brief The equivalent input sequence length of the request.
   * "-1" means the input length is unknown due to the existence
   * of untokenized text data.
   */
  int prompt_tokens = -1;
  /*!
   * \brief The sampling configuration which may contain temperature,
   * top_p, repetition_penalty, max_gen_len, etc.
   */
  GenerationConfig generation_cfg;
  /*! \brief Backward reference to the request state. */
  Object* rstate = nullptr;

  static constexpr const char* _type_key = "mlc.serve.Request";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(RequestNode, Object);
};

class Request : public ObjectRef {
 public:
  explicit Request(String id, Array<Data> inputs, GenerationConfig generation_cfg);

  /*!
   * \brief Return a request object with all text data tokenized,
   * and the request ID kept the same as the input one.
   * \param request The request to be tokenized.
   * \param tokenizer The tokenizer to tokenize the input data of the given request.
   * \return The request object whose data are tokenized.
   */
  static Request FromUntokenized(const Request& request, const Tokenizer& tokenizer);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Request, ObjectRef, RequestNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_REQUEST_H_
