/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request.h
 * \brief Implementation of llm chat.
 */
#ifndef MLC_LLM_SERVE_REQUEST_H_
#define MLC_LLM_SERVE_REQUEST_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include "config.h"
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** Request ******************/

/*!
 * \brief The user submitted text-generation request, which contains
 * a list of multi-modal inputs, a set of generation configuration
 * parameters and a callback function for request finish handling.
 * \note Request is immutable and can be re-dispatched to another
 * node and restart the request handling on the new one.
 */
class RequestNode : public Object {
 public:
  /*!
   * \brief The user inputs of a request. Input may have multi-modality.
   * \sa data.h
   */
  Array<Data> inputs;
  /*!
   * \brief The sampling configuration which may contain temperature,
   * top_p, repetition_penalty, max_gen_len, etc.
   */
  GenerationConfig generation_cfg;
  /*!
   * \brief The provided callback function to handle the generation
   * output. It has the signature of `(Request, Data) -> None`,
   * which takes the request and the generation output as parameters.
   */
  PackedFunc fcallback;

  static constexpr const char* _type_key = "mlc.serve.Request";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(RequestNode, Object);
};

class Request : public ObjectRef {
 public:
  explicit Request(Array<Data> inputs, GenerationConfig generation_cfg, PackedFunc fcallback);

  TVM_DEFINE_OBJECT_REF_METHODS(Request, ObjectRef, RequestNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_REQUEST_H_
