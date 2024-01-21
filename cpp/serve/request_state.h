/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request_state.h
 * \brief The data structure maintaining the generation states of user requests.
 */
#ifndef MLC_LLM_SERVE_REQUEST_STATE_H_
#define MLC_LLM_SERVE_REQUEST_STATE_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include "../random.h"
#include "config.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The state of a request with regard to some single model.
 * \details In MLC LLM, the serving engine may leverage multiple models
 * to fulfill a user generation request (e.g., use speculation decoding).
 * For each request, we isolate its states (e.g. the generated tokens)
 * on each model. This is to say, we use RequestModelState to store
 * the state of a user request on a single model (rather than all models).
 */
class RequestModelStateNode : public Object {
 public:
  /*! \brief The request that this state corresponds to. */
  Request request;
  /*!
   * \brief The internal request id of this state.
   * It is the **physical index** of the request in the running request queue.
   * If the request is on hold (not in the running queue), the request id
   * should be -1.
   */
  int64_t internal_id = -1;
  /*! \brief The corresponding model id of this state. */
  int model_id = -1;
  /*!
   * \brief The committed generated token ids. A token is "committed"
   * means it will no longer be updated (or changed).
   */
  std::vector<int32_t> committed_tokens;
  /*! \brief The list of input data yet for the model to prefill. */
  Array<Data> inputs;

  // NOTE: The following fields are reserved for future speculative inference
  // settings, and are produced by the speculative small models.
  /*!
   * \brief The draft generated token ids, which are usually generated
   * by "small" speculative models. These tokens will be fed to a "large"
   * model to determine the final result of speculation.
   */
  std::vector<int32_t> draft_output_tokens;
  /*!
   * \brief The probability distribution on each position in the
   * draft. We keep the distributions for stochastic sampling when merging
   * speculations from multiple models.
   * \note We only need this value when we have multiple parallel small models
   * and draft outputs in speculative inference settings.
   */
  std::vector<NDArray> draft_output_prob_dist;
  /*!
   * \brief The probability of the sampled token on each position in the
   * draft. We keep the probabilities for stochastic sampling when merging
   * speculations from multiple models.
   *
   * \note `draft_token_prob` can be inferred from `draft_tokens` and
   * `draft_prob_dist`, but we still keep it so that we can have option
   * choosing only to use one between them.
   */
  std::vector<float> draft_output_token_prob;
  /*! \brief The appeared committed and draft tokens and their occurrence times. */
  std::unordered_map<int32_t, int32_t> appeared_token_ids;

  /*! \brief Return the total length of the input data. */
  int GetInputLength() const;
  /*! \brief Commit a new token into committed_tokens. Update appeared_token_ids. */
  void CommitToken(int32_t token_id);
  /*! \brief Add a draft token into draft_output_tokens. Update appeared_token_ids. */
  void AddDraftToken(int32_t token_id);
  /*! \brief Remove the last token from draft_output_tokens. Update appeared_token_ids. */
  void RemoveLastDraftToken();
  /*! \brief Remove all draft tokens from draft_output_tokens. Update appeared_token_ids. */
  void RemoveAllDraftTokens();

  static constexpr const char* _type_key = "mlc.serve.RequestModelState";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(RequestModelStateNode, Object);
};

class RequestModelState : public ObjectRef {
 public:
  explicit RequestModelState(Request request, int model_id, int64_t internal_id,
                             Array<Data> inputs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RequestModelState, ObjectRef, RequestModelStateNode);
};

class RequestStateNode : public Object {
 public:
  /*! \brief The request that this state corresponds to. */
  Request request;
  /*!
   * \brief The state with regard to each model.
   * \sa RequestModelState
   */
  Array<RequestModelState> mstates;
  /*! \brief The random number generator of this request. */
  RandomGenerator rng;
  /*!
   * \brief The start position of the committed tokens in the
   * next request stream callback invocation.
   */
  int next_callback_token_pos;

  /*! \brief The time of adding the request to engine. */
  std::chrono::high_resolution_clock::time_point tadd;
  /*! \brief The time of finishing prefill stage. */
  std::chrono::high_resolution_clock::time_point tprefill_finish;

  /*!
   * \brief Check if the request generation is finished and return the
   * finish reason if finished.
   * \param max_single_sequence_length The maximum allowed single sequence length.
   * \return The finish reason in string if the request is finished,
   * or None if the request has not finished.
   */
  Optional<String> GenerationFinished(int max_single_sequence_length) const;

  static constexpr const char* _type_key = "mlc.serve.RequestState";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(RequestStateNode, Object);
};

class RequestState : public ObjectRef {
 public:
  explicit RequestState(Request request, int num_models, int64_t internal_id);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RequestState, ObjectRef, RequestStateNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_REQUEST_STATE_H_
