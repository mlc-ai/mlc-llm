/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/logit_processor.h
 * \brief The header for logit processor.
 */

#ifndef MLC_LLM_SERVE_LOGIT_PROCESSOR_H_
#define MLC_LLM_SERVE_LOGIT_PROCESSOR_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"
#include "config.h"
#include "event_trace_recorder.h"
#include "function_table.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief The logit processor class that updates logits with regard
 * presence/frequency penalties, logit bias, etc..
 */
class LogitProcessorObj : public Object {
 public:
  /*!
   * \brief In-place update a batch of logits with regard to the given
   * generation config and request states.
   * \param logits The batch of raw logits, in shape (num_total_token, vocab_size),
   * where `num_total_token` may be larger than the number of sequences
   * indicated by `generation_cfg`, in which case some sequences may have
   * more than one token.
   * \param generation_cfg The generation config of each sequence in the batch.
   * \param mstates The request states of each sequence in the batch.
   * \param request_ids The ids of each request.
   * \param cum_num_token The pointer to the cumulative token length of the sequences.
   * If the pointer is nullptr, it means each sequence has only one token.
   * \param draft_tokens The pointer to the draft tokens of each sequence
   * when speculation is enabled, in which case some sequences may have
   * more than one token.
   */
  virtual void InplaceUpdateLogits(
      NDArray logits, const Array<GenerationConfig>& generation_cfg,
      const Array<RequestModelState>& mstates, const Array<String>& request_ids,
      const std::vector<int>* cum_num_token = nullptr,
      const std::vector<std::vector<SampleResult>>* draft_tokens = nullptr) = 0;

  /*!
   * \brief Compute probability distributions for the input batch of logits.
   * \param logits The batch of updated logits.
   * \param generation_cfg The generation config of each sequence in the batch.
   * \param request_ids The ids of each request.
   * \param cum_num_token The pointer to the cumulative token length of the sequences.
   * If the pointer is nullptr, it means each sequence has only one token.
   * \return The batch of computed probability distributions on GPU.
   */
  virtual NDArray ComputeProbsFromLogits(NDArray logits,
                                         const Array<GenerationConfig>& generation_cfg,
                                         const Array<String>& request_ids,
                                         const std::vector<int>* cum_num_token = nullptr) = 0;

  static constexpr const char* _type_key = "mlc.serve.LogitProcessor";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(LogitProcessorObj, Object);
};

class LogitProcessor : public ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param max_num_token The max number of tokens in the token processor.
   * \param vocab_size The model's vocabulary size.
   * \param ft The packed function table.
   * \param device The device that the model runs on.
   * \param trace_recorder The event trace recorder.
   */
  explicit LogitProcessor(int max_num_token, int vocab_size, FunctionTable* ft, DLDevice device,
                          Optional<EventTraceRecorder> trace_recorder);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LogitProcessor, ObjectRef, LogitProcessorObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_LOGIT_PROCESSOR_H_
