/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_stats.h
 */
#ifndef MLC_LLM_SERVE_ENGINE_STATS_H_
#define MLC_LLM_SERVE_ENGINE_STATS_H_

#include <tvm/runtime/container/string.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief Runtime statistics of engine. */
struct EngineStats {
  /*! \brief The current total sequence length in the first model. */
  int64_t current_total_seq_len;
  /*! \brief The sum of "prefill time of each request". */
  double request_total_prefill_time = 0.0f;
  /*! \brief The sum of "decode time of each request". */
  double request_total_decode_time = 0.0f;
  /*! \brief The total engine time on prefill. */
  double engine_total_prefill_time = 0.0f;
  /*! \brief The total engine time on decode. */
  double engine_total_decode_time = 0.0f;
  /*! \brief The total number of processed tokens in prefill. */
  int64_t total_prefill_length = 0;
  /*! \brief The total number of processed tokens in decode. */
  int64_t total_decode_length = 0;

  /*!
   * \brief Return the engine runtime statistics in JSON string.
   * We collect the following entries:
   * - single token prefill latency (s/tok): avg latency of processing one token in prefill
   * - single token decode latency (s/tok): avg latency of processing one token in decode
   * - engine time for prefill (sec)
   * - engine time for decode (sec)
   * - total number of processed tokens in prefill.
   * - total number of processed tokens in decode.
   * \return The statistics in JSON string.
   */
  String AsJSON() const;
  /*! \brief Reset all the statistics. */
  void Reset();
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_STATS_H_
