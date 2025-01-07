/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_prefill_base.h
 */

#include <tvm/runtime/nvtx.h>

#include "../config.h"
#include "../model.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The base action of that prefills requests in the `waiting_queue` of
 * the engine state.
 */
class BatchPrefillBaseActionObj : public EngineActionObj {
 protected:
  /*! \brief The class of request state entry and its maximum allowed length for prefill. */
  struct PrefillInput {
    RequestStateEntry rsentry;
    int max_prefill_length = 0;
    int num_child_to_activate = 0;
    bool is_decode = false;
  };

  BatchPrefillBaseActionObj(Array<Model> models, EngineConfig engine_config,
                            std::vector<picojson::object> model_configs,
                            Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Find one or multiple request state entries to run prefill.
   * \param estate The engine state.
   * \return The request entries to prefill, together with their input lengths.
   */
  std::vector<PrefillInput> GetRequestStateEntriesToPrefill(EngineState estate);

  /*! \brief Check if the input requests can be prefilled under conditions. */
  bool CanPrefill(EngineState estate, int num_prefill_rsentries, int total_input_length,
                  int num_required_pages, int num_available_pages, int current_total_seq_len,
                  int num_running_rsentries, KVStateKind kv_state_kind,
                  bool sliding_window_enabled);

  /*!
   * \brief Chunk the input of the given RequestModelState for prefill
   * with regard to the provided maximum allowed prefill length.
   * Return the list of input for prefill and the total prefill length.
   * The `inputs` field of the given `mstate` will be mutated to exclude
   * the returned input.
   * \param mstate The RequestModelState whose input data is to be chunked.
   * \param max_prefill_length The maximum allowed prefill length for the mstate.
   * \return The list of input for prefill and the total prefill length.
   */
  std::pair<Array<Data>, int> ChunkPrefillInputData(const RequestModelState& mstate,
                                                    int max_prefill_length);

  /*!
   * \brief Update status of request states from pending to alive and collect request state entries
   * from the prefill input.
   * \param prefill_inputs The prefill input.
   * \param estate The engine state.
   * \param[out] request_ids The array to store the request ids of the request state entries.
   * \param[out] rstates_of_entries The vector to store the request state entries.
   * \param[out] status_before_prefill The vector to store the status of the request state entries
   * before prefill.
   */
  void UpdateRequestToAlive(const std::vector<PrefillInput>& prefill_inputs,
                            const EngineState& estate, Array<String>* request_ids,
                            std::vector<RequestState>* rstates_of_entries,
                            std::vector<RequestStateStatus>* status_before_prefill);

  /*!
   * \brief Remove the request from waiting queue if all its request states are now alive and have
   * no remaining chunked inputs.
   * \param prefill_inputs The prefill input.
   * \param estate The engine state.
   * \param rstates_of_entries The request state entries for each prefill input.
   * \return The processed requests.
   */
  std::vector<Request> RemoveProcessedRequests(const std::vector<PrefillInput>& prefill_inputs,
                                               const EngineState& estate,
                                               const std::vector<RequestState>& rstates_of_entries);

  /*!
   * \brief Update the committed tokens of states. If a request is first-time prefilled, set the
   * prefill finish time.
   * \param rsentries_for_sample The request state entries for sample.
   * \param rsentry_activated The activation status of the request state entries.
   * \param sample_results The sample results.
   */
  void UpdateRequestStateEntriesWithSampleResults(
      const std::vector<RequestStateEntry>& rsentries_for_sample,
      const std::vector<bool>& rsentry_activated, const std::vector<SampleResult>& sample_results);

  /*!
   * \brief Get the concatenated IntTuple of RequestModelState input data, return empty IntTuple if
   * there is untokenized data.
   * \param mstate The RequestModelState whose input data is to be concatenated.
   * \return The concatenate IntTuple.
   */
  std::vector<int32_t> GetConcatPrefillInputData(const RequestModelState& mstate);

  /*!
   * \brief Pop the prefix tokens of the RequestModelState input data array.
   * \param mstate The RequestModelState to be popped.
   * \param num_tokens The number of prefix tokens to be popped.
   */
  void PopPrefillInputData(const RequestModelState& mstate, size_t num_tokens);

  /*!
   * \brief Match the request state entry with prefix cache, to skip prefilling common prefix
   * tokens. If the request state entry is not added to KVCache yet, this method will add/fork the
   * request in the KVCache, depending on the matching result from prefix cache.
   * \param estate The engine state.
   * \param[in, out] input The prefill input to be matched and updated.
   * \return The matched length in prefix cache.
   */
  virtual int MatchPrefixCache(EngineState estate, PrefillInput* input) = 0;

  /*! \brief The models to run prefill in. */
  Array<Model> models_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief The KV state kind. */
  KVStateKind kv_state_kind_;
  /*! \brief The sliding window size of each model. */
  std::vector<int> sliding_window_sizes_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
};

/*!
 * \brief A utility function to check whether there is enough spare space in
 * KV cache for the number of required pages and total input length.
 */
bool HasPrefillSpace(int num_required_pages, bool sliding_window_enabled, int new_batch_size,
                     int num_available_pages, int current_total_seq_len, int total_input_length,
                     int max_total_sequence_length);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
