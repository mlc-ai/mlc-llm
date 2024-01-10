/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.cc
 * \brief The implementation for runtime module of sampler functions.
 */
#define __STDC_FORMAT_MACROS

#include "sampler.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>

#include <cmath>

#include "../random.h"

namespace mlc {
namespace llm {
namespace serve {

/***** Utility function for in-place logits/prob update on CPU *****/

/*!
 * \brief In-place apply repetition penalty to logits based on history tokens.
 * \param logits The logits (a batch) to be in-place mutated.
 * \param token_offset The offset of the token in the batch
 * whose logits will be updated.
 * \param state The request state that contains history tokens.
 * \param repetition_penalty The value of repetition penalty.
 */
void ApplyRepetitionPenaltyOnCPU(NDArray logits, int token_offset, RequestModelState state,
                                 double repetition_penalty) {
  // logits: (n, v)
  CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  CHECK_EQ(logits->ndim, 2);
  CHECK_EQ(logits->device.device_type, DLDeviceType::kDLCPU);
  int vocab_size = logits->shape[1];

  float* logits_raw_data = static_cast<float*>(logits->data) + (token_offset * vocab_size);
  for (const auto& it : state->appeared_token_ids) {
    int token_id = it.first;
    ICHECK_GE(token_id, 0);
    ICHECK_LT(token_id, vocab_size);
    if (logits_raw_data[token_id] <= 0) {
      logits_raw_data[token_id] *= repetition_penalty;
    } else {
      logits_raw_data[token_id] /= repetition_penalty;
    }
  }
}

/*!
 * \brief In-place apply frequency and presence penalty to logits based on history tokens.
 * \param logits The logits (a batch) to be in-place mutated.
 * \param token_offset The offset of the token in the batch
 * whose logits will be updated.
 * \param state The request state that contains history tokens.
 * \param frequency_penalty The value of frequency penalty.
 * \param presence_penalty The value of presence penalty.
 */
void ApplyFrequencyAndPresencePenaltyOnCPU(NDArray logits, int token_offset,
                                           RequestModelState state, double frequency_penalty,
                                           double presence_penalty) {
  // logits: (n, v)
  CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  CHECK_EQ(logits->ndim, 2);
  CHECK_EQ(logits->device.device_type, DLDeviceType::kDLCPU);
  int vocab_size = logits->shape[1];

  float* logits_raw_data = static_cast<float*>(logits->data) + (token_offset * vocab_size);
  for (const auto& it : state->appeared_token_ids) {
    int token_id = it.first;
    int occurrences = it.second;
    ICHECK_GE(token_id, 0);
    ICHECK_LT(token_id, vocab_size);
    logits_raw_data[token_id] -= occurrences * frequency_penalty + presence_penalty;
  }
}

/*!
 * \brief In-place compute softmax with temperature on CPU.
 * \param logits The logits (a batch) to compute softmax from.
 * \param token_offset The offset of the token in the batch
 * to compute softmax for. Only the logits of the specified
 * token will be updated to probability after softmax.
 * \param temperature The temperature to apply before softmax.
 */
void ApplySoftmaxWithTemperatureOnCPU(NDArray logits, int token_offset, double temperature) {
  // logits: (n, v)
  CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  CHECK_EQ(logits->ndim, 2);
  CHECK_EQ(logits->device.device_type, DLDeviceType::kDLCPU);
  int vocab_size = logits->shape[1];

  float* __restrict logits_raw_data =
      static_cast<float*>(__builtin_assume_aligned(logits->data, 4)) + (token_offset * vocab_size);
  float m = std::numeric_limits<float>::min();
  float inv_temp = 1.0f / temperature;
  double d = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    float m_prev = m;
    m = std::max(m, x);
    d = d * std::exp(m_prev - m) + std::exp(x - m);
  }
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    logits_raw_data[i] = std::exp(x - m) / d;
  }
}

/*!
 * \brief In-place set probability via argmax.
 * This is used for zero-temperature sampling cases.
 * \param logits The logits (a batch) to set probability.
 * \param token_offset The offset of the token in the batch
 * to set probability for. Only the logits of the specified
 * token will be updated to probability.
 */
void SetProbWithArgmaxOnCPU(NDArray logits, int token_offset) {
  // logits: (n, v)
  CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  CHECK_EQ(logits->ndim, 2);
  CHECK_EQ(logits->device.device_type, kDLCPU);
  int vocab_size = logits->shape[1];

  float* logits_raw_data = static_cast<float*>(logits->data) + (token_offset * vocab_size);
  int argmax_pos = -1;
  float max_logits = std::numeric_limits<float>::lowest();
  for (int i = 0; i < vocab_size; ++i) {
    if (logits_raw_data[i] > max_logits) {
      max_logits = logits_raw_data[i];
      argmax_pos = i;
    }
  }

  ICHECK_NE(argmax_pos, -1);
  for (int i = 0; i < vocab_size; ++i) {
    logits_raw_data[i] = i == argmax_pos ? 1.0f : 0.0f;
  }
}

/*!
 * \brief Sample a value from the input probability distribution with top-p.
 * The input is a batch of distributions, and we use `unit_offset` to specify
 * which distribution to sample from.
 * \param prob The input batch of probability distributions.
 * \param unit_offset The offset specifying which distribution to sample from.
 * \param top_p The top-p value of sampling.
 * \param uniform_sample The random number in [0, 1] for sampling.
 * \param output_prob_dist Optional pointer to store the corresponding probability distribution of
 * each token, offset by unit_offset. If nullptr provided, nothing will be stored out.
 * \return The sampled prob and value.
 * \note This function is an enhancement of SampleTopPFromProb in TVM Unity.
 * We will upstream the enhancement after it gets stable.
 */
std::pair<float, int64_t> SampleTopPFromProb(NDArray prob, int unit_offset, double top_p,
                                             double uniform_sample,
                                             std::vector<NDArray>* output_prob_dist = nullptr) {
  // prob: (*, v)
  // The prob array may have arbitrary ndim and shape.
  // The last dimension corresponds to the prob distribution size.
  // We use the `unit_offset` parameter to determine which slice
  // of the prob array we sample from.

  ICHECK(prob.IsContiguous());
  ICHECK(prob.DataType() == DataType::Float(32));

  if (prob->device.device_type != kDLCPU) {
    prob = prob.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK(prob->device.device_type == kDLCPU);

  int64_t ndata = prob->shape[prob->ndim - 1];
  const float* __restrict p_prob =
      static_cast<float*>(__builtin_assume_aligned(prob->data, 4)) + (unit_offset * ndata);
  constexpr double one = 1.0f - 1e-5f;

  if (output_prob_dist) {
    if (!(*output_prob_dist)[unit_offset].defined()) {
      (*output_prob_dist)[unit_offset] = NDArray::Empty({ndata}, prob->dtype, DLDevice{kDLCPU, 0});
    }
    (*output_prob_dist)[unit_offset].CopyFromBytes(p_prob, ndata * sizeof(float));
  }

  if (top_p >= one) {
    // Specially handle case where top_p == 1.
    double prob_sum = 0.0f;
    for (int64_t i = 0; i < ndata; ++i) {
      prob_sum += p_prob[i];
      if (prob_sum >= uniform_sample) {
        return std::make_pair(p_prob[i], i);
      }
    }
    LOG(INFO) << "prob sum = " << prob_sum << ", sample = " << uniform_sample;
    ICHECK(false) << "Possibly prob distribution contains NAN.";
  }

  // Key observation: when we are doing top_p sampling
  // usually we only need to preserve some of the elements with
  // high probabilities before we do sort
  thread_local std::vector<std::pair<float, int>> data;

  auto sample_top_p_with_filter = [&](float cuttoff) -> std::pair<float, int64_t> {
    data.clear();
    // filter the data with cuttoff
    float cutoff_sum = 0.0f;
    for (int64_t i = 0; i < ndata; ++i) {
      if (p_prob[i] >= cuttoff) {
        cutoff_sum += p_prob[i];
        data.emplace_back(std::make_pair(p_prob[i], static_cast<int>(i)));
        if (cutoff_sum > 1 - cuttoff) {
          // Short cut. When the remaining parts cannot have total
          // probability larger than cutoff, we can quit.
          break;
        }
      }
    }
    if (data.size() == 0) return std::make_pair(-1, -1);
    auto fcmp = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
      return lhs.first > rhs.first;
    };
    std::sort(data.begin(), data.end(), fcmp);

    // short cut, if we know that
    // uniform sample < p[0] / top_p
    // we know that unform_sample < p[0] / top_p_sum
    // because top_p_sum guarantees to be smaller than top_p
    // so we can simply return the argmax sample
    // without computing anything
    if (uniform_sample < data[0].first / top_p) {
      return std::make_pair(data[0].first, data[0].second);
    }

    // compute top_p_sum
    float cum_sum_prob = 0.0f;
    float top_p_sum = 0.0f;
    for (auto it = data.begin(); it != data.end(); ++it) {
      float prob = it->first;
      if (cum_sum_prob < top_p) {
        top_p_sum += prob;
      } else {
        // we get to the right cutoff pt
        break;
      }
      cum_sum_prob += prob;
      it->first = cum_sum_prob;
    }
    // we find that the current total sum by the given cutoff
    // is not sufficient to cover everything
    // this means we might need to retry a smaller cutoff pt.
    if (cum_sum_prob < top_p && cuttoff != 0.0f) return std::make_pair(-1, -1);

    float last_cum_sum_prob = 0.0;
    for (auto it = data.begin(); it != data.end(); ++it) {
      if (uniform_sample < it->first / top_p_sum) {
        return std::make_pair(it->first - last_cum_sum_prob, it->second);
      }
      last_cum_sum_prob = it->first;
    }
    return std::make_pair(data[data.size() - 1].first - last_cum_sum_prob,
                          data[data.size() - 1].second);
  };

  if (top_p < 1) {
    // sample through cutoff by a number
    // by pigeonhole principle we will get at most 1024 elements
    // usually it is much less by applying this filtering(order of 10 - 20)
    data.reserve(256);
    std::pair<float, int64_t> sampled_index = sample_top_p_with_filter(top_p / 1024);
    if (sampled_index.second >= 0) return sampled_index;
  }
  // fallback via full prob, rare case
  data.reserve(ndata);
  std::pair<float, int64_t> sampled_index = sample_top_p_with_filter(0.0f);
  ICHECK_GE(sampled_index.second, 0);
  return sampled_index;
}

/*!
 * \brief Copy logits or prob distributions from device to CPU.
 * The input array is in layout (b, n, v).
 * This function flattens the first dimension, returns an NDArray
 * in shape (b * n, v).
 */
NDArray CopyLogitsOrProbsToCPU(NDArray arr_on_device, NDArray* arr_on_cpu) {
  // arr_on_device: (b, n, v)
  ICHECK_EQ(arr_on_device->ndim, 3);
  ICHECK(!arr_on_cpu->defined() || (*arr_on_cpu)->ndim == 2);
  ICHECK(arr_on_device->device.device_type != kDLCPU);
  if (arr_on_cpu->defined()) {
    ICHECK_EQ((*arr_on_cpu)->shape[1], arr_on_device->shape[2]);
  }

  int64_t init_size = arr_on_cpu->defined() ? (*arr_on_cpu)->shape[0] : 32;
  int64_t num_tokens = arr_on_device->shape[0] * arr_on_device->shape[1];
  int64_t vocab_size = arr_on_device->shape[2];
  while (init_size < num_tokens) {
    init_size *= 2;
  }
  if (!arr_on_cpu->defined() || init_size != (*arr_on_cpu)->shape[0]) {
    (*arr_on_cpu) =
        NDArray::Empty({init_size, vocab_size}, arr_on_device->dtype, DLDevice{kDLCPU, 0});
  }
  ICHECK_LE(num_tokens, (*arr_on_cpu)->shape[0]);
  NDArray view = arr_on_cpu->CreateView({num_tokens, vocab_size}, arr_on_device->dtype);
  view.CopyFrom(arr_on_device);
  return view;
}

/********************* CPU Sampler *********************/

class CPUSampler : public SamplerObj {
 public:
  explicit CPUSampler(Optional<EventTraceRecorder> trace_recorder)
      : trace_recorder_(std::move(trace_recorder)), rng_(RandomGenerator::GetInstance()) {
    // Set customized "logits -> prob" function.
    const PackedFunc* f_logits_to_probs =
        Registry::Get("mlc.llm.compute_probs_from_logits_inplace");
    if (f_logits_to_probs != nullptr) {
      flogits_to_probs_inplace_ = *f_logits_to_probs;
    }
  }

  std::vector<int32_t> BatchSampleTokens(NDArray logits_on_device, Model model,
                                         Array<RequestModelState> request_mstates,
                                         Array<GenerationConfig> generation_cfg,
                                         std::vector<NDArray>* output_prob_dist,
                                         std::vector<float>* output_token_probs) final {
    NDArray probs_on_cpu = BatchComputeProb(logits_on_device, /*cum_sequence_length=*/nullptr,
                                            model, request_mstates, generation_cfg);
    // - Sample tokens from probabilities.
    // NOTE: Though we have the probability field in RequestModelState,
    //       we do not save the probabilities right now.
    //       We will handle this in the future when we work on speculation.
    std::vector<int32_t> output_tokens = SampleTokensFromProbs(
        probs_on_cpu, request_mstates, generation_cfg, output_prob_dist, output_token_probs);
    return output_tokens;
  }

  std::vector<std::vector<int32_t>> BatchVerifyDraftTokens(
      NDArray logits_on_device, const std::vector<int>& cum_verify_lengths, Model model,
      const Array<RequestModelState>& request_mstates,
      const Array<GenerationConfig>& generation_cfg,
      const std::vector<std::vector<int>>& draft_output_tokens,
      const std::vector<std::vector<float>>& draft_output_token_prob,
      const std::vector<std::vector<NDArray>>& draft_output_prob_dist) final {
    bool can_compute_prob_in_parallel = CanComputeProbInParallel(generation_cfg);
    NDArray logits_or_probs_on_cpu{nullptr};
    Array<String> request_ids =
        request_mstates.Map([](const RequestModelState& mstate) { return mstate->request->id; });
    if (can_compute_prob_in_parallel) {
      logits_or_probs_on_cpu = BatchComputeProb(logits_on_device, &cum_verify_lengths, model,
                                                request_mstates, generation_cfg);
    } else {
      RECORD_EVENT(trace_recorder_, request_ids, "start copy logits to CPU");
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(logits_on_device, &logits_or_probs_on_cpu_);
      RECORD_EVENT(trace_recorder_, request_ids, "finish copy logits to CPU");
    }
    ICHECK(logits_or_probs_on_cpu->device.device_type == kDLCPU);
    ICHECK_EQ(logits_or_probs_on_cpu->ndim, 2);

    int num_sequence = static_cast<int>(cum_verify_lengths.size()) - 1;
    CHECK_EQ(draft_output_tokens.size(), num_sequence);
    CHECK_EQ(draft_output_token_prob.size(), num_sequence);
    CHECK_EQ(draft_output_prob_dist.size(), num_sequence);

    std::vector<std::vector<int>> accepted_tokens;
    std::vector<double> random_numbers;
    accepted_tokens.resize(num_sequence);
    random_numbers.reserve(num_sequence * 2);
    for (int i = 0; i < num_sequence; ++i) {
      random_numbers.push_back(rng_.GetRandomNumber());
      random_numbers.push_back(rng_.GetRandomNumber());
    }

    float* __restrict global_p_probs =
        static_cast<float*>(__builtin_assume_aligned(logits_or_probs_on_cpu->data, 4));
    int vocab_size = logits_or_probs_on_cpu->shape[1];

    tvm::runtime::parallel_for_with_threading_backend(
        [&](int i) {
          int verify_start = cum_verify_lengths[i];
          int verify_end = cum_verify_lengths[i + 1];
          for (int cur_token_idx = 0; cur_token_idx < verify_end - verify_start; ++cur_token_idx) {
            if (!can_compute_prob_in_parallel) {
              SinglePosComputeProbsFromLogitsInplace(logits_or_probs_on_cpu,
                                                     verify_start + cur_token_idx,
                                                     request_mstates[i], generation_cfg[i]);
            }

            float* p_probs = global_p_probs + (verify_start + cur_token_idx) * vocab_size;
            int cur_token = draft_output_tokens[i][cur_token_idx];
            float q_value = draft_output_token_prob[i][cur_token_idx];
            float p_value = p_probs[cur_token];

            if (p_value >= q_value) {
              request_mstates[i]->CommitToken(cur_token);
              accepted_tokens[i].push_back(cur_token);
              continue;
            }
            float r = random_numbers[i * 2];
            if (r < p_value / (q_value + eps_)) {
              request_mstates[i]->CommitToken(cur_token);
              accepted_tokens[i].push_back(cur_token);
              continue;
            }

            // normalize a new probability distribution
            double sum_v = 0.0;
            NDArray q_dist = draft_output_prob_dist[i][cur_token_idx];
            ICHECK(q_dist->device.device_type == kDLCPU);
            ICHECK(q_dist->ndim == 1);
            ICHECK(vocab_size == q_dist->shape[q_dist->ndim - 1]);
            const float* __restrict p_qdist =
                static_cast<float*>(__builtin_assume_aligned(q_dist->data, 4));

            for (int j = 0; j < vocab_size; ++j) {
              p_probs[j] = std::max(p_probs[j] - p_qdist[j], 0.0f);
              sum_v += p_probs[j];
            }
            for (int j = 0; j < vocab_size; ++j) {
              p_probs[j] /= sum_v;
            }

            // sample a new token from the new distribution
            int32_t new_token =
                SampleTopPFromProb(logits_or_probs_on_cpu, verify_start + cur_token_idx,
                                   generation_cfg[i]->top_p, random_numbers[i * 2 + 1])
                    .second;
            request_mstates[i]->CommitToken(new_token);
            accepted_tokens[i].push_back(cur_token);
            break;
          }
        },
        0, num_sequence);
    return accepted_tokens;
  }

 private:
  /*!
   * \brief Given the generation config of a batch, check if the
   * probability distributions needs to be computed on device via softmax.
   * \param generation_cfg The input generation config.
   * \return A boolean flag indicating if the check result.
   */
  bool RequireGPUSoftmax(Array<GenerationConfig> generation_cfg) {
    // - Return false if there is customized probability compute function.
    if (flogits_to_probs_inplace_.defined()) {
      return false;
    }
    // - Return false if any sampling param has frequency/presence penalty other than 0.0.
    // - Return false if any sampling param has repetition penalty other than 1.0.
    // - Return false if any sampling param has zero temperature.
    for (GenerationConfig cfg : generation_cfg) {
      if (cfg->frequency_penalty != 0.0 || cfg->presence_penalty != 0.0 ||
          cfg->repetition_penalty != 1.0 || cfg->temperature < 1e-6) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Given the generation config of a batch, check if the
   * probability distributions need to be computed serially.
   */
  bool CanComputeProbInParallel(const Array<GenerationConfig>& generation_cfg) {
    for (const GenerationConfig& cfg : generation_cfg) {
      if (cfg->frequency_penalty != 0.0 || cfg->presence_penalty != 0.0 ||
          cfg->repetition_penalty != 1.0) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Compute the probability distribution of the input logits.
   * \param logits_on_device The logits to compute probability distribution for.
   * \param model The LLM model which contains the softmax
   * function on device that might be used to compute probability distribution.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \return The probability distribution of the input logits.
   */
  NDArray BatchComputeProb(NDArray logits_on_device, const std::vector<int>* cum_sequence_length,
                           Model model, const Array<RequestModelState>& request_mstates,
                           const Array<GenerationConfig>& generation_cfg) {
    ICHECK(logits_on_device.defined());
    ICHECK_EQ(logits_on_device->ndim, 3);
    int num_sequence;
    if (cum_sequence_length == nullptr) {
      ICHECK_EQ(logits_on_device->shape[1], 1)
          << "Multi-token sampling for one sequence requiring `cum_sequence_length`.";
      num_sequence = logits_on_device->shape[0];
    } else {
      ICHECK(!cum_sequence_length->empty());
      num_sequence = static_cast<int>(cum_sequence_length->size()) - 1;
      ICHECK_EQ(logits_on_device->shape[0], 1);
      ICHECK_EQ(logits_on_device->shape[1], cum_sequence_length->back());
    }
    ICHECK_EQ(generation_cfg.size(), num_sequence);
    ICHECK_EQ(request_mstates.size(), num_sequence);

    Array<String> request_ids =
        request_mstates.Map([](const RequestModelState& mstate) { return mstate->request->id; });

    RECORD_EVENT(trace_recorder_, request_ids, "start query need GPU softmax");
    bool require_gpu_softmax = RequireGPUSoftmax(generation_cfg);
    RECORD_EVENT(trace_recorder_, request_ids, "finish query need GPU softmax");

    // - Compute probabilities from logits.
    NDArray logits_or_probs_on_cpu{nullptr};
    if (require_gpu_softmax) {
      RECORD_EVENT(trace_recorder_, request_ids, "start GPU softmax");
      Array<GenerationConfig> generation_cfg_for_softmax;
      if (cum_sequence_length == nullptr) {
        generation_cfg_for_softmax = generation_cfg;
      } else {
        logits_on_device = logits_on_device.CreateView(
            {logits_on_device->shape[1], 1, logits_on_device->shape[2]}, logits_on_device->dtype);
        generation_cfg_for_softmax.reserve(logits_on_device->shape[1]);
        for (int i = 0; i < num_sequence; ++i) {
          for (int pos = cum_sequence_length->at(i); pos < cum_sequence_length->at(i + 1); ++pos) {
            generation_cfg_for_softmax.push_back(generation_cfg[i]);
          }
        }
      }
      NDArray probs_on_device =
          model->SoftmaxWithTemperature(logits_on_device, generation_cfg_for_softmax);
      RECORD_EVENT(trace_recorder_, request_ids, "finish GPU softmax");
      RECORD_EVENT(trace_recorder_, request_ids, "start copy probs to CPU");
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(probs_on_device, &logits_or_probs_on_cpu_);
      RECORD_EVENT(trace_recorder_, request_ids, "finish copy probs to CPU");
    } else {
      RECORD_EVENT(trace_recorder_, request_ids, "start copy logits to CPU");
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(logits_on_device, &logits_or_probs_on_cpu_);
      RECORD_EVENT(trace_recorder_, request_ids, "finish copy logits to CPU");
      // The "BatchComputeProbsFromLogitsInplace" function updates
      // `logits_or_probs_on_cpu` in place.
      BatchComputeProbsFromLogitsInplace(logits_or_probs_on_cpu, cum_sequence_length,
                                         std::move(request_mstates), generation_cfg);
    }
    // `CopyLogitsOrProbsToCPU` flattens the first two dimensions.
    ICHECK_EQ(logits_or_probs_on_cpu->ndim, 2);
    return logits_or_probs_on_cpu;
  }

  /*!
   * \brief Compute the probability distribution from on-cpu logits for
   * a batch of tokens **in place**.
   * \param logits The input logits on CPU.
   * \param states The request states, which contains the history generated tokens.
   * \param generation_cfg The generation config.
   * \note The function returns nothing. It in-place updates the input logits array.
   */
  void BatchComputeProbsFromLogitsInplace(NDArray logits,
                                          const std::vector<int>* cum_sequence_length,
                                          Array<RequestModelState> states,
                                          Array<GenerationConfig> generation_cfg) {
    // logits: (n, v)
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->device.device_type, kDLCPU);

    // - Invoke environment compute function if exists.
    if (flogits_to_probs_inplace_.defined()) {
      IntTuple cum_sequence_length_obj;
      if (cum_sequence_length != nullptr) {
        cum_sequence_length_obj =
            IntTuple{cum_sequence_length->begin(), cum_sequence_length->end()};
      }
      flogits_to_probs_inplace_(logits, cum_sequence_length_obj, states, generation_cfg);
      return;
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &logits, cum_sequence_length, &states, &generation_cfg](int i) {
          int offset_start = cum_sequence_length == nullptr ? i : cum_sequence_length->at(i);
          int offset_end = cum_sequence_length == nullptr ? i + 1 : cum_sequence_length->at(i + 1);
          for (int offset = offset_start; offset < offset_end; ++offset) {
            SinglePosComputeProbsFromLogitsInplace(logits, offset, states[i], generation_cfg[i]);
          }
        },
        0, logits->shape[0]);
  }

  void SinglePosComputeProbsFromLogitsInplace(NDArray logits, int offset,
                                              const RequestModelState& state,
                                              const GenerationConfig& generation_cfg) {
    // - Apply frequency/presence penalty or repetition penalty (inplace).
    if (generation_cfg->frequency_penalty != 0.0 || generation_cfg->presence_penalty != 0.0) {
      RECORD_EVENT(trace_recorder_, state->request->id, "start frequency/presence penalty");
      ApplyFrequencyAndPresencePenaltyOnCPU(logits, offset, state,
                                            generation_cfg->frequency_penalty,
                                            generation_cfg->presence_penalty);
      RECORD_EVENT(trace_recorder_, state->request->id, "finish frequency/presence penalty");
    } else if (generation_cfg->repetition_penalty != 1.0) {
      RECORD_EVENT(trace_recorder_, state->request->id, "start repetition penalty");
      ApplyRepetitionPenaltyOnCPU(logits, offset, state, generation_cfg->repetition_penalty);
      RECORD_EVENT(trace_recorder_, state->request->id, "finish repetition penalty");
    }
    // - Compute probability (inplace) from logits.
    //   Using softmax if temperature is non-zero.
    //   Or set probability of the max-logit position to 1.
    if (generation_cfg->temperature >= 1e-6) {
      RECORD_EVENT(trace_recorder_, state->request->id, "start CPU softmax");
      ApplySoftmaxWithTemperatureOnCPU(logits, offset, generation_cfg->temperature);
      RECORD_EVENT(trace_recorder_, state->request->id, "finish CPU softmax");
    } else {
      RECORD_EVENT(trace_recorder_, state->request->id, "start argmax");
      SetProbWithArgmaxOnCPU(logits, offset);
      RECORD_EVENT(trace_recorder_, state->request->id, "finish argmax");
    }
  }

  std::vector<int32_t> SampleTokensFromProbs(NDArray probs,
                                             Array<RequestModelState> request_mstates,
                                             Array<GenerationConfig> generation_cfg,
                                             std::vector<NDArray>* output_prob_dist,
                                             std::vector<float>* output_token_probs) {
    // probs: (n, v)
    CHECK_EQ(probs->ndim, 2);
    CHECK_EQ(probs->device.device_type, kDLCPU);

    Array<String> request_ids =
        request_mstates.Map([](const RequestModelState& mstate) { return mstate->request->id; });

    int n = probs->shape[0];
    std::vector<double> random_numbers;
    std::vector<int32_t> sampled_tokens;
    random_numbers.reserve(n);
    sampled_tokens.resize(n);
    if (output_prob_dist) {
      output_prob_dist->resize(n);
    }
    if (output_token_probs) {
      output_token_probs->resize(n);
    }
    for (int i = 0; i < n; ++i) {
      random_numbers.push_back(rng_.GetRandomNumber());
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &sampled_tokens, &probs, &generation_cfg, &random_numbers, &request_ids,
         output_prob_dist, output_token_probs](int i) {
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "start sample token");
          // Sample top p from probability.
          std::pair<float, int64_t> sample_result = SampleTopPFromProb(
              probs, i, generation_cfg[i]->top_p, random_numbers[i], output_prob_dist);
          sampled_tokens[i] = sample_result.second;
          if (output_token_probs) {
            (*output_token_probs)[i] = sample_result.first;
          }
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "finish sample token");
        },
        0, n);
    return sampled_tokens;
  }

  /*! \brief The event trace recorder for requests. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief The random generator. */
  RandomGenerator& rng_;
  /*! \brief Customized function which computes prob distribution from logits */
  PackedFunc flogits_to_probs_inplace_;
  /*! \brief Shared array for logits and probability distributions on cpu. */
  NDArray logits_or_probs_on_cpu_{nullptr};
  const float eps_ = 1e-9;
};

/*********************** Sampler ***********************/

TVM_REGISTER_OBJECT_TYPE(SamplerObj);

Sampler Sampler::Create(std::string sampler_kind, Optional<EventTraceRecorder> trace_recorder) {
  if (sampler_kind == "cpu") {
    return Sampler(make_object<CPUSampler>(std::move(trace_recorder)));
  } else {
    LOG(FATAL) << "Unsupported sampler_kind \"" << sampler_kind << "\"";
    throw;
  }
}

namespace detail {

// The detailed implementation of `parallel_for_with_threading_backend`.
// To avoid template expansion, the implementation cannot be placed
// in .cc files.

template <typename T>
struct ParallelForWithThreadingBackendLambdaInvoker {
  static int TVMParallelLambdaInvoke(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
    int num_task = penv->num_task;
    // Convert void* back to lambda type.
    T* lambda_ptr = static_cast<T*>(cdata);
    // Invoke the lambda with the task id (thread id).
    (*lambda_ptr)(task_id, num_task);
    return 0;
  }
};

template <typename T>
inline void parallel_launch_with_threading_backend(T flambda) {
  // Launch the lambda by passing its address.
  void* cdata = &flambda;
  TVMBackendParallelLaunch(ParallelForWithThreadingBackendLambdaInvoker<T>::TVMParallelLambdaInvoke,
                           cdata, /*num_task=*/0);
}

}  // namespace detail

template <typename T>
inline void parallel_for_with_threading_backend(T flambda, int64_t begin, int64_t end) {
  auto flaunch = [begin, end, flambda](int task_id, int num_task) {
    // For each thread, do static division and call into flambda.
    int64_t total_len = end - begin;
    int64_t step = (total_len + num_task - 1) / num_task;
    int64_t local_begin = std::min(begin + step * task_id, end);
    int64_t local_end = std::min(local_begin + step, end);
    for (int64_t i = local_begin; i < local_end; ++i) {
      flambda(i);
    }
  };
  // Launch with all threads.
  detail::parallel_launch_with_threading_backend(flaunch);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
