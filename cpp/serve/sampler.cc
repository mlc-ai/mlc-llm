/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.cc
 * \brief The implementation for runtime module of sampler functions.
 */
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
 * \return The sampled value and probability.
 * \note This function is an enhancement of SampleTopPFromProb in TVM Unity.
 * We will upstream the enhancement after it gets stable.
 */
TokenProbPair SampleTopPFromProb(NDArray prob, int unit_offset, double top_p, double uniform_sample,
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
  }

  if (top_p == 0) {
    // Specially handle case where top_p == 0.
    // This case is equivalent to doing argmax.
    int argmax_pos = -1;
    float max_prob = 0.0;
    float sum_prob = 0.0;
    for (int i = 0; i < ndata; ++i) {
      if (p_prob[i] > max_prob) {
        max_prob = p_prob[i];
        argmax_pos = i;
      }
      // Early exit.
      sum_prob += p_prob[i];
      if (1 - sum_prob <= max_prob) {
        break;
      }
    }
    if (output_prob_dist) {
      float* __restrict p_output_prob =
          static_cast<float*>(__builtin_assume_aligned((*output_prob_dist)[unit_offset]->data, 4));
      for (int i = 0; i < ndata; ++i) {
        p_output_prob[i] = i == argmax_pos ? 1.0 : 0.0;
      }
    }
    return {argmax_pos, 1.0};
  }

  if (output_prob_dist) {
    (*output_prob_dist)[unit_offset].CopyFromBytes(p_prob, ndata * sizeof(float));
  }

  if (top_p >= one) {
    // Specially handle case where top_p == 1.
    double prob_sum = 0.0f;
    for (int64_t i = 0; i < ndata; ++i) {
      prob_sum += p_prob[i];
      if (prob_sum >= uniform_sample) {
        return {i, p_prob[i]};
      }
    }
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
    if (sampled_index.second >= 0) return {sampled_index.second, sampled_index.first};
  }
  // fallback via full prob, rare case
  data.reserve(ndata);
  std::pair<float, int64_t> sampled_index = sample_top_p_with_filter(0.0f);
  ICHECK_GE(sampled_index.second, 0);
  return {sampled_index.second, sampled_index.first};
}

namespace detail {

/*! \brief Implementation of getting top probs on CPU. */
template <int num_top_probs>
std::vector<TokenProbPair> ComputeTopProbsImpl(const float* p_prob, int ndata) {
  std::vector<TokenProbPair> top_probs;
  top_probs.reserve(num_top_probs);
  for (int i = 0; i < num_top_probs; ++i) {
    top_probs.emplace_back(-1, -1.0f);
  }

  float sum_prob = 0.0;
  // Selection argsort.
  for (int p = 0; p < ndata; ++p) {
    int i = num_top_probs - 1;
    for (; i >= 0; --i) {
      if (p_prob[p] > top_probs[i].second) {
        if (i != num_top_probs - 1) {
          top_probs[i + 1] = top_probs[i];
        }
      } else {
        break;
      }
    }
    if (i != num_top_probs - 1) {
      top_probs[i + 1] = {p, p_prob[p]};
    }

    // Early exit.
    sum_prob += p_prob[p];
    if (1 - sum_prob <= top_probs[num_top_probs - 1].second) {
      break;
    }
  }
  return top_probs;
}

}  // namespace detail

/*! \brief Get the probs of a few number of tokens with top probabilities. */
inline std::vector<TokenProbPair> ComputeTopProbs(NDArray prob, int unit_offset,
                                                  int num_top_probs) {
  ICHECK_LE(num_top_probs, 5);
  ICHECK_EQ(prob->ndim, 2);
  int ndata = prob->shape[1];
  const float* __restrict p_prob =
      static_cast<float*>(__builtin_assume_aligned(prob->data, 4)) + (unit_offset * ndata);
  switch (num_top_probs) {
    case 0:
      return {};
    case 1:
      return detail::ComputeTopProbsImpl<1>(p_prob, ndata);
    case 2:
      return detail::ComputeTopProbsImpl<2>(p_prob, ndata);
    case 3:
      return detail::ComputeTopProbsImpl<3>(p_prob, ndata);
    case 4:
      return detail::ComputeTopProbsImpl<4>(p_prob, ndata);
    case 5:
      return detail::ComputeTopProbsImpl<5>(p_prob, ndata);
  }
  throw;
}

/********************* CPU Sampler *********************/

class CPUSampler : public SamplerObj {
 public:
  explicit CPUSampler(Optional<EventTraceRecorder> trace_recorder)
      : trace_recorder_(std::move(trace_recorder)) {
    // Set customized "logits -> prob" function.
    const PackedFunc* f_logits_to_probs =
        Registry::Get("mlc.llm.compute_probs_from_logits_inplace");
    if (f_logits_to_probs != nullptr) {
      flogits_to_probs_inplace_ = *f_logits_to_probs;
    }
  }

  std::vector<SampleResult> BatchSampleTokens(NDArray probs_on_device,                        //
                                              const std::vector<int>& sample_indices,         //
                                              const Array<String>& request_ids,               //
                                              const Array<GenerationConfig>& generation_cfg,  //
                                              const std::vector<RandomGenerator*>& rngs,      //
                                              std::vector<NDArray>* output_prob_dist) final {
    // probs_on_device: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start sampling");
    CHECK_EQ(probs_on_device->ndim, 2);
    // - Copy probs to CPU
    RECORD_EVENT(trace_recorder_, request_ids, "start copy probs to CPU");
    NDArray probs_host = CopyProbsToCPU(probs_on_device);
    RECORD_EVENT(trace_recorder_, request_ids, "finish copy probs to CPU");

    // - Sample tokens from probabilities.
    int n = request_ids.size();
    ICHECK_EQ(generation_cfg.size(), n);
    ICHECK_EQ(rngs.size(), n);

    std::vector<SampleResult> sample_results;
    sample_results.resize(n);
    if (output_prob_dist) {
      output_prob_dist->resize(n);
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &sample_results, &probs_host, &generation_cfg, &rngs, &request_ids, sample_indices,
         output_prob_dist](int i) {
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "start sample token");
          // Sample top p from probability.
          sample_results[i].sampled_token_id = SampleTopPFromProb(
              probs_host, sample_indices[i],
              generation_cfg[i]->temperature < eps_ ? 0.0 : generation_cfg[i]->top_p,
              rngs[i]->GetRandomNumber(), output_prob_dist);
          if (output_prob_dist == nullptr) {
            // When `output_prob_dist` is not nullptr, it means right now
            // we are sampling for a small model in speculation, in which
            // case we do not need to get the top probs.
            sample_results[i].top_prob_tokens =
                ComputeTopProbs(probs_host, i, generation_cfg[i]->top_logprobs);
          }
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "finish sample token");
        },
        0, n);
    RECORD_EVENT(trace_recorder_, request_ids, "finish sampling");
    return sample_results;
  }

  std::vector<std::vector<SampleResult>> BatchVerifyDraftTokens(
      NDArray probs_on_device, const Array<String>& request_ids,
      const std::vector<int>& cum_verify_lengths, const Array<GenerationConfig>& generation_cfg,
      const std::vector<RandomGenerator*>& rngs,
      const std::vector<std::vector<SampleResult>>& draft_output_tokens,
      const std::vector<std::vector<NDArray>>& draft_output_prob_dist) final {
    // probs_on_device: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start draft verification");
    CHECK_EQ(probs_on_device->ndim, 2);
    // - Copy probs to CPU
    RECORD_EVENT(trace_recorder_, request_ids, "start copy probs to CPU");
    NDArray probs_host = CopyProbsToCPU(probs_on_device);
    RECORD_EVENT(trace_recorder_, request_ids, "finish copy probs to CPU");

    int num_sequence = static_cast<int>(cum_verify_lengths.size()) - 1;
    CHECK_EQ(rngs.size(), num_sequence);
    CHECK_EQ(draft_output_tokens.size(), num_sequence);
    CHECK_EQ(draft_output_prob_dist.size(), num_sequence);

    std::vector<std::vector<SampleResult>> sample_results;
    sample_results.resize(num_sequence);

    float* __restrict global_p_probs =
        static_cast<float*>(__builtin_assume_aligned(probs_host->data, 4));
    int vocab_size = probs_host->shape[1];

    tvm::runtime::parallel_for_with_threading_backend(
        [&](int i) {
          int verify_start = cum_verify_lengths[i];
          int verify_end = cum_verify_lengths[i + 1];
          for (int cur_token_idx = 0; cur_token_idx < verify_end - verify_start; ++cur_token_idx) {
            float* p_probs = global_p_probs + (verify_start + cur_token_idx) * vocab_size;
            int cur_token = draft_output_tokens[i][cur_token_idx].sampled_token_id.first;
            float q_value = draft_output_tokens[i][cur_token_idx].sampled_token_id.second;
            float p_value = p_probs[cur_token];

            if (p_value >= q_value) {
              sample_results[i].push_back(
                  SampleResult{{cur_token, p_value},
                               ComputeTopProbs(probs_host, verify_start + cur_token_idx,
                                               generation_cfg[i]->top_logprobs)});
              continue;
            }
            float r = rngs[i]->GetRandomNumber();
            if (r < p_value / (q_value + eps_)) {
              sample_results[i].push_back(
                  SampleResult{{cur_token, p_value},
                               ComputeTopProbs(probs_host, verify_start + cur_token_idx,
                                               generation_cfg[i]->top_logprobs)});
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
            SampleResult sample_result;
            sample_result.sampled_token_id = SampleTopPFromProb(
                probs_host, verify_start + cur_token_idx,
                generation_cfg[i]->temperature < eps_ ? 0.0 : generation_cfg[i]->top_p,
                rngs[i]->GetRandomNumber());
            sample_result.top_prob_tokens = ComputeTopProbs(
                probs_host, verify_start + cur_token_idx, generation_cfg[i]->top_logprobs);
            sample_results[i].push_back(sample_result);
            break;
          }
        },
        0, num_sequence);
    RECORD_EVENT(trace_recorder_, request_ids, "finish draft verification");
    return sample_results;
  }

 private:
  /*! \brief Copy prob distributions from device to CPU. */
  NDArray CopyProbsToCPU(NDArray probs_on_device) {
    // probs_on_device: (n, v)
    ICHECK(probs_on_device->device.device_type != kDLCPU);
    if (probs_host_.defined()) {
      ICHECK_EQ(probs_host_->shape[1], probs_on_device->shape[1]);
    }

    int64_t init_size = probs_host_.defined() ? probs_host_->shape[0] : 32;
    int64_t num_tokens = probs_on_device->shape[0];
    int64_t vocab_size = probs_on_device->shape[1];
    while (init_size < num_tokens) {
      init_size *= 2;
    }
    if (!probs_host_.defined() || init_size != probs_host_->shape[0]) {
      probs_host_ =
          NDArray::Empty({init_size, vocab_size}, probs_on_device->dtype, DLDevice{kDLCPU, 0});
    }
    ICHECK_LE(num_tokens, probs_host_->shape[0]);
    NDArray view = probs_host_.CreateView({num_tokens, vocab_size}, probs_on_device->dtype);
    view.CopyFrom(probs_on_device);
    return view;
  }

  /*! \brief The event trace recorder for requests. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Customized function which computes prob distribution from logits */
  PackedFunc flogits_to_probs_inplace_;
  /*! \brief Probability distribution array on CPU. */
  NDArray probs_host_{nullptr};
  const float eps_ = 1e-5;
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

}  // namespace serve
}  // namespace llm
}  // namespace mlc
