/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/sampler/cpu_sampler.cc
 * \brief The implementation for CPU sampler functions.
 */
#include <tvm/ffi/function.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/threading_backend.h>

#include <algorithm>
#include <cmath>

#include "../../support/random.h"
#include "sampler.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_FFI_STATIC_INIT_BLOCK() { SamplerObj::RegisterReflection(); }

/*!
 * \brief Sample a value from the input probability distribution with top-p.
 * The input is a batch of distributions, and we use `unit_offset` to specify
 * which distribution to sample from.
 * \param prob The input batch of probability distributions.
 * \param unit_offset The offset specifying which distribution to output
 * \param input_prob_offset The offset specifying which distribution to sample from.
 * \param top_p The top-p value of sampling.
 * \param uniform_sample The random number in [0, 1] for sampling.
 * \return The sampled value and probability.
 * \note This function is an enhancement of SampleTopPFromProb in TVM Unity.
 * We will upstream the enhancement after it gets stable.
 */
TokenProbPair SampleTopPFromProb(Tensor prob, int unit_offset, int input_prob_offset, double top_p,
                                 double uniform_sample) {
  // prob: (*, v)
  // The prob array may have arbitrary ndim and shape.
  // The last dimension corresponds to the prob distribution size.
  // We use the `unit_offset` parameter to determine which slice
  // of the prob array we sample from.

  TVM_FFI_ICHECK(prob.IsContiguous());
  TVM_FFI_ICHECK(prob.DataType() == DataType::Float(32));
  TVM_FFI_ICHECK_EQ(prob->device.device_type, DLDeviceType::kDLCPU);

  int64_t ndata = prob->shape[prob->ndim - 1];
  const float* __restrict p_prob =
      static_cast<float*>(__builtin_assume_aligned(prob->data, 4)) + (input_prob_offset * ndata);
  constexpr double one = 1.0f - 1e-5f;

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
    return {argmax_pos, 1.0};
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
    TVM_FFI_ICHECK(false) << "Possibly prob distribution contains NAN.";
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
    return std::make_pair(data[static_cast<int64_t>(data.size()) - 1].first - last_cum_sum_prob,
                          data[static_cast<int64_t>(data.size()) - 1].second);
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
  TVM_FFI_ICHECK_GE(sampled_index.second, 0);
  return {sampled_index.second, sampled_index.first};
}

/*!
 * \brief Renormalize the probability distribution by the top p value.
 * \param prob The input batch of probability distributions.
 * \param unit_offset The offset specifying which distribution to output
 * \param top_p The top p value for renormalization.
 * \param eps A small epsilon value for comparison stability.
 */
void RenormalizeProbByTopP(Tensor prob, int unit_offset, double top_p, double eps) {
  // prob: (*, v)
  // The prob array may have arbitrary ndim and shape.
  // The last dimension corresponds to the prob distribution size.
  // We use the `unit_offset` parameter to determine which slice
  // of the prob array we will renormalize.
  TVM_FFI_ICHECK(prob.IsContiguous());
  TVM_FFI_ICHECK(prob.DataType() == DataType::Float(32));
  TVM_FFI_ICHECK_EQ(prob->device.device_type, DLDeviceType::kDLCPU);

  if (top_p == 1.0) {
    // No renormalization is needed if top_p is 1.
    return;
  }

  int vocab_size = prob->shape[prob->ndim - 1];
  float* __restrict p_prob =
      static_cast<float*>(__builtin_assume_aligned(prob->data, 4)) + (unit_offset * vocab_size);

  // We manually choice the cutoff values of "top_p / 256" and "top_p / 8192".
  // In most of the cases, only one round is needed.
  std::vector<double> cutoff_values{top_p / 256, top_p / 8192, 0.0f};

  // Create the upper partition vector and the lower partition rolling vectors.
  std::vector<float> upper_partition;
  std::vector<float> lower_partitions[2];
  upper_partition.reserve(vocab_size);
  lower_partitions[0].reserve(vocab_size);
  lower_partitions[1].reserve(vocab_size);
  float upper_partition_sum = 0.0;
  for (int round = 0; round < static_cast<int>(cutoff_values.size()); ++round) {
    const float* lower_partition_begin;
    const float* lower_partition_end;
    if (round == 0) {
      lower_partition_begin = p_prob;
      lower_partition_end = p_prob + vocab_size;
    } else {
      int idx = (round - 1) & 1;
      lower_partition_begin = lower_partitions[idx].data();
      lower_partition_end = lower_partitions[idx].data() + lower_partitions[idx].size();
    }

    // - Partition the last round lower partition into upper and lower
    // based on the new cutoff value.
    std::vector<float>& lower_partition = lower_partitions[round & 1];
    lower_partition.clear();
    for (const float* ptr = lower_partition_begin; ptr != lower_partition_end; ++ptr) {
      if (*ptr >= cutoff_values[round]) {
        upper_partition.push_back(*ptr);
        upper_partition_sum += *ptr;
      } else {
        lower_partition.push_back(*ptr);
      }
    }
    // - If the upper partition sum is at least top p, exit the loop.
    if (upper_partition_sum >= top_p - eps) {
      break;
    }
  }

  // - Sort the upper partition in descending order.
  std::sort(upper_partition.begin(), upper_partition.end(), std::greater<>());
  // - Find the top p boundary prob value.
  float boundary_value = -1.0;
  upper_partition_sum = 0.0;
  for (float upper_value : upper_partition) {
    upper_partition_sum += upper_value;
    if (upper_partition_sum >= top_p - eps) {
      boundary_value = upper_value;
      break;
    }
  }
  // - Mask all values smaller than the boundary to 0.
  float renormalize_sum = 0.0;
  std::vector<int> upper_partition_indices;
  upper_partition_indices.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    if (p_prob[i] >= boundary_value) {
      upper_partition_indices.push_back(i);
      renormalize_sum += p_prob[i];
    } else {
      p_prob[i] = 0.0;
    }
  }
  // - Renormalize.
  for (int idx : upper_partition_indices) {
    p_prob[idx] /= renormalize_sum;
  }
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
inline std::vector<TokenProbPair> ComputeTopProbs(Tensor prob, int unit_offset, int num_top_probs) {
  TVM_FFI_ICHECK_LE(num_top_probs, 5);
  TVM_FFI_ICHECK_EQ(prob->ndim, 2);
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
      : trace_recorder_(std::move(trace_recorder)) {}

  Tensor BatchRenormalizeProbsByTopP(Tensor probs_on_device,                  //
                                     const std::vector<int>& sample_indices,  //
                                     const Array<String>& request_ids,        //
                                     const Array<GenerationConfig>& generation_cfg) final {
    // probs_on_device: (n, v)
    CHECK_EQ(probs_on_device->ndim, 2);
    // - Copy probs to CPU
    RECORD_EVENT(trace_recorder_, request_ids, "start copy probs to CPU");
    Tensor probs_on_host = CopyProbsToCPU(probs_on_device);
    RECORD_EVENT(trace_recorder_, request_ids, "finish copy probs to CPU");
    int num_samples = sample_indices.size();
    int num_probs = probs_on_device->shape[0];
    int vocab_size = probs_on_device->shape[1];
    TVM_FFI_ICHECK_EQ(request_ids.size(), num_samples);
    TVM_FFI_ICHECK_EQ(generation_cfg.size(), num_samples);

    std::vector<int> top_p_indices;
    std::vector<double> top_p_values;
    for (int i = 0; i < num_samples; ++i) {
      if (top_p_indices.empty() || top_p_indices.back() != sample_indices[i]) {
        top_p_indices.push_back(sample_indices[i]);
        top_p_values.push_back(generation_cfg[i]->top_p);
      } else {
        CHECK(fabs(top_p_values.back() - generation_cfg[i]->top_p) < eps_)
            << "Sampler requires the top_p values for each prob distribution are the same.";
      }
    }
    if (top_p_indices.empty()) {
      // Return if no top p needs to apply.
      return probs_on_host;
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &probs_on_host, &request_ids, &top_p_indices, &top_p_values](int i) {
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "start renormalize by top p");
          RenormalizeProbByTopP(probs_on_host, top_p_indices[i], top_p_values[i], eps_);
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "finish renormalize by top p");
        },
        0, static_cast<int64_t>(top_p_indices.size()));

    return probs_on_host;
  }

  std::vector<SampleResult> BatchSampleTokensWithProbBeforeTopP(
      Tensor probs_on_device,                         //
      const std::vector<int>& sample_indices,         //
      const Array<String>& request_ids,               //
      const Array<GenerationConfig>& generation_cfg,  //
      const std::vector<RandomGenerator*>& rngs) final {
    // probs_on_device: (n, v)
    CHECK_EQ(probs_on_device->ndim, 2);
    // - Copy probs to CPU
    RECORD_EVENT(trace_recorder_, request_ids, "start copy probs to CPU");
    Tensor probs_on_host = CopyProbsToCPU(probs_on_device);
    RECORD_EVENT(trace_recorder_, request_ids, "finish copy probs to CPU");

    return BatchSampleTokensImpl(probs_on_host, sample_indices, request_ids, generation_cfg, rngs,
                                 /*top_p_applied=*/false);
  }

  std::vector<SampleResult> BatchSampleTokensWithProbAfterTopP(
      Tensor probs_on_host,                           //
      const std::vector<int>& sample_indices,         //
      const Array<String>& request_ids,               //
      const Array<GenerationConfig>& generation_cfg,  //
      const std::vector<RandomGenerator*>& rngs) final {
    return BatchSampleTokensImpl(probs_on_host, sample_indices, request_ids, generation_cfg, rngs,
                                 /*top_p_applied=*/true);
  }

  std::pair<std::vector<std::vector<SampleResult>>, std::vector<int>>
  BatchVerifyDraftTokensWithProbAfterTopP(
      Tensor probs_on_host, const Array<String>& request_ids,
      const std::vector<int>& cum_verify_lengths, const Array<GenerationConfig>& generation_cfg,
      const std::vector<RandomGenerator*>& rngs,
      const std::vector<std::vector<SampleResult>>& draft_output_tokens,
      const std::vector<int64_t>& token_tree_parent_ptr, Tensor draft_probs_on_device) final {
    // probs_on_host: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start draft verification");
    CHECK_EQ(probs_on_host->ndim, 2);

    int num_sequence = static_cast<int>(cum_verify_lengths.size()) - 1;
    CHECK_EQ(rngs.size(), num_sequence);
    CHECK_EQ(draft_output_tokens.size(), num_sequence);

    Tensor draft_probs_on_host = draft_probs_on_device.CopyTo(DLDevice{kDLCPU, 0});
    std::vector<std::vector<SampleResult>> sample_results;
    sample_results.resize(num_sequence);

    float* __restrict global_p_probs =
        static_cast<float*>(__builtin_assume_aligned(probs_on_host->data, 4));
    int vocab_size = probs_on_host->shape[1];

    std::vector<int> last_accepted_tree_node(num_sequence, 0);
    tvm::runtime::parallel_for_with_threading_backend(
        [&](int i) {
          int verify_start = cum_verify_lengths[i];
          int verify_end = cum_verify_lengths[i + 1];

          CHECK_EQ(token_tree_parent_ptr[verify_start], -1);
          for (int j = verify_start + 1; j < verify_end; ++j) {
            CHECK_EQ(token_tree_parent_ptr[j], j - verify_start - 1)
                << "CPU sampler only supports chain-style draft tokens.";
          }

          int cur_token_idx = 0;
          // Sub 1 to ignore the last prediction.
          for (; cur_token_idx < verify_end - verify_start - 1; ++cur_token_idx) {
            float* p_probs = global_p_probs + (verify_start + cur_token_idx) * vocab_size;
            int cur_token = draft_output_tokens[i][cur_token_idx].GetTokenId();
            float q_value = draft_output_tokens[i][cur_token_idx].sampled_token_id.second;
            float p_value = p_probs[cur_token];

            if (p_value >= q_value) {
              sample_results[i].push_back(
                  SampleResult{{cur_token, p_value},
                               ComputeTopProbs(probs_on_host, verify_start + cur_token_idx,
                                               generation_cfg[i]->top_logprobs)});
              continue;
            }
            float r = rngs[i]->GetRandomNumber();
            if (r < p_value / (q_value + eps_)) {
              sample_results[i].push_back(
                  SampleResult{{cur_token, p_value},
                               ComputeTopProbs(probs_on_host, verify_start + cur_token_idx,
                                               generation_cfg[i]->top_logprobs)});
              continue;
            }

            // normalize a new probability distribution
            double sum_v = 0.0;
            const float* __restrict p_qdist =
                static_cast<float*>(__builtin_assume_aligned(draft_probs_on_host->data, 4)) +
                (verify_start + cur_token_idx + 1) * vocab_size;

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
                probs_on_host, verify_start + cur_token_idx, verify_start + cur_token_idx,
                /*top_p=*/1.0f, rngs[i]->GetRandomNumber());
            sample_result.top_prob_tokens = ComputeTopProbs(
                probs_on_host, verify_start + cur_token_idx, generation_cfg[i]->top_logprobs);
            sample_results[i].push_back(sample_result);
            break;
          }
          last_accepted_tree_node[i] = cur_token_idx;
          // if cur_token_idx == verify_end - verify_start - 1
          // all draft tokens are accepted
          // we sample a new token
          if (cur_token_idx == verify_end - verify_start - 1) {
            SampleResult sample_result;
            // sample a new token from the original distribution
            sample_result.sampled_token_id = SampleTopPFromProb(
                probs_on_host, verify_start + cur_token_idx, verify_start + cur_token_idx,
                /*top_p=*/1.0f, rngs[i]->GetRandomNumber());
            sample_result.top_prob_tokens = ComputeTopProbs(
                probs_on_host, verify_start + cur_token_idx, generation_cfg[i]->top_logprobs);
            sample_results[i].push_back(sample_result);
          }
        },
        0, num_sequence);
    RECORD_EVENT(trace_recorder_, request_ids, "finish draft verification");
    return {sample_results, last_accepted_tree_node};
  }

 private:
  std::vector<SampleResult> BatchSampleTokensImpl(Tensor probs_on_host,                           //
                                                  const std::vector<int>& sample_indices,         //
                                                  const Array<String>& request_ids,               //
                                                  const Array<GenerationConfig>& generation_cfg,  //
                                                  const std::vector<RandomGenerator*>& rngs,      //
                                                  bool top_p_applied) {
    // probs_on_host: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start sampling");
    TVM_FFI_ICHECK_EQ(probs_on_host->ndim, 2);
    TVM_FFI_ICHECK_EQ(probs_on_host->device.device_type, DLDeviceType::kDLCPU);

    // - Sample tokens from probabilities.
    int n = request_ids.size();
    TVM_FFI_ICHECK_EQ(generation_cfg.size(), n);
    TVM_FFI_ICHECK_EQ(rngs.size(), n);

    std::vector<SampleResult> sample_results;
    sample_results.resize(n);

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &sample_results, &probs_on_host, &generation_cfg, &rngs, &request_ids, top_p_applied,
         sample_indices](int i) {
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "start sample token");
          // Sample top p from probability.
          double top_p =
              top_p_applied
                  ? 1.0f
                  : (generation_cfg[i]->temperature < eps_ ? 0.0 : generation_cfg[i]->top_p);
          sample_results[i].sampled_token_id = SampleTopPFromProb(
              probs_on_host, i, sample_indices[i], top_p, rngs[i]->GetRandomNumber());
          sample_results[i].top_prob_tokens =
              ComputeTopProbs(probs_on_host, i, generation_cfg[i]->top_logprobs);
          RECORD_EVENT(this->trace_recorder_, request_ids[i], "finish sample token");
        },
        0, n);
    RECORD_EVENT(trace_recorder_, request_ids, "finish sampling");
    return sample_results;
  }

  /*! \brief Copy prob distributions from device to CPU. */
  Tensor CopyProbsToCPU(Tensor probs_on_device) {
    // probs_on_device: (n, v)
    if (probs_on_device->device.device_type == kDLCPU) {
      return probs_on_device;
    }

    TVM_FFI_ICHECK(probs_on_device->device.device_type != kDLCPU);
    if (probs_host_.defined()) {
      TVM_FFI_ICHECK_EQ(probs_host_->shape[1], probs_on_device->shape[1]);
    }

    int64_t init_size = probs_host_.defined() ? probs_host_->shape[0] : 32;
    int64_t num_tokens = probs_on_device->shape[0];
    int64_t vocab_size = probs_on_device->shape[1];
    while (init_size < num_tokens) {
      init_size *= 2;
    }
    if (!probs_host_.defined() || init_size != probs_host_->shape[0]) {
      probs_host_ =
          Tensor::Empty({init_size, vocab_size}, probs_on_device->dtype, DLDevice{kDLCPU, 0});
    }
    TVM_FFI_ICHECK_LE(num_tokens, probs_host_->shape[0]);
    Tensor view = probs_host_.CreateView({num_tokens, vocab_size}, probs_on_device->dtype);
    view.CopyFrom(probs_on_device);
    return view;
  }

  /*! \brief The event trace recorder for requests. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Customized function which computes prob distribution from logits */
  Function flogits_to_probs_inplace_;
  /*! \brief Probability distribution array on CPU. */
  Tensor probs_host_{nullptr};
  const float eps_ = 1e-5;
};

Sampler Sampler::CreateCPUSampler(Optional<EventTraceRecorder> trace_recorder) {
  return Sampler(tvm::ffi::make_object<CPUSampler>(std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
