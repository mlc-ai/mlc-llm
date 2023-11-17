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

  // Collect appeared tokens.
  std::unordered_set<int32_t> appeared_token_ids;
  appeared_token_ids.insert(state->committed_tokens.begin(), state->committed_tokens.end());
  appeared_token_ids.insert(state->draft_output_tokens.begin(), state->draft_output_tokens.end());

  float* logits_raw_data = static_cast<float*>(logits->data) + (token_offset * vocab_size);
  for (int32_t token_id : appeared_token_ids) {
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
 * \return The sampled value.
 * \note This function is an enhancement of SampleTopPFromProb in TVM Unity.
 * We will upstream the enhancement after it gets stable.
 */
int SampleTopPFromProb(NDArray prob, int unit_offset, double top_p, double uniform_sample) {
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

  if (top_p >= one) {
    // Specially handle case where top_p == 1.
    double prob_sum = 0.0f;
    for (int64_t i = 0; i < ndata; ++i) {
      prob_sum += p_prob[i];
      if (prob_sum >= uniform_sample) {
        return i;
      }
    }
    LOG(INFO) << "prob sum = " << prob_sum << ", sample = " << uniform_sample;
    ICHECK(false) << "Possibly prob distribution contains NAN.";
  }

  // Key observation: when we are doing top_p sampling
  // usually we only need to preserve some of the elements with
  // high probabilities before we do sort
  thread_local std::vector<std::pair<float, int>> data;

  auto sample_top_p_with_filter = [&](float cuttoff) -> int64_t {
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
    if (data.size() == 0) return -1;
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
    if (uniform_sample < data[0].first / top_p) return data[0].second;

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
    if (cum_sum_prob < top_p && cuttoff != 0.0f) return -1;

    for (auto it = data.begin(); it != data.end(); ++it) {
      if (uniform_sample < it->first / top_p_sum) {
        return it->second;
      }
    }
    return data[data.size() - 1].second;
  };

  if (top_p < 1) {
    // sample through cutoff by a number
    // by pigeonhole principle we will get at most 1024 elements
    // usually it is much less by applying this filtering(order of 10 - 20)
    data.reserve(256);
    int64_t sampled_index = sample_top_p_with_filter(top_p / 1024);
    if (sampled_index >= 0) return sampled_index;
  }
  // fallback via full prob, rare case
  data.reserve(ndata);
  int64_t sampled_index = sample_top_p_with_filter(0.0f);
  ICHECK_GE(sampled_index, 0);
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
  explicit CPUSampler() : rng_(RandomGenerator::GetInstance()) {
    // Set customized "logits -> prob" function.
    const PackedFunc* f_logits_to_probs =
        Registry::Get("mlc.llm.compute_probs_from_logits_inplace");
    if (f_logits_to_probs != nullptr) {
      flogits_to_probs_inplace_ = *f_logits_to_probs;
    }
  }

  std::vector<int32_t> SampleTokens(NDArray logits_on_device, Model model,
                                    Array<RequestModelState> request_mstates,
                                    Array<GenerationConfig> generation_cfg) final {
    ICHECK(logits_on_device.defined());
    ICHECK_EQ(logits_on_device->ndim, 3);
    ICHECK_EQ(logits_on_device->shape[1], 1)
        << "Multi-token sampling for one sequence is not supported yet.";
    ICHECK_EQ(logits_on_device->shape[0], generation_cfg.size());
    ICHECK_EQ(request_mstates.size(), generation_cfg.size());

    int num_sequence = logits_on_device->shape[0];
    bool require_gpu_softmax = RequireGPUSoftmax(generation_cfg);

    // - Compute probabilities from logits.
    NDArray logits_or_probs_on_cpu{nullptr};
    if (require_gpu_softmax) {
      NDArray probs_on_device = model->SoftmaxWithTemperature(logits_on_device, generation_cfg);
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(probs_on_device, &logits_or_probs_on_cpu_);
    } else {
      logits_or_probs_on_cpu = CopyLogitsOrProbsToCPU(logits_on_device, &logits_or_probs_on_cpu_);
      // The "ComputeProbsFromLogitsInplace" function updates
      // `logits_or_probs_on_cpu` in place.
      ComputeProbsFromLogitsInplace(logits_or_probs_on_cpu, std::move(request_mstates),
                                    generation_cfg);
    }
    // `CopyLogitsOrProbsToCPU` flattens the first two dimensions.
    ICHECK_EQ(logits_or_probs_on_cpu->ndim, 2);

    // - Sample tokens from probabilities.
    // NOTE: Though we have the probability field in RequestModelState,
    //       we do not save the probabilities right now.
    //       We will handle this in the future when we work on speculation.
    return SampleTokenFromProbs(logits_or_probs_on_cpu, generation_cfg);
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
    // - Return false if any sampling param has repetition penalty other than 1.0.
    // - Return false if any sampling param has zero temperature.
    for (GenerationConfig cfg : generation_cfg) {
      if (cfg->repetition_penalty != 1.0 || cfg->temperature < 1e-6) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Compute the probability distribution from on-cpu logits for
   * a batch of tokens **in place**.
   * \param logits The input logits on CPU.
   * \param states The request states, which contains the history generated tokens.
   * \param generation_cfg The generation config.
   * \note The function returns nothing. It in-place updates the input logits array.
   */
  void ComputeProbsFromLogitsInplace(NDArray logits, Array<RequestModelState> states,
                                     Array<GenerationConfig> generation_cfg) {
    // logits: (n, v)
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->device.device_type, kDLCPU);

    // - Invoke environment compute function if exists.
    if (flogits_to_probs_inplace_.defined()) {
      flogits_to_probs_inplace_(logits, states, generation_cfg);
      return;
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [this, &logits, &states, &generation_cfg](int i) {
          // - Apply repetition penalty (inplace).
          if (generation_cfg[i]->repetition_penalty != 1.0) {
            ApplyRepetitionPenaltyOnCPU(logits, i, states[i],
                                        generation_cfg[i]->repetition_penalty);
          }
          // - Compute probability (inplace) from logits.
          //   Using softmax if temperature is non-zero.
          //   Or set probability of the max-logit position to 1.
          if (generation_cfg[i]->temperature >= 1e-6) {
            ApplySoftmaxWithTemperatureOnCPU(logits, i, generation_cfg[i]->temperature);
          } else {
            SetProbWithArgmaxOnCPU(logits, i);
          }
        },
        0, logits->shape[0]);
  }

  /*!
   * \brief Sample tokens from a batch of input probability distributions.
   * \param probs The input batch of probability distributions.
   * \param generation_cfg The generation config.
   * \return The sampled tokens, one for each instance of the batch.
   */
  std::vector<int32_t> SampleTokenFromProbs(NDArray probs, Array<GenerationConfig> generation_cfg) {
    // probs: (n, v)
    CHECK_EQ(probs->ndim, 2);
    CHECK_EQ(probs->device.device_type, kDLCPU);

    int n = probs->shape[0];
    std::vector<double> random_numbers;
    std::vector<int32_t> sampled_tokens;
    random_numbers.reserve(n);
    sampled_tokens.resize(n);
    for (int i = 0; i < n; ++i) {
      random_numbers.push_back(rng_.GetRandomNumber());
    }

    tvm::runtime::parallel_for_with_threading_backend(
        [&sampled_tokens, &probs, &generation_cfg, &random_numbers](int i) {
          // Sample top p from probability.
          sampled_tokens[i] =
              SampleTopPFromProb(probs, i, generation_cfg[i]->top_p, random_numbers[i]);
        },
        0, n);
    return sampled_tokens;
  }

  /*! \brief The random generator. */
  RandomGenerator& rng_;
  /*! \brief Customized function which computes prob distribution from logits */
  PackedFunc flogits_to_probs_inplace_;
  /*! \brief Shared array for logits and probability distributions on cpu. */
  NDArray logits_or_probs_on_cpu_{nullptr};
};

/*********************** Sampler ***********************/

TVM_REGISTER_OBJECT_TYPE(SamplerObj);

Sampler Sampler::Create(std::string sampler_kind) {
  if (sampler_kind == "cpu") {
    return Sampler(make_object<CPUSampler>());
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
