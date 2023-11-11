/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.cc
 * \brief The implementation for runtime module of sampler functions.
 */
#define __STDC_FORMAT_MACROS

#include "sampler.h"

#include <omp.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cmath>

#include "../random.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The sampler runtime module.
 * It contains functions to
 * - compute probability distribution out from logits,
 * - sample token from probability distribution.
 */
class SamplerModule : public ModuleNode {
 public:
  explicit SamplerModule(DLDevice device) : device_(device), rng_(RandomGenerator::GetInstance()) {
    // Set number of sampling threads from OpenMP.
    num_threads_ = static_cast<int>(omp_get_max_threads()) / 2;

    // Set sampling function.
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;

    // Set customized "logits -> prob" function.
    const PackedFunc* f_logits_to_probs =
        Registry::Get("mlc.llm.compute_probs_from_logits_inplace");
    if (f_logits_to_probs != nullptr) {
      flogits_to_probs_inplace_ = *f_logits_to_probs;
    }
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "compute_probs_from_logits_inplace") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3);
        ComputeProbsFromLogitsInplace(args[0], args[1], args[2]);
      });
    } else if (name == "sample_token_from_probs") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 2);
        *rv = SampleTokenFromProbs(args[0], args[1]);
      });
    } else if (name == "require_gpu_softmax") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        *rv = RequireGPUSoftmax(args[0]);
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  const char* type_key() const final { return "mlc.serve.Sampler"; }

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

    int n = logits->shape[0];
#pragma omp parallel for num_threads(this->num_threads_)
    for (int i = 0; i < n; ++i) {
      // - Apply repetition penalty (inplace).
      if (generation_cfg[i]->repetition_penalty != 1.0) {
        ApplyRepetitionPenaltyOnCPU(logits, i, states[i], generation_cfg[i]->repetition_penalty);
      }
      // - Compute probability (inplace) from logits.
      //   Using softmax if temperature is non-zero.
      //   Or set probability of the max-logit position to 1.
      if (generation_cfg[i]->temperature >= 1e-6) {
        ApplySoftmaxWithTemperatureOnCPU(logits, i, generation_cfg[i]->temperature);
      } else {
        SetProbWithArgmaxOnCPU(logits, i);
      }
    }
  }

  /*!
   * \brief Sample tokens from a batch of input probability distributions.
   * \param probs The input batch of probability distributions.
   * \param generation_cfg The generation config.
   * \return The sampled tokens, one for each instance of the batch.
   */
  ShapeTuple SampleTokenFromProbs(NDArray probs, Array<GenerationConfig> generation_cfg) {
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

#pragma omp parallel for num_threads(this->num_threads_)
    for (int i = 0; i < n; ++i) {
      // Sample top p from probability.
      sampled_tokens[i] =
          fsample_topp_from_prob_(probs, i, generation_cfg[i]->top_p, random_numbers[i]);
    }
    return ShapeTuple(sampled_tokens.begin(), sampled_tokens.end());
  }

  /*! \brief Apply repetition penalty to logits based on history tokens. */
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

  /*! \brief Compute softmax with temperature on CPU. */
  void ApplySoftmaxWithTemperatureOnCPU(NDArray logits, int token_offset, double temperature) {
    // logits: (n, v)
    CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->device.device_type, DLDeviceType::kDLCPU);
    int vocab_size = logits->shape[1];

    float* __restrict logits_raw_data =
        static_cast<float*>(__builtin_assume_aligned(logits->data, 4)) +
        (token_offset * vocab_size);
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
   * \brief Inplace set probability via argmax.
   * This is used for zero-temperature sampling cases
   */
  void SetProbWithArgmaxOnCPU(NDArray logits, int token_offset) {
    // logits: (n, v)
    CHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->device.device_type, kDLCPU);
    int vocab_size = logits->shape[1];

    float* logits_raw_data = static_cast<float*>(logits->data) + (token_offset * vocab_size);
    int argmax_pos = -1;
    float max_logits = std::numeric_limits<float>::min();
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

  /*! \brief The runtime device where the input logits is. */
  DLDevice device_;
  /*! \brief Number of CPU threads for parallel sampling and other computation. */
  int num_threads_;
  /*! \brief The random generator. */
  RandomGenerator& rng_;
  /*! \brief Customized function which computes prob distribution from logits */
  PackedFunc flogits_to_probs_inplace_;
  /*! \brief Function which samples a token from prob distribution with top_p value. */
  PackedFunc fsample_topp_from_prob_;
};

tvm::runtime::Module CreateSamplerModule(DLDevice device) {
  ObjectPtr<SamplerModule> n = make_object<SamplerModule>(device);
  return Module(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
