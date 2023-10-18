/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.cc
 * \brief The implementation for runtime module of sampler functions.
 */
#define __STDC_FORMAT_MACROS

#include "sampler.h"

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
  explicit SamplerModule(DLDevice device) : device_(device) {
    // Set sampling function.
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "compute_probs_from_logits") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 4);
        *rv = ComputeProbsFromLogits(args[0], args[1], args[2], args[3]);
      });
    } else if (name == "sample_token_from_probs") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3);
        *rv = SampleTokenFromProbs(args[0], args[1], args[2]);
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

  void Init(DLDevice device) { device_ = device; }

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
    const PackedFunc* f_logits_to_probs = Registry::Get("mlc.llm.f_logits_to_probs");
    if (f_logits_to_probs != nullptr) {
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
   * \brief Compute the probability distribution from logits for
   * a **single token**.
   * The input logits may be batched. We use an input "token offset"
   * to determine the start logit offset of the token to compute.
   * \param logits The input logits.
   * \param token_offset The input token offset to determine where the
   * logits of the target token start.
   * \param state The request state, which contains the history generated tokens.
   * \param generation_cfg The generation config.
   * \return The computed probability distribution for the specified token.
   */
  NDArray ComputeProbsFromLogits(NDArray logits, int token_offset, RequestModelState state,
                                 GenerationConfig generation_cfg) {
    // logits: (b, n, v)
    CHECK_EQ(logits->ndim, 3);

    // - Copy logits to host.
    //   We only copy a slice of size `vocab_size` and do not copy the entire array.
    NDArray logits_on_cpu = UpdateLogitsOrProbsOnCPUSync(logits, token_offset, nullptr);
    ICHECK(logits_on_cpu.defined());
    ICHECK_EQ(logits_on_cpu->ndim, 1);
    ICHECK_EQ(logits_on_cpu->shape[0], logits->shape[2]);

    // - Invoke environment compute function if exists.
    const PackedFunc* f_logits_to_probs = Registry::Get("mlc.llm.compute_probs_from_logits");
    if (f_logits_to_probs != nullptr) {
      return (*f_logits_to_probs)(logits_on_cpu, state, generation_cfg);
    }

    // - Apply repetition penalty (inplace).
    if (generation_cfg->repetition_penalty != 1.0) {
      ApplyRepetitionPenaltyOnCPU(logits_on_cpu, state, generation_cfg->repetition_penalty);
    }
    // - Compute probability (inplace) from logits.
    //   Using softmax if temperature is non-zero.
    //   Or set probability of the max-logit position to 1.
    if (generation_cfg->temperature >= 1e-6) {
      ApplySoftmaxWithTemperatureOnCPU(logits_on_cpu, generation_cfg->temperature);
    } else {
      SetProbWithArgmaxOnCPU(logits_on_cpu);
    }

    return logits_on_cpu;
  }

  /*!
   * \brief Sample a token from the input probability distribution.
   * The input logits may be batched. We use an input "token offset"
   * to determine the start probability offset of the token to compute.
   * \param probs The input probability distribution.
   * \param token_offset The input token offset to determine where the
   * probability distribution of the target token start.
   * \param generation_cfg The generation config.
   * \return The sampled token.
   */
  int32_t SampleTokenFromProbs(NDArray probs, int token_offset, GenerationConfig generation_cfg) {
    // probs: (b, n, v) or (v,)
    CHECK(probs->ndim == 3 || probs->ndim == 1);

    NDArray probs_on_cpu{nullptr};
    if (probs->ndim == 3) {
      // - Copy probs to host.
      //   We only copy a slice of size `vocab_size` and do not copy the entire array.
      if (probs->device.device_type == kDLCPU && probs->shape[0] == 1 && probs->shape[1] == 1) {
        // No need to copy.
        probs_on_cpu = probs_on_cpu.CreateView({probs->shape[2]}, probs->dtype);
      } else {
        probs_on_cpu = UpdateLogitsOrProbsOnCPUSync(probs, token_offset, &probs_on_cpu_);
      }
      ICHECK(probs_on_cpu.defined());
      ICHECK_EQ(probs_on_cpu->shape[0], probs->shape[2]);
    } else {
      probs_on_cpu = probs;
    }
    ICHECK_EQ(probs_on_cpu->ndim, 1);

    // - Invoke environment compute function if exists.
    const PackedFunc* f_sample_from_probs = Registry::Get("mlc.llm.sample_from_probs");
    if (f_sample_from_probs != nullptr) {
      return (*f_sample_from_probs)(probs_on_cpu, generation_cfg);
    }

    // Sample top p from probability.
    return fsample_topp_from_prob_(probs_on_cpu, generation_cfg->top_p, GetRandomNumber());
  }

  /*! \brief Copy logits or probabilities to CPU memory. */
  NDArray UpdateLogitsOrProbsOnCPUSync(NDArray arr_on_device, int token_offset,
                                       NDArray* p_arr_on_cpu) {
    // arr_on_device: (b, n, v)
    ICHECK_EQ(arr_on_device->ndim, 3);
    int vocab_size = arr_on_device->shape[2];
    DLDataType dtype = arr_on_device->dtype;

    // - Reuse the NDArray object if `p_arr_on_cpu` is not null.
    // - Otherwise, allocate a new NDArray.
    NDArray arr_on_cpu;
    if (p_arr_on_cpu != nullptr) {
      if (p_arr_on_cpu->defined()) {
        ICHECK_EQ((*p_arr_on_cpu)->ndim, 1);
        ICHECK_EQ((*p_arr_on_cpu)->shape[0], vocab_size)
            << "Expect vocabulary size remain unchanged";
        ICHECK((*p_arr_on_cpu).DataType() == arr_on_device.DataType());
      } else {
        *p_arr_on_cpu = NDArray::Empty({vocab_size}, dtype, DLDevice{kDLCPU, 0});
      }
      arr_on_cpu = *p_arr_on_cpu;
    } else {
      arr_on_cpu = NDArray::Empty({vocab_size}, dtype, DLDevice{kDLCPU, 0});
    }

    DLTensor copy_dst = *(arr_on_cpu.operator->());
    DLTensor copy_src = *(arr_on_device.operator->());
    copy_src.byte_offset = token_offset * vocab_size * ((dtype.bits * dtype.lanes + 7) / 8);
    copy_src.shape = arr_on_cpu->shape;
    copy_src.ndim = 1;
    NDArray::CopyFromTo(&copy_src, &copy_dst);
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    return arr_on_cpu;
  }

  /*! \brief Apply repetition penalty to logits based on history tokens. */
  void ApplyRepetitionPenaltyOnCPU(NDArray logits_on_cpu, RequestModelState state,
                                   double repetition_penalty) {
    // logits: (v,)
    CHECK(logits_on_cpu.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    CHECK_EQ(logits_on_cpu->ndim, 1);
    CHECK_EQ(logits_on_cpu->device.device_type, DLDeviceType::kDLCPU);
    int vocab_size = logits_on_cpu->shape[0];

    // Collect appeared tokens.
    std::unordered_set<int32_t> appeared_token_ids;
    appeared_token_ids.insert(state->committed_tokens.begin(), state->committed_tokens.end());
    appeared_token_ids.insert(state->draft_output_tokens.begin(), state->draft_output_tokens.end());

    float* logits_raw_data = static_cast<float*>(logits_on_cpu->data);
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
  void ApplySoftmaxWithTemperatureOnCPU(NDArray logits_on_cpu, double temperature) {
    // logits: (v,)
    CHECK(logits_on_cpu.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    CHECK_EQ(logits_on_cpu->ndim, 1);
    CHECK_EQ(logits_on_cpu->device.device_type, DLDeviceType::kDLCPU);
    int vocab_size = logits_on_cpu->shape[0];

    float* logits_raw_data = static_cast<float*>(logits_on_cpu->data);
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
  void SetProbWithArgmaxOnCPU(NDArray logits_on_cpu) {
    // logits: (v,)
    CHECK(logits_on_cpu.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    CHECK_EQ(logits_on_cpu->ndim, 1);
    CHECK_EQ(logits_on_cpu->device.device_type, kDLCPU);
    int vocab_size = logits_on_cpu->shape[0];

    float* logits_raw_data = static_cast<float*>(logits_on_cpu->data);
    int argmax_pos = -1;
    float max_logits = std::numeric_limits<float>::min();
    for (int i = 0; i < vocab_size; ++i) {
      if (logits_raw_data[i] > max_logits) {
        max_logits = logits_raw_data[i];
        argmax_pos = i;
      }
    }
    ICHECK_NE(argmax_pos, -1);

    std::vector<float> probs(/*count=*/vocab_size, /*value=*/0.0);
    probs[argmax_pos] = 1.0;
    logits_on_cpu.CopyFromBytes(probs.data(), vocab_size * sizeof(float));
  }

  static double GetRandomNumber() { return RandomGenerator::GetInstance().GetRandomNumber(); }

  /*! \brief The runtime device where the input logits is. */
  DLDevice device_;
  /*! \brief Function which samples a token from prob distribution with top_p value. */
  PackedFunc fsample_topp_from_prob_;
  /*! \brief Shared array for probability distribution on cpu */
  NDArray probs_on_cpu_{nullptr};
};

tvm::runtime::Module CreateSamplerModule(DLDevice device) {
  ObjectPtr<SamplerModule> n = make_object<SamplerModule>(device);
  return Module(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
