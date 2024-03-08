/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/logit_processor.cc
 * \brief The implementation of logit processor.
 */
#include "logit_processor.h"

#include <picojson.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>

namespace mlc {
namespace llm {
namespace serve {

inline void CopyArray(NDArray src, NDArray dst) {
  DLTensor dl_dst = *(dst.operator->());
  NDArray::CopyFromTo(src.operator->(), &dl_dst);
}

/***************** LogitProcessor Implementation *****************/

TVM_REGISTER_OBJECT_TYPE(LogitProcessorObj);

class LogitProcessorImpl : public LogitProcessorObj {
 public:
  /*! * \brief Constructor of LogitProcessorImpl. */
  explicit LogitProcessorImpl(int max_num_token, int vocab_size, FunctionTable* ft, DLDevice device,
                              Optional<EventTraceRecorder> trace_recorder)
      : max_num_token_(max_num_token),
        vocab_size_(vocab_size),
        bitmask_size_((vocab_size + 31) / 32),
        softmax_func_(ft->softmax_func_),
        device_(device),
        apply_logit_bias_func_(ft->apply_logit_bias_func_),
        apply_penalty_func_(ft->apply_penalty_func_),
        apply_bitmask_func_(ft->apply_bitmask_func_),
        trace_recorder_(std::move(trace_recorder)) {
    DLDevice device_cpu{DLDeviceType::kDLCPU, /*device_id=*/0};
    // Initialize auxiliary arrays on CPU.
    seq_ids_host_ = NDArray::Empty({max_num_token}, dtype_i32_, device_cpu);
    pos2seq_id_host_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device_cpu);
    token_ids_host_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device_cpu);
    token_cnt_host_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device_cpu);
    token_logit_bias_host_ = NDArray::Empty({max_num_token * vocab_size}, dtype_f32_, device_cpu);
    penalties_host_ = NDArray::Empty({max_num_token, 3}, dtype_f32_, device_cpu);
    bitmask_host_ = NDArray::Empty({max_num_token, bitmask_size_}, dtype_u32_, device_cpu);
    temperature_host_ = NDArray::Empty({max_num_token}, dtype_f32_, device_cpu);
    // Initialize auxiliary arrays on GPU.
    seq_ids_device_ = NDArray::Empty({max_num_token}, dtype_i32_, device);
    pos2seq_id_device_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_ids_device_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_cnt_device_ = NDArray::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_logit_bias_device_ = NDArray::Empty({max_num_token * vocab_size}, dtype_f32_, device);
    penalties_device_ = NDArray::Empty({max_num_token, 3}, dtype_f32_, device);
    bitmask_device_ = NDArray::Empty({max_num_token, bitmask_size_}, dtype_i32_, device);
    temperature_device_ = NDArray::Empty({max_num_token}, dtype_f32_, device);

    CHECK(apply_logit_bias_func_.defined())
        << "Function \"apply_logit_bias_inplace\" not found in model";
    CHECK(apply_penalty_func_.defined()) << "Function \"apply_penalty_inplace\" not found in model";
    CHECK(apply_bitmask_func_.defined()) << "Function \"apply_bitmask_inplace\" not found in model";
  }

  void InplaceUpdateLogits(NDArray logits,                                 //
                           const Array<GenerationConfig>& generation_cfg,  //
                           const Array<RequestModelState>& mstates,        //
                           const Array<String>& request_ids,               //
                           const std::vector<int>* cum_num_token,          //
                           const std::vector<std::vector<SampleResult>>* draft_tokens) final {
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->shape[1], vocab_size_);
    CHECK(logits.DataType() == DataType::Float(32));
    CHECK_EQ(generation_cfg.size(), mstates.size());
    CHECK_LE(logits->shape[0], max_num_token_);
    int num_total_token = logits->shape[0];
    int num_sequence = generation_cfg.size();

    CHECK((cum_num_token == nullptr) == (draft_tokens == nullptr));
    if (cum_num_token != nullptr) {
      CHECK_EQ(draft_tokens->size(), num_sequence);
      CHECK_EQ(cum_num_token->size(), num_sequence + 1);
      CHECK_EQ(cum_num_token->back(), num_total_token);
    } else {
      CHECK_EQ(num_sequence, num_total_token);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start update logits");

    // Update 1. logit bias
    RECORD_EVENT(trace_recorder_, request_ids, "start apply logit bias");
    UpdateWithLogitBias(logits, generation_cfg, cum_num_token);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply logit bias");

    // Update 2. penalties
    RECORD_EVENT(trace_recorder_, request_ids, "start apply penalty");
    UpdateWithPenalty(logits, generation_cfg, mstates, cum_num_token, draft_tokens);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply penalty");

    // Update 3. Vocabulary mask.
    RECORD_EVENT(trace_recorder_, request_ids, "start apply logit mask");
    UpdateWithMask(logits, mstates, cum_num_token, draft_tokens);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply logit mask");

    RECORD_EVENT(trace_recorder_, request_ids, "finish update logits");
  }

  NDArray ComputeProbsFromLogits(NDArray logits, const Array<GenerationConfig>& generation_cfg,
                                 const Array<String>& request_ids,
                                 const std::vector<int>* cum_num_token) final {
    // logits: (n, v)
    CHECK_EQ(logits->ndim, 2);
    CHECK_LE(logits->shape[0], max_num_token_);
    CHECK_EQ(logits->shape[1], vocab_size_);
    CHECK(logits.DataType() == DataType::Float(32));
    int num_total_token = logits->shape[0];
    int num_sequence = generation_cfg.size();

    if (cum_num_token != nullptr) {
      CHECK_EQ(cum_num_token->size(), num_sequence + 1);
      CHECK_EQ(cum_num_token->back(), num_total_token);
    } else {
      CHECK_EQ(num_sequence, num_total_token);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start softmax");

    // Construct:
    // - temperature (max_num_token,) float32
    float* p_temperature = static_cast<float*>(temperature_host_->data);

    // - Set arrays.
    for (int i = 0; i < num_sequence; ++i) {
      int num_token_to_process =
          cum_num_token == nullptr ? 1 : (cum_num_token->at(i + 1) - cum_num_token->at(i));
      int token_offset = cum_num_token == nullptr ? i : cum_num_token->at(i);
      for (int j = 0; j < num_token_to_process; ++j) {
        p_temperature[token_offset + j] = std::max(generation_cfg[i]->temperature, eps_);
      }
    }

    // - View arrays.
    NDArray temperature_host = temperature_host_.CreateView({num_total_token}, dtype_f32_);
    NDArray temperature_device = temperature_device_.CreateView({num_total_token}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/temperature_host, /*dst=*/temperature_device);

    // - Call kernel.
    NDArray probs = softmax_func_(logits.CreateView({num_total_token, 1, vocab_size_}, dtype_f32_),
                                  temperature_device);
    ICHECK_EQ(probs->ndim, 3);
    ICHECK_EQ(probs->shape[0], num_total_token);
    ICHECK_EQ(probs->shape[1], 1);
    ICHECK_EQ(probs->shape[2], vocab_size_);
    if (trace_recorder_.defined()) {
      TVMSynchronize(device_.device_type, device_.device_id, /*stream=*/nullptr);
    }
    RECORD_EVENT(trace_recorder_, request_ids, "finish softmax");
    return probs.CreateView({num_total_token, vocab_size_}, probs->dtype);
  }

 private:
  void UpdateWithLogitBias(NDArray logits, const Array<GenerationConfig>& generation_cfg,
                           const std::vector<int>* cum_num_token) {
    // Construct:
    // - pos2seq_id (max_num_token * vocab_size,) int32
    // - token_ids (max_num_token * vocab_size,) int32
    // - token_logit_bias (max_num_token * vocab_size,) float32
    int* p_pos2seq_id = static_cast<int*>(pos2seq_id_host_->data);
    int* p_token_ids = static_cast<int*>(token_ids_host_->data);
    float* p_token_logit_bias = static_cast<float*>(token_logit_bias_host_->data);

    // - Set arrays.
    int num_token_for_bias = 0;
    int num_bias_token = 0;
    for (int i = 0; i < static_cast<int>(generation_cfg.size()); ++i) {
      int num_token_to_process =
          cum_num_token == nullptr ? 1 : (cum_num_token->at(i + 1) - cum_num_token->at(i));
      int token_offset = cum_num_token == nullptr ? i : cum_num_token->at(i);
      for (int j = 0; j < num_token_to_process; ++j) {
        if (!generation_cfg[i]->logit_bias.empty()) {
          for (auto [token_id, bias] : generation_cfg[i]->logit_bias) {
            p_pos2seq_id[num_bias_token] = token_offset + j;
            p_token_ids[num_bias_token] = token_id;
            p_token_logit_bias[num_bias_token] = bias;
            ++num_bias_token;
          }
          ++num_token_for_bias;
        }
      }
    }

    if (num_token_for_bias == 0) {
      return;
    }

    // - View arrays.
    int num_token = num_bias_token;
    NDArray pos2seq_id_host = pos2seq_id_host_.CreateView({num_token}, dtype_i32_);
    NDArray pos2seq_id_device = pos2seq_id_device_.CreateView({num_token}, dtype_i32_);
    NDArray token_ids_host = token_ids_host_.CreateView({num_token}, dtype_i32_);
    NDArray token_ids_device = token_ids_device_.CreateView({num_token}, dtype_i32_);
    NDArray token_logit_bias_host = token_logit_bias_host_.CreateView({num_token}, dtype_f32_);
    NDArray token_logit_bias_device = token_logit_bias_device_.CreateView({num_token}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/pos2seq_id_host, /*dst=*/pos2seq_id_device);
    CopyArray(/*src=*/token_ids_host, /*dst=*/token_ids_device);
    CopyArray(/*src=*/token_logit_bias_host, /*dst=*/token_logit_bias_device);

    // - Call kernel.
    apply_logit_bias_func_(logits, pos2seq_id_device, token_ids_device, token_logit_bias_device);
    if (trace_recorder_.defined()) {
      TVMSynchronize(device_.device_type, device_.device_id, /*stream=*/nullptr);
    }
  }

  void UpdateWithPenalty(NDArray logits, const Array<GenerationConfig>& generation_cfg,
                         const Array<RequestModelState>& mstates,
                         const std::vector<int>* cum_num_token,
                         const std::vector<std::vector<SampleResult>>* draft_tokens) {
    // Construct:
    // - seq_ids (max_num_token,) int32
    // - pos2seq_id (max_num_token * vocab_size,) int32
    // - token_ids (max_num_token * vocab_size,) int32
    // - token_cnt (max_num_token * vocab_size,) int32
    // - penalties (max_num_token, 3) float32
    int* p_seq_ids = static_cast<int*>(seq_ids_host_->data);
    int* p_pos2seq_id = static_cast<int*>(pos2seq_id_host_->data);
    int* p_token_ids = static_cast<int*>(token_ids_host_->data);
    int* p_token_cnt = static_cast<int*>(token_cnt_host_->data);
    float* p_penalties = static_cast<float*>(penalties_host_->data);

    // - Set arrays.
    int num_token_for_penalty = 0;
    int num_penalty_appeared_token = 0;
    for (int i = 0; i < static_cast<int>(generation_cfg.size()); ++i) {
      if (generation_cfg[i]->frequency_penalty != 0.0 ||
          generation_cfg[i]->presence_penalty != 0.0 ||
          generation_cfg[i]->repetition_penalty != 1.0) {
        int num_token_to_process =
            cum_num_token == nullptr ? 1 : (cum_num_token->at(i + 1) - cum_num_token->at(i));
        int token_offset = cum_num_token == nullptr ? i : cum_num_token->at(i);
        CHECK(num_token_to_process == 1 || mstates[i]->draft_output_tokens.empty());
        for (int j = 0; j < num_token_to_process; ++j) {
          p_seq_ids[num_token_for_penalty] = token_offset + j;
          for (auto [token_id, cnt] : mstates[i]->appeared_token_ids) {
            p_pos2seq_id[num_penalty_appeared_token] = num_token_for_penalty;
            p_token_ids[num_penalty_appeared_token] = token_id;
            p_token_cnt[num_penalty_appeared_token] = cnt;
            ++num_penalty_appeared_token;
          }
          p_penalties[num_token_for_penalty * 3] = generation_cfg[i]->presence_penalty;
          p_penalties[num_token_for_penalty * 3 + 1] = generation_cfg[i]->frequency_penalty;
          p_penalties[num_token_for_penalty * 3 + 2] = generation_cfg[i]->repetition_penalty;
          ++num_token_for_penalty;
          if (j > 0) {
            mstates[i]->AddDraftToken(draft_tokens->at(i)[j - 1], NDArray());
          }
        }
        if (num_token_to_process != 1) {
          // Roll back.
          mstates[i]->RemoveAllDraftTokens();
        }
      }
    }

    if (num_token_for_penalty == 0) {
      return;
    }

    // - View arrays.
    int num_seq = num_token_for_penalty;
    int num_token = num_penalty_appeared_token;
    NDArray seq_ids_host = seq_ids_host_.CreateView({num_seq}, dtype_i32_);
    NDArray seq_ids_device = seq_ids_device_.CreateView({num_seq}, dtype_i32_);
    NDArray pos2seq_id_host = pos2seq_id_host_.CreateView({num_token}, dtype_i32_);
    NDArray pos2seq_id_device = pos2seq_id_device_.CreateView({num_token}, dtype_i32_);
    NDArray token_ids_host = token_ids_host_.CreateView({num_token}, dtype_i32_);
    NDArray token_ids_device = token_ids_device_.CreateView({num_token}, dtype_i32_);
    NDArray token_cnt_host = token_cnt_host_.CreateView({num_token}, dtype_i32_);
    NDArray token_cnt_device = token_cnt_device_.CreateView({num_token}, dtype_i32_);
    NDArray penalties_host = penalties_host_.CreateView({num_seq, 3}, dtype_f32_);
    NDArray penalties_device = penalties_device_.CreateView({num_seq, 3}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/seq_ids_host, /*dst=*/seq_ids_device);
    CopyArray(/*src=*/pos2seq_id_host, /*dst=*/pos2seq_id_device);
    CopyArray(/*src=*/token_ids_host, /*dst=*/token_ids_device);
    CopyArray(/*src=*/token_cnt_host, /*dst=*/token_cnt_device);
    CopyArray(/*src=*/penalties_host, /*dst=*/penalties_device);

    // - Call kernel.
    apply_penalty_func_(logits, seq_ids_device, pos2seq_id_device, token_ids_device,
                        token_cnt_device, penalties_device);
    if (trace_recorder_.defined()) {
      TVMSynchronize(device_.device_type, device_.device_id, /*stream=*/nullptr);
    }
  }

  void UpdateWithMask(NDArray logits, const Array<RequestModelState>& mstates,
                      const std::vector<int>* cum_num_token,
                      const std::vector<std::vector<SampleResult>>* draft_tokens) {
    // Construct:
    // - seq_ids (max_num_token,) int32
    // - bitmask (max_num_token, ceildiv(vocab_size, 32)), int32
    int32_t* p_seq_ids = static_cast<int32_t*>(seq_ids_host_->data);
    uint32_t* p_bitmask = static_cast<uint32_t*>(bitmask_host_->data);

    // - Set arrays.
    int batch_size = logits->shape[0];
    ICHECK((cum_num_token == nullptr && batch_size == mstates.size()) ||
           (cum_num_token != nullptr && batch_size == cum_num_token->back()));

    std::memset(p_seq_ids, 0, batch_size * sizeof(int32_t));

    for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
      int token_start_offset = cum_num_token == nullptr ? i : cum_num_token->at(i);
      int token_number =
          cum_num_token == nullptr ? 1 : (cum_num_token->at(i + 1) - cum_num_token->at(i));
      CHECK(token_number == 1 || mstates[i]->draft_output_tokens.empty());
      bool require_mask = mstates[i]->RequireNextTokenBitmask();
      for (int j = 0; j < token_number; ++j) {
        if (require_mask) {
          // Find a slice of bitmask_host_: bitmask_host_[num_token_for_mask, :]
          auto bitmask_dltensor = *bitmask_host_.operator->();
          int64_t bitmask_shape[] = {bitmask_size_};
          bitmask_dltensor.data = p_bitmask + (token_start_offset + j) * bitmask_size_;
          bitmask_dltensor.shape = bitmask_shape;
          bitmask_dltensor.ndim = 1;

          mstates[i]->FindNextTokenBitmask(&bitmask_dltensor);
          p_seq_ids[token_start_offset + j] = 1;
        }
        if (j > 0) {
          mstates[i]->AddDraftToken(draft_tokens->at(i)[j - 1], NDArray());
        }
      }
      if (token_number != 1) {
        // Roll back.
        mstates[i]->RemoveAllDraftTokens();
      }
    }

    int num_token_for_mask = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (p_seq_ids[i] == 1) {
        p_seq_ids[num_token_for_mask] = i;
        ++num_token_for_mask;
      }
    }

    if (num_token_for_mask == 0) {
      return;
    }

    // - View arrays.
    int num_seq = num_token_for_mask;
    NDArray seq_ids_host = seq_ids_host_.CreateView({num_seq}, dtype_i32_);
    NDArray seq_ids_device = seq_ids_device_.CreateView({num_seq}, dtype_i32_);
    NDArray bitmask_host = bitmask_host_.CreateView({batch_size, bitmask_size_}, dtype_i32_);
    NDArray bitmask_device = bitmask_device_.CreateView({batch_size, bitmask_size_}, dtype_i32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/seq_ids_host, /*dst=*/seq_ids_device);
    CopyArray(/*src=*/bitmask_host, /*dst=*/bitmask_device);

    // - Call kernel.
    apply_bitmask_func_(logits, seq_ids_device, bitmask_device);
    if (trace_recorder_.defined()) {
      TVMSynchronize(device_.device_type, device_.device_id, /*stream=*/nullptr);
    }
  }

  // Model configurations
  const int max_num_token_;
  const int vocab_size_;
  const int bitmask_size_;
  const DLDataType dtype_i32_ = DataType::Int(32);
  const DLDataType dtype_u32_ = DataType::UInt(32);
  const DLDataType dtype_f32_ = DataType::Float(32);
  // Packed functions.
  Device device_;
  PackedFunc softmax_func_;
  PackedFunc apply_logit_bias_func_;
  PackedFunc apply_penalty_func_;
  PackedFunc apply_bitmask_func_;
  // Auxiliary NDArrays on CPU
  NDArray seq_ids_host_;
  NDArray pos2seq_id_host_;
  NDArray token_ids_host_;
  NDArray token_cnt_host_;
  NDArray token_logit_bias_host_;
  NDArray penalties_host_;
  NDArray bitmask_host_;
  NDArray temperature_host_;
  // Auxiliary NDArrays on GPU
  NDArray seq_ids_device_;
  NDArray pos2seq_id_device_;
  NDArray token_ids_device_;
  NDArray token_cnt_device_;
  NDArray token_logit_bias_device_;
  NDArray penalties_device_;
  NDArray bitmask_device_;
  NDArray temperature_device_;
  // Event trace recorder.
  Optional<EventTraceRecorder> trace_recorder_;
  // A small epsilon.
  const double eps_ = 1e-5;
};

LogitProcessor::LogitProcessor(int max_num_token, int vocab_size, FunctionTable* ft,
                               DLDevice device, Optional<EventTraceRecorder> trace_recorder) {
  data_ = make_object<LogitProcessorImpl>(max_num_token, vocab_size, ft, device,
                                          std::move(trace_recorder));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
