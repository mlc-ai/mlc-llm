/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/logit_processor.cc
 * \brief The implementation of logit processor.
 */
#include "logit_processor.h"

#include <picojson.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/threading_backend.h>

namespace mlc {
namespace llm {
namespace serve {

inline void CopyArray(Tensor src, Tensor dst, TVMStreamHandle copy_stream) {
  DLTensor dl_dst = *(dst.operator->());
  Tensor::CopyFromTo(src.operator->(), &dl_dst, copy_stream);
}

inline void SyncCopyStream(Device device, TVMStreamHandle compute_stream,
                           TVMStreamHandle copy_stream) {
  // - If there is no particular copy stream, no action is needed.
  if (copy_stream == nullptr) {
    return;
  }
  // - Sync two streams.
  DeviceAPI::Get(device)->SyncStreamFromTo(device, copy_stream, compute_stream);
}

/***************** LogitProcessor Implementation *****************/

TVM_FFI_STATIC_INIT_BLOCK() { LogitProcessorObj::RegisterReflection(); }

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
    Device preferred_host_device = GetPreferredHostDevice(device);
    // Initialize auxiliary arrays on CPU.
    seq_ids_host_ = Tensor::Empty({max_num_token}, dtype_i32_, preferred_host_device);
    pos2seq_id_host_ =
        Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, preferred_host_device);
    token_ids_host_ =
        Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, preferred_host_device);
    token_cnt_host_ =
        Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, preferred_host_device);
    token_logit_bias_host_ =
        Tensor::Empty({max_num_token * vocab_size}, dtype_f32_, preferred_host_device);
    penalties_host_ = Tensor::Empty({max_num_token, 3}, dtype_f32_, preferred_host_device);
    bitmask_host_ =
        Tensor::Empty({max_num_token, bitmask_size_}, dtype_i32_, preferred_host_device);
    temperature_host_ = Tensor::Empty({max_num_token}, dtype_f32_, preferred_host_device);
    // Initialize auxiliary arrays on GPU.
    seq_ids_device_ = Tensor::Empty({max_num_token}, dtype_i32_, device);
    pos2seq_id_device_ = Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_ids_device_ = Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_cnt_device_ = Tensor::Empty({max_num_token * vocab_size}, dtype_i32_, device);
    token_logit_bias_device_ = Tensor::Empty({max_num_token * vocab_size}, dtype_f32_, device);
    penalties_device_ = Tensor::Empty({max_num_token, 3}, dtype_f32_, device);
    bitmask_device_ = Tensor::Empty({max_num_token, bitmask_size_}, dtype_i32_, device);
    temperature_device_ = Tensor::Empty({max_num_token}, dtype_f32_, device);

    CHECK(apply_logit_bias_func_.defined())
        << "Function \"apply_logit_bias_inplace\" not found in model";
    CHECK(apply_penalty_func_.defined()) << "Function \"apply_penalty_inplace\" not found in model";
    CHECK(apply_bitmask_func_.defined()) << "Function \"apply_bitmask_inplace\" not found in model";

    // If the device is CUDA/ROCm, we create a standalone copy stream, in
    // purpose to hide the latency of auxiliary stream copy.
    if (device.device_type == DLDeviceType::kDLCUDA ||
        device.device_type == DLDeviceType::kDLROCM) {
      // The compute stream is the default stream.
      compute_stream_ = DeviceAPI::Get(device)->GetCurrentStream(device);
      copy_stream_ = DeviceAPI::Get(device)->CreateStream(device);
    }
  }

  ~LogitProcessorImpl() {
    // Free the copy stream if defined.
    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, copy_stream_);
    }
  }

  void InplaceUpdateLogits(Tensor logits,                                  //
                           const Array<GenerationConfig>& generation_cfg,  //
                           const Array<RequestModelState>& mstates,        //
                           const Array<String>& request_ids,               //
                           const std::vector<int>* cum_num_token,          //
                           const Array<RequestModelState>* draft_mstates,  //
                           const std::vector<std::vector<int>>* draft_token_indices) final {
    NVTXScopedRange nvtx_scope("Logit inplace update");
    CHECK_EQ(logits->ndim, 2);
    CHECK_EQ(logits->shape[1], vocab_size_);
    CHECK(logits.DataType() == DataType::Float(32));
    CHECK_EQ(generation_cfg.size(), mstates.size());
    CHECK_LE(logits->shape[0], max_num_token_);
    int num_total_token = logits->shape[0];
    int num_sequence = generation_cfg.size();

    CHECK((draft_mstates == nullptr) == (draft_token_indices == nullptr));
    if (cum_num_token != nullptr) {
      TVM_FFI_ICHECK(draft_mstates != nullptr);
      CHECK_EQ(cum_num_token->size(), num_sequence + 1);
      CHECK_EQ(cum_num_token->back(), num_total_token);
    } else {
      CHECK_EQ(num_sequence, num_total_token);
    }

    if (draft_mstates != nullptr) {
      CHECK_EQ(draft_mstates->size(), num_sequence);
      CHECK_EQ(draft_token_indices->size(), num_sequence);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start update logits");

    // Update 1. logit bias
    RECORD_EVENT(trace_recorder_, request_ids, "start apply logit bias");
    UpdateWithLogitBias(logits, generation_cfg, cum_num_token);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply logit bias");

    // Update 2. penalties
    RECORD_EVENT(trace_recorder_, request_ids, "start apply penalty");
    UpdateWithPenalty(logits, generation_cfg, mstates, cum_num_token, draft_mstates,
                      draft_token_indices);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply penalty");

    // Update 3. Vocabulary mask.
    // Note: The mask application must be placed as the last step in logit processor.
    // This is because the masked logits are set to the minimal value.
    // Further logit subtraction may cause issue such as underflow.
    RECORD_EVENT(trace_recorder_, request_ids, "start apply logit mask");
    UpdateWithMask(logits, mstates, cum_num_token, draft_mstates, draft_token_indices);
    RECORD_EVENT(trace_recorder_, request_ids, "finish apply logit mask");

    RECORD_EVENT(trace_recorder_, request_ids, "finish update logits");
  }

  Tensor ComputeProbsFromLogits(Tensor logits, const Array<GenerationConfig>& generation_cfg,
                                const Array<String>& request_ids,
                                const std::vector<int>* cum_num_token) final {
    NVTXScopedRange nvtx_scope("Compute probs from logits");
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
        p_temperature[token_offset + j] = std::max(generation_cfg[i]->temperature, 0.0);
      }
    }

    // - View arrays.
    Tensor temperature_host = temperature_host_.CreateView({num_total_token}, dtype_f32_);
    Tensor temperature_device = temperature_device_.CreateView({num_total_token}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/temperature_host, /*dst=*/temperature_device, copy_stream_);
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    // - Call kernel.
    Tensor probs = softmax_func_(logits.CreateView({num_total_token, 1, vocab_size_}, dtype_f32_),
                                 temperature_device)
                       .cast<Tensor>();
    TVM_FFI_ICHECK_EQ(probs->ndim, 3);
    TVM_FFI_ICHECK_EQ(probs->shape[0], num_total_token);
    TVM_FFI_ICHECK_EQ(probs->shape[1], 1);
    TVM_FFI_ICHECK_EQ(probs->shape[2], vocab_size_);
    if (trace_recorder_.defined()) {
      DeviceAPI::Get(device_)->StreamSync(device_, /*stream=*/nullptr);
    }
    RECORD_EVENT(trace_recorder_, request_ids, "finish softmax");
    return probs.CreateView({num_total_token, vocab_size_}, probs->dtype);
  }

 private:
  void UpdateWithLogitBias(Tensor logits, const Array<GenerationConfig>& generation_cfg,
                           const std::vector<int>* cum_num_token) {
    NVTXScopedRange nvtx_scope("UpdateWithLogitBias");
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
    Tensor pos2seq_id_host = pos2seq_id_host_.CreateView({num_token}, dtype_i32_);
    Tensor pos2seq_id_device = pos2seq_id_device_.CreateView({num_token}, dtype_i32_);
    Tensor token_ids_host = token_ids_host_.CreateView({num_token}, dtype_i32_);
    Tensor token_ids_device = token_ids_device_.CreateView({num_token}, dtype_i32_);
    Tensor token_logit_bias_host = token_logit_bias_host_.CreateView({num_token}, dtype_f32_);
    Tensor token_logit_bias_device = token_logit_bias_device_.CreateView({num_token}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/pos2seq_id_host, /*dst=*/pos2seq_id_device, copy_stream_);
    CopyArray(/*src=*/token_ids_host, /*dst=*/token_ids_device, copy_stream_);
    CopyArray(/*src=*/token_logit_bias_host, /*dst=*/token_logit_bias_device, copy_stream_);
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    // - Call kernel.
    apply_logit_bias_func_(logits, pos2seq_id_device, token_ids_device, token_logit_bias_device);
    if (trace_recorder_.defined()) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
  }

  void UpdateWithPenalty(Tensor logits, const Array<GenerationConfig>& generation_cfg,
                         const Array<RequestModelState>& mstates,
                         const std::vector<int>* cum_num_token,
                         const Array<RequestModelState>* draft_mstates,
                         const std::vector<std::vector<int>>* draft_token_indices) {
    NVTXScopedRange nvtx_scope("UpdateWithPenalty");
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
        TVM_FFI_ICHECK(draft_token_indices == nullptr ||
                       draft_token_indices->at(i).size() == num_token_to_process);
        for (int j = 0; j < num_token_to_process; ++j) {
          p_seq_ids[num_token_for_penalty] = token_offset + j;

          std::vector<SampleResult> draft_token_seq;
          // Update appeared_token_ids with draft tokens
          if (draft_token_indices != nullptr) {
            int cur_draft_token_index = draft_token_indices->at(i)[j];
            while (cur_draft_token_index != -1) {
              draft_token_seq.push_back(
                  (*draft_mstates)[i]->draft_output_tokens[cur_draft_token_index]);
              cur_draft_token_index =
                  (*draft_mstates)[i]->draft_token_parent_idx[cur_draft_token_index];
            }
          }
          auto& appeared_token_ids = mstates[i]->appeared_token_ids;
          for (const auto& token : draft_token_seq) {
            appeared_token_ids[token.GetTokenId()] += 1;
          }
          for (auto [token_id, cnt] : appeared_token_ids) {
            p_pos2seq_id[num_penalty_appeared_token] = num_token_for_penalty;
            p_token_ids[num_penalty_appeared_token] = token_id;
            p_token_cnt[num_penalty_appeared_token] = cnt;
            ++num_penalty_appeared_token;
          }
          for (const auto& token : draft_token_seq) {
            if ((--appeared_token_ids[token.GetTokenId()]) == 0) {
              appeared_token_ids.erase(token.GetTokenId());
            }
          }
          p_penalties[num_token_for_penalty * 3] = generation_cfg[i]->presence_penalty;
          p_penalties[num_token_for_penalty * 3 + 1] = generation_cfg[i]->frequency_penalty;
          p_penalties[num_token_for_penalty * 3 + 2] = generation_cfg[i]->repetition_penalty;
          ++num_token_for_penalty;
        }
      }
    }

    if (num_token_for_penalty == 0) {
      return;
    }

    // - View arrays.
    int num_seq = num_token_for_penalty;
    int num_token = num_penalty_appeared_token;
    Tensor seq_ids_host = seq_ids_host_.CreateView({num_seq}, dtype_i32_);
    Tensor seq_ids_device = seq_ids_device_.CreateView({num_seq}, dtype_i32_);
    Tensor pos2seq_id_host = pos2seq_id_host_.CreateView({num_token}, dtype_i32_);
    Tensor pos2seq_id_device = pos2seq_id_device_.CreateView({num_token}, dtype_i32_);
    Tensor token_ids_host = token_ids_host_.CreateView({num_token}, dtype_i32_);
    Tensor token_ids_device = token_ids_device_.CreateView({num_token}, dtype_i32_);
    Tensor token_cnt_host = token_cnt_host_.CreateView({num_token}, dtype_i32_);
    Tensor token_cnt_device = token_cnt_device_.CreateView({num_token}, dtype_i32_);
    Tensor penalties_host = penalties_host_.CreateView({num_seq, 3}, dtype_f32_);
    Tensor penalties_device = penalties_device_.CreateView({num_seq, 3}, dtype_f32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/seq_ids_host, /*dst=*/seq_ids_device, copy_stream_);
    CopyArray(/*src=*/pos2seq_id_host, /*dst=*/pos2seq_id_device, copy_stream_);
    CopyArray(/*src=*/token_ids_host, /*dst=*/token_ids_device, copy_stream_);
    CopyArray(/*src=*/token_cnt_host, /*dst=*/token_cnt_device, copy_stream_);
    CopyArray(/*src=*/penalties_host, /*dst=*/penalties_device, copy_stream_);
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    // - Call kernel.
    apply_penalty_func_(logits, seq_ids_device, pos2seq_id_device, token_ids_device,
                        token_cnt_device, penalties_device);
    if (trace_recorder_.defined()) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
  }

  void UpdateWithMask(Tensor logits, const Array<RequestModelState>& mstates,
                      const std::vector<int>* cum_num_token,
                      const Array<RequestModelState>* draft_mstates,
                      const std::vector<std::vector<int>>* draft_token_indices) {
    NVTXScopedRange nvtx_scope("UpdateWithMask");
    // Construct:
    // - seq_ids (max_num_token,) int32
    // - bitmask (max_num_token, ceildiv(vocab_size, 32)), int32
    int32_t* p_seq_ids = static_cast<int32_t*>(seq_ids_host_->data);
    uint32_t* p_bitmask = static_cast<uint32_t*>(bitmask_host_->data);

    // - Set arrays.
    int batch_size = logits->shape[0];
    TVM_FFI_ICHECK((cum_num_token == nullptr && batch_size == mstates.size()) ||
                   (cum_num_token != nullptr && batch_size == cum_num_token->back()));

    std::memset(p_seq_ids, 0, batch_size * sizeof(int32_t));

    for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
      int token_start_offset = cum_num_token == nullptr ? i : cum_num_token->at(i);
      int token_number =
          cum_num_token == nullptr ? 1 : (cum_num_token->at(i + 1) - cum_num_token->at(i));
      bool require_mask = mstates[i]->RequireNextTokenBitmask();
      TVM_FFI_ICHECK(draft_token_indices == nullptr ||
                     draft_token_indices->at(i).size() == token_number);
      for (int j = 0; j < token_number; ++j) {
        if (require_mask) {
          std::vector<SampleResult> draft_token_seq;
          if (draft_token_indices != nullptr) {
            int cur_draft_token_index = draft_token_indices->at(i)[j];
            while (cur_draft_token_index != -1) {
              draft_token_seq.push_back(
                  (*draft_mstates)[i]->draft_output_tokens[cur_draft_token_index]);
              cur_draft_token_index =
                  (*draft_mstates)[i]->draft_token_parent_idx[cur_draft_token_index];
            }
            for (auto it = draft_token_seq.rbegin(); it != draft_token_seq.rend(); ++it) {
              mstates[i]->grammar_matcher.value().AcceptToken(it->GetTokenId());
            }
          }
          // Find a slice of bitmask_host_: bitmask_host_[num_token_for_mask, :]
          DLTensor bitmask_dltensor = *bitmask_host_.operator->();
          int64_t bitmask_shape[] = {bitmask_size_};
          bitmask_dltensor.data = p_bitmask + (token_start_offset + j) * bitmask_size_;
          bitmask_dltensor.shape = bitmask_shape;
          bitmask_dltensor.ndim = 1;

          mstates[i]->GetNextTokenBitmask(&bitmask_dltensor);
          p_seq_ids[token_start_offset + j] = 1;

          if (draft_token_seq.size() > 0) {
            mstates[i]->grammar_matcher.value().Rollback(draft_token_seq.size());
          }
        }
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
    Tensor seq_ids_host = seq_ids_host_.CreateView({num_seq}, dtype_i32_);
    Tensor seq_ids_device = seq_ids_device_.CreateView({num_seq}, dtype_i32_);
    Tensor bitmask_host = bitmask_host_.CreateView({batch_size, bitmask_size_}, dtype_i32_);
    Tensor bitmask_device = bitmask_device_.CreateView({batch_size, bitmask_size_}, dtype_i32_);

    // - Copy arrays to GPU.
    CopyArray(/*src=*/seq_ids_host, /*dst=*/seq_ids_device, copy_stream_);
    CopyArray(/*src=*/bitmask_host, /*dst=*/bitmask_device, copy_stream_);
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    // - Call kernel.
    apply_bitmask_func_(logits, seq_ids_device, bitmask_device);
    if (trace_recorder_.defined()) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
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
  Function softmax_func_;
  Function apply_logit_bias_func_;
  Function apply_penalty_func_;
  Function apply_bitmask_func_;
  // Auxiliary Tensors on CPU
  Tensor seq_ids_host_;
  Tensor pos2seq_id_host_;
  Tensor token_ids_host_;
  Tensor token_cnt_host_;
  Tensor token_logit_bias_host_;
  Tensor penalties_host_;
  Tensor bitmask_host_;
  Tensor temperature_host_;
  // Auxiliary Tensors on GPU
  Tensor seq_ids_device_;
  Tensor pos2seq_id_device_;
  Tensor token_ids_device_;
  Tensor token_cnt_device_;
  Tensor token_logit_bias_device_;
  Tensor penalties_device_;
  Tensor bitmask_device_;
  Tensor temperature_device_;
  // Event trace recorder.
  Optional<EventTraceRecorder> trace_recorder_;
  // The device stream for the default computation operations.
  TVMStreamHandle compute_stream_ = nullptr;
  // The device stream for copying auxiliary data structure to GPU.
  TVMStreamHandle copy_stream_ = nullptr;
  // A small epsilon.
  const double eps_ = 1e-5;
};

LogitProcessor::LogitProcessor(int max_num_token, int vocab_size, FunctionTable* ft,
                               DLDevice device, Optional<EventTraceRecorder> trace_recorder) {
  data_ = tvm::ffi::make_object<LogitProcessorImpl>(max_num_token, vocab_size, ft, device,
                                                    std::move(trace_recorder));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
