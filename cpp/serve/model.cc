/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/model.cc
 * \brief The implementation of runtime module of LLM functions (prefill/decode/etc.)
 */
#include "model.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/nvtx.h>

#include <fstream>

#include "../support/json_parser.h"
#include "../support/vlm_utils.h"
#include "config.h"
#include "logit_processor.h"

namespace mlc {
namespace llm {
namespace serve {

/*********************** Model Implementation ***********************/

TVM_FFI_STATIC_INIT_BLOCK() { ModelObj::RegisterReflection(); }

class ModelImpl;

Model Model::Create(String reload_lib_path, String model_path, const picojson::object& model_config,
                    DLDevice device, const Optional<Session>& session, int num_shards,
                    int num_stages, bool trace_enabled) {
  return Model(tvm::ffi::make_object<ModelImpl>(reload_lib_path, model_path, model_config, device,
                                                session, num_shards, num_stages, trace_enabled));
}

Result<picojson::object> Model::LoadModelConfig(const String& model_path) {
  using TResult = Result<picojson::object>;
  picojson::object model_config;
  std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
  std::ostringstream config_ostream;
  TVM_FFI_ICHECK(config_istream);
  config_ostream << config_istream.rdbuf();
  std::string config_str = config_ostream.str();
  picojson::value config_json;
  std::string err = picojson::parse(config_json, config_str);
  if (!err.empty()) {
    return TResult::Error(err);
  }
  picojson::object config = config_json.get<picojson::object>();
  return TResult::Ok(config);
}

class ModelImpl : public ModelObj {
 public:
  /*!
   * \brief Constructor of ModelImpl.
   * \sa Model::Create
   */
  explicit ModelImpl(String reload_lib_path, String model_path, picojson::object model_config,
                     DLDevice device, const Optional<Session>& session, int num_shards,
                     int num_stages, bool trace_enabled)
      : model_(model_path), device_(device), trace_enabled_(trace_enabled) {
    // Step 1. Process model config json string.
    LoadModelConfigJSON(model_config);
    // Step 2. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    this->ft_.Init(reload_lib_path, device_, model_config, session, num_shards, num_stages);
    this->num_shards_ = ft_.model_metadata_.tensor_parallel_shards;
    this->num_stages_ = ft_.model_metadata_.pipeline_parallel_stages;
    this->seqlen_padding_factor_ = ft_.model_metadata_.seqlen_padding_factor;
    // Step 3. Reset
    this->Reset();
    // Step 4. Set model type
    this->kind = GetMetadata().kv_state_kind;
  }

  /*********************** Model Computation  ***********************/

  ObjectRef TokenEmbed(IntTuple token_ids, ObjectRef* dst, int offset) final {
    NVTXScopedRange nvtx_scope("TokenEmbed");
    int num_tokens = token_ids.size();
    if (seqlen_padding_factor_ > 1) {
      num_tokens = (offset + num_tokens + seqlen_padding_factor_ - 1) / seqlen_padding_factor_ *
                   seqlen_padding_factor_;
    }
    // Copy input token ids to device.
    DLDataType dtype(DataType::Int(32));
    Tensor token_ids_nd;
    {
      NVTXScopedRange nvtx_scope("Allocate token_ids at offset");
      token_ids_nd = token_ids_storage_->AllocTensor(offset * 4, {num_tokens}, dtype);
      int* p_token_ids = static_cast<int*>(token_ids_nd->data) + (token_ids_nd->byte_offset) / 4;
      for (int i = 0; i < static_cast<int>(token_ids.size()); ++i) {
        p_token_ids[i] = token_ids[i];
      }
      for (int i = static_cast<int>(token_ids.size()); i < num_tokens; ++i) {
        p_token_ids[i] = 0;
      }
    }
    TVM_FFI_ICHECK_EQ(token_ids_nd->ndim, 1);
    TVM_FFI_ICHECK_EQ(token_ids_nd->shape[0], num_tokens);
    TVM_FFI_ICHECK_NE(prefill_chunk_size_, -1);
    ObjectRef token_ids_dref_or_nd;
    {
      NVTXScopedRange nvtx_scope("Copy to worker 0");
      token_ids_dref_or_nd = ft_.CopyToWorker0(token_ids_nd, "token_ids", {prefill_chunk_size_});
    }

    ObjectRef embeddings = ft_.embed_func_(token_ids_dref_or_nd, params_).cast<ObjectRef>();
    if (dst != nullptr) {
      CHECK(dst->defined());
      ft_.nd_copy_embedding_to_offset_func_(embeddings, *dst, offset);
      return *dst;
    } else {
      CHECK_EQ(offset, 0);
      return embeddings;
    }
  }

  ObjectRef ImageEmbed(const Tensor& image, ObjectRef* dst, int offset) final {
    NVTXScopedRange nvtx_scope("ImageEmbed");
    CHECK(ft_.image_embed_func_.defined()) << "`image_embed` function is not found in the model. ";

    int tmp_h = 0, tmp_w = 0;
    CalculateResizeShape(image, this->model_type_, &tmp_h, &tmp_w);
    Shape resize_h = {tmp_h};
    Shape resize_w = {tmp_w};

    CalculateCropShape(image, this->model_type_, &tmp_h, &tmp_w);
    Shape crop_h = {tmp_h};
    Shape crop_w = {tmp_w};

    auto image_dref_or_nd = ft_.CopyToWorker0(image, "image", image.Shape());
    ObjectRef embeddings =
        ft_.image_embed_func_(image_dref_or_nd, resize_h, resize_w, crop_h, crop_w, params_)
            .cast<ObjectRef>();
    if (dst != nullptr) {
      CHECK(dst->defined());
      ft_.nd_copy_embedding_to_offset_func_(embeddings, *dst, offset);
      return *dst;
    } else {
      CHECK_EQ(offset, 0);
      return embeddings;
    }
  }

  bool CanGetLogits() final {
    return ft_.get_logits_func_.defined() && ft_.batch_get_logits_func_.defined();
  }

  Tensor GetLogits(const ObjectRef& hidden_states) final {
    NVTXScopedRange nvtx_scope("GetLogits");
    CHECK(ft_.get_logits_func_.defined()) << "`get_logits` function is not found in the model.";

    ObjectRef hidden_states_dref_or_nd{nullptr};
    if (!ft_.use_disco && hidden_states->IsInstance<DRefObj>()) {
      hidden_states_dref_or_nd =
          Downcast<DRef>(hidden_states)->DebugGetFromRemote(0).cast<ObjectRef>();
    } else {
      hidden_states_dref_or_nd = hidden_states;
    }
    ObjectRef ret = ft_.get_logits_func_(hidden_states_dref_or_nd, params_).cast<ObjectRef>();
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    Tensor logits{nullptr};
    if (ft_.use_disco) {
      logits = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Tensor>();
    } else {
      logits = Downcast<Tensor>(ret);
    }
    // logits: (b * s, v)
    return logits;
  }

  Array<Tensor> GetMultiStepLogits(const ObjectRef& hidden_states) final {
    NVTXScopedRange nvtx_scope("GetMultiStepLogits");
    CHECK(ft_.get_logits_func_.defined()) << "`get_logits` function is not found in the model.";

    ObjectRef hidden_states_dref_or_nd{nullptr};
    ObjectRef ret = ft_.get_logits_func_(hidden_states, params_).cast<ObjectRef>();
    Array<Tensor> logits{nullptr};
    if (ft_.use_disco) {
      logits = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Array<Tensor>>();
    } else {
      logits = Downcast<Array<Tensor>>(ret);
    }
    return logits;
  }

  ObjectRef FuseEmbedHidden(const ObjectRef& embeddings, const ObjectRef& previous_hidden_states,
                            int batch_size, int seq_len) final {
    NVTXScopedRange nvtx_scope("FuseEmbedHidden");

    ObjectRef embeddings_dref_or_nd{nullptr};
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (n, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], batch_size * seq_len);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({batch_size * seq_len, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{batch_size * seq_len, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }

    ObjectRef previous_hidden_states_dref_or_nd{nullptr};
    if (!ft_.use_disco && previous_hidden_states->IsInstance<DRefObj>()) {
      previous_hidden_states_dref_or_nd =
          Downcast<DRef>(previous_hidden_states)->DebugGetFromRemote(0).cast<ObjectRef>();
    } else {
      previous_hidden_states_dref_or_nd = previous_hidden_states;
    }
    ObjectRef fused = ft_.fuse_embed_hidden_func_(embeddings_dref_or_nd,
                                                  previous_hidden_states_dref_or_nd, params_)
                          .cast<ObjectRef>();
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    Shape out_shape{batch_size, seq_len, hidden_size_};
    if (ft_.use_disco) {
      return ft_.nd_view_func_(fused, out_shape).cast<ObjectRef>();
    } else {
      Tensor fused_nd = Downcast<Tensor>(fused);
      TVM_FFI_ICHECK_EQ(fused_nd->ndim, 2);
      TVM_FFI_ICHECK_EQ(fused_nd->shape[0], batch_size * seq_len);
      return fused_nd.CreateView(out_shape, fused_nd->dtype);
    }
  }

  Tensor BatchPrefill(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                      const std::vector<int>& lengths) final {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;

    int* p_logit_pos = static_cast<int*>(logit_pos_arr_->data);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      p_logit_pos[i] = total_length - 1;
    }
    bool padded = total_length % seqlen_padding_factor_ != 0;
    if (padded) {
      total_length = (total_length + seqlen_padding_factor_ - 1) / seqlen_padding_factor_ *
                     seqlen_padding_factor_;
    }
    NVTXScopedRange nvtx_scope("BatchPrefill num_seq=" + std::to_string(num_sequences) +
                               " total_len=" + std::to_string(total_length));
    Tensor logit_pos_nd = logit_pos_arr_.CreateView({num_sequences}, DataType::Int(32));

    CHECK(ft_.prefill_func_.defined())
        << "`prefill_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], total_length);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }
    TVM_FFI_ICHECK_NE(max_num_sequence_, -1);
    ObjectRef logit_pos_dref_or_nd =
        ft_.CopyToWorker0(logit_pos_nd, "logit_pos", {max_num_sequence_});

    Function single_batch_prefill_func = ft_.single_batch_prefill_func_;
    Function prefill_func = ft_.prefill_func_;
    if (ft_.single_batch_extend_func_.defined()) {
      CHECK(ft_.extend_func_.defined()) << "`batch_extend` function is not found in the model.";
      bool has_existing_sequence = false;
      for (int64_t seq_id : seq_ids) {
        if (prefilled_seq_ids_.count(seq_id)) {
          has_existing_sequence = true;
          break;
        }
      }
      if (has_existing_sequence) {
        single_batch_prefill_func = ft_.single_batch_extend_func_;
        prefill_func = ft_.extend_func_;
      }

      for (int64_t seq_id : seq_ids) {
        prefilled_seq_ids_.insert(seq_id);
      }
    }

    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1 && !padded) {
      ret = single_batch_prefill_func(embeddings_dref_or_nd, kv_cache_, params_).cast<ObjectRef>();
    } else {
      ret = prefill_func(embeddings_dref_or_nd, logit_pos_dref_or_nd, kv_cache_, params_)
                .cast<ObjectRef>();
    }
    Tensor logits;
    if (ft_.use_disco) {
      ret = ft_.tuple_getitem_func_(ret, 0).cast<ObjectRef>();
      if (num_stages_ > 1) {
        // Send the result from the last worker group to worker 0.
        Shape shape{1, num_sequences, vocab_size_};
        DataType dtype = DataType::Float(32);
        ret = ft_.last_group_send_to_worker_0_(ret, disco_logits_arr_, shape, dtype)
                  .cast<ObjectRef>();
      }
      logits = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Tensor>();
    } else {
      logits = Downcast<Array<Tensor>>(ret)[0];
    }
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, num_sequences, v)
    TVM_FFI_ICHECK_EQ(logits->ndim, 3);
    TVM_FFI_ICHECK_EQ(logits->shape[0], 1);
    TVM_FFI_ICHECK_EQ(logits->shape[1], num_sequences);
    return logits;
  }

  ObjectRef BatchPrefillToLastHidden(const ObjectRef& embedding_or_hidden_states,
                                     const std::vector<int64_t>& seq_ids,
                                     const std::vector<int>& lengths) final {
    NVTXScopedRange nvtx_scope("BatchPrefillToLastHidden");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;

    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
    }

    ObjectRef embedding_or_hidden_states_dref_or_nd{nullptr};
    Shape hidden_states_shape{1, total_length, hidden_size_};
    if (!ft_.use_disco) {
      Tensor embedding_or_hidden_states_nd = Downcast<Tensor>(embedding_or_hidden_states);
      embedding_or_hidden_states_dref_or_nd = embedding_or_hidden_states_nd.CreateView(
          hidden_states_shape, embedding_or_hidden_states_nd->dtype);
    } else {
      embedding_or_hidden_states_dref_or_nd =
          ft_.nd_view_func_(embedding_or_hidden_states, hidden_states_shape).cast<ObjectRef>();
    }

    CHECK(ft_.prefill_to_last_hidden_func_.defined())
        << "`prefill_to_last_hidden_states` function is not found in the model.";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef result{nullptr};
    if (seq_ids.size() == 1) {
      CHECK(ft_.single_batch_prefill_to_last_hidden_func_.defined())
          << "`single_batch_prefill_to_last_hidden_states` function is not found in the model.";
      result = ft_.single_batch_prefill_to_last_hidden_func_(embedding_or_hidden_states_dref_or_nd,
                                                             kv_cache_, params_)
                   .cast<ObjectRef>();
    } else {
      result = ft_.prefill_to_last_hidden_func_(embedding_or_hidden_states_dref_or_nd, kv_cache_,
                                                params_)
                   .cast<ObjectRef>();
    }
    ObjectRef hidden_states = ft_.tuple_getitem_func_(result, 0).cast<ObjectRef>();

    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    Shape out_shape{total_length, hidden_size_};
    if (ft_.use_disco) {
      return ft_.nd_view_func_(hidden_states, out_shape).cast<ObjectRef>();
    } else {
      Tensor hidden_states_nd = Downcast<Tensor>(hidden_states);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->ndim, 3);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[0], 1);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[1], total_length);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      return hidden_states_nd.CreateView(out_shape, hidden_states_nd->dtype);
    }
  }

  Tensor BatchDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids) final {
    NVTXScopedRange nvtx_scope("BatchDecode num_seqs=" + std::to_string(seq_ids.size()));
    int num_sequence = seq_ids.size();

    CHECK(ft_.decode_func_.defined())
        << "`decode_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(std::vector<int64_t>(/*n=*/seq_ids.size(), /*v=*/1));
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, b, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], num_sequence);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({num_sequence, 1, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{num_sequence, 1, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }

    // args: embeddings, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1) {
      ret = ft_.single_batch_decode_func_(embeddings_dref_or_nd, kv_cache_, params_)
                .cast<ObjectRef>();
    } else {
      ret = ft_.decode_func_(embeddings_dref_or_nd, kv_cache_, params_).cast<ObjectRef>();
    }
    Tensor logits;
    if (ft_.use_disco) {
      ret = ft_.tuple_getitem_func_(ret, 0).cast<ObjectRef>();
      if (num_stages_ > 1) {
        // Send the result from the last worker group to worker 0.
        Shape shape{num_sequence, 1, vocab_size_};
        DataType dtype = DataType::Float(32);
        ret = ft_.last_group_send_to_worker_0_(ret, disco_logits_arr_, shape, dtype)
                  .cast<ObjectRef>();
      }
      logits = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Tensor>();
    } else {
      logits = Downcast<Array<Tensor>>(ret)[0];
    }
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (b, 1, v)
    TVM_FFI_ICHECK_EQ(logits->ndim, 3);
    TVM_FFI_ICHECK_EQ(logits->shape[0], num_sequence);
    TVM_FFI_ICHECK_EQ(logits->shape[1], 1);
    return logits;
  }

  Tensor BatchTreeDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                         const std::vector<int>& lengths,
                         const std::vector<int64_t>& token_tree_parent_ptr) {
    // This is similar to BatchDecode, except that it takes 'length', so that each sequence can have
    // multiple leaf nodes for decoding.
    NVTXScopedRange nvtx_scope("BatchTreeDecode num_seqs=" + std::to_string(seq_ids.size()));
    int num_sequence = seq_ids.size();
    int total_length = 0;
    for (int i = 0; i < num_sequence; ++i) {
      total_length += lengths[i];
    }
    CHECK_EQ(total_length, token_tree_parent_ptr.size());

    CHECK(ft_.decode_func_.defined())
        << "`tree_decode_with_embed` function is not found in the model. Please make sure the "
           "model "
           "is compiled with flag `--sep-embed` and `--enable-batching`";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    IntTuple token_tree_parent_ptr_tuple(token_tree_parent_ptr);
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple,
                                     token_tree_parent_ptr_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], total_length);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({total_length, 1, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{total_length, 1, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }

    // same as BatchDecode
    ObjectRef ret;
    if (0 && seq_ids.size() == 1) {
      ret = ft_.single_batch_decode_func_(embeddings_dref_or_nd, kv_cache_, params_)
                .cast<ObjectRef>();
    } else {
      ret = ft_.decode_func_(embeddings_dref_or_nd, kv_cache_, params_).cast<ObjectRef>();
    }
    Tensor logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Array<ObjectRef>>();
      logits = Downcast<Tensor>(result[0]);
    } else {
      logits = Downcast<Array<Tensor>>(ret)[0];
    }
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (b, 1, v)
    TVM_FFI_ICHECK_EQ(logits->ndim, 3);
    TVM_FFI_ICHECK_EQ(logits->shape[0], total_length);
    TVM_FFI_ICHECK_EQ(logits->shape[1], 1);
    return logits;
  }

  ObjectRef BatchDecodeToLastHidden(const ObjectRef& hidden_states_dref_or_nd,
                                    const std::vector<int64_t>& seq_ids) final {
    NVTXScopedRange nvtx_scope("BatchDecodeToLastHidden num_seqs=" +
                               std::to_string(seq_ids.size()));
    int num_sequence = seq_ids.size();

    CHECK(ft_.decode_to_last_hidden_func_.defined())
        << "`batch_decode_to_last_hidden_states` function is not found in the model.";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(std::vector<int64_t>(/*n=*/seq_ids.size(), /*v=*/1));
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    // args: embeddings, kv_cache, params
    ObjectRef result{nullptr};
    if (seq_ids.size() == 1) {
      CHECK(ft_.single_batch_decode_to_last_hidden_func_.defined())
          << "`decode_to_last_hidden_states` function is not found in the model.";
      result =
          ft_.single_batch_decode_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_, params_)
              .cast<ObjectRef>();
    } else {
      result = ft_.decode_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_, params_)
                   .cast<ObjectRef>();
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);
    ObjectRef hidden_states = ft_.tuple_getitem_func_(result, 0).cast<ObjectRef>();

    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }

    // hidden_states: (b, 1, v) to (b, v)
    Shape out_shape{num_sequence, hidden_size_};
    if (ft_.use_disco) {
      return ft_.nd_view_func_(hidden_states, out_shape).cast<ObjectRef>();
    } else {
      Tensor hidden_states_nd = Downcast<Tensor>(hidden_states);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->ndim, 3);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[0], num_sequence);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[1], 1);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      return hidden_states_nd.CreateView(out_shape, hidden_states_nd->dtype);
    }
  }

  Tensor BatchVerify(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                     const std::vector<int>& lengths,
                     const std::vector<int64_t>& token_tree_parent_ptr) final {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
    }
    CHECK_EQ(total_length, token_tree_parent_ptr.size());

    NVTXScopedRange nvtx_scope("BatchVerify num_tokens=" + std::to_string(total_length));

    CHECK(ft_.verify_func_.defined())
        << "`verify_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    IntTuple token_tree_parent_ptr_tuple(token_tree_parent_ptr);
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple,
                                     token_tree_parent_ptr_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], total_length);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }
    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret = ft_.verify_func_(embeddings_dref_or_nd, kv_cache_, params_).cast<ObjectRef>();
    Tensor logits;
    if (ft_.use_disco) {
      ret = ft_.tuple_getitem_func_(ret, 0).cast<ObjectRef>();
      if (num_stages_ > 1) {
        // Send the result from the last worker group to worker 0.
        Shape shape{1, total_length, vocab_size_};
        DataType dtype = DataType::Float(32);
        ret = ft_.last_group_send_to_worker_0_(ret, disco_logits_arr_, shape, dtype)
                  .cast<ObjectRef>();
      }
      logits = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<Tensor>();
    } else {
      logits = Downcast<Array<Tensor>>(ret)[0];
    }
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, total_length, v)
    TVM_FFI_ICHECK_EQ(logits->ndim, 3);
    TVM_FFI_ICHECK_EQ(logits->shape[0], 1);
    TVM_FFI_ICHECK_EQ(logits->shape[1], total_length);
    return logits;
  }

  ObjectRef BatchVerifyToLastHidden(const ObjectRef& embeddings,
                                    const std::vector<int64_t>& seq_ids,
                                    const std::vector<int>& lengths,
                                    const std::vector<int64_t>& token_tree_parent_ptr) final {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
    }
    CHECK_EQ(total_length, token_tree_parent_ptr.size());
    NVTXScopedRange nvtx_scope("BatchVerifyToLastHidden num_tokens=" +
                               std::to_string(total_length));

    CHECK(ft_.verify_to_last_hidden_func_.defined())
        << "`batch_verify_to_last_hidden_states` function is not found in the model.";
    TVM_FFI_ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    TVM_FFI_ICHECK(ft_.kv_cache_end_forward_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      Tensor embeddings_nd = Downcast<Tensor>(embeddings);
      TVM_FFI_ICHECK_NE(hidden_size_, -1);
      TVM_FFI_ICHECK_EQ(embeddings_nd->ndim, 2);
      TVM_FFI_ICHECK_GE(embeddings_nd->shape[0], total_length);
      TVM_FFI_ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      TVM_FFI_ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      Shape embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape).cast<ObjectRef>();
    }
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    IntTuple token_tree_parent_ptr_tuple(token_tree_parent_ptr);
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple,
                                     token_tree_parent_ptr_tuple);

    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef result = ft_.verify_to_last_hidden_func_(embeddings_dref_or_nd, kv_cache_, params_)
                           .cast<ObjectRef>();
    ft_.kv_cache_end_forward_func_(kv_cache_);
    ObjectRef hidden_states = ft_.tuple_getitem_func_(result, 0).cast<ObjectRef>();
    if (trace_enabled_) {
      DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    }

    Shape out_shape{total_length, hidden_size_};
    if (!ft_.use_disco) {
      Tensor hidden_states_nd = Downcast<Tensor>(hidden_states);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->ndim, 3);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[0], 1);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[1], total_length);
      TVM_FFI_ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      return hidden_states_nd.CreateView(out_shape, hidden_states_nd->dtype);
    } else {
      return ft_.nd_view_func_(hidden_states, out_shape).cast<ObjectRef>();
    }
  }

  /*********************** KV Cache Management  ***********************/

  void CreateKVCache(int page_size, int max_num_sequence, int64_t max_total_sequence_length,
                     int64_t prefill_chunk_size, int max_history_size) final {
    KVStateKind kv_state_kind = GetMetadata().kv_state_kind;
    if (kv_state_kind == KVStateKind::kKVCache) {
      IntTuple max_num_sequence_tuple{max_num_sequence};
      IntTuple max_total_sequence_length_tuple{max_total_sequence_length};
      IntTuple prefill_chunk_size_tuple{prefill_chunk_size};
      IntTuple page_size_tuple{page_size};
      IntTuple support_sliding_window{sliding_window_size_ != -1};
      kv_cache_ = ft_.create_kv_cache_func_(max_num_sequence_tuple, max_total_sequence_length_tuple,
                                            prefill_chunk_size_tuple, page_size_tuple,
                                            support_sliding_window)
                      .cast<ObjectRef>();
      local_kv_cache_ = ft_.use_disco
                            ? Downcast<DRef>(kv_cache_)->DebugGetFromRemote(0).cast<ObjectRef>()
                            : kv_cache_;
    } else if (kv_state_kind == KVStateKind::kRNNState) {
      IntTuple max_num_sequence_tuple{max_num_sequence};
      IntTuple max_history_size_tuple = {std::max(max_history_size, 1)};
      kv_cache_ = ft_.create_kv_cache_func_(max_num_sequence_tuple, max_history_size_tuple)
                      .cast<ObjectRef>();
      local_kv_cache_ = ft_.use_disco
                            ? Downcast<DRef>(kv_cache_)->DebugGetFromRemote(0).cast<ObjectRef>()
                            : kv_cache_;
    } else if (kv_state_kind == KVStateKind::kNone) {
      // Do nothing
    } else {
      LOG(FATAL) << "Unknown kv_state_kind: " << static_cast<int>(kv_state_kind);
    }
  }

  void AddNewSequence(int64_t seq_id) final {
    if (ft_.model_metadata_.kv_state_kind == KVStateKind::kNone) {
      return;
    }
    ft_.kv_cache_add_sequence_func_(kv_cache_, seq_id);
  }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos) final {
    if (ft_.model_metadata_.kv_state_kind == KVStateKind::kNone) {
      return;
    }
    ft_.kv_cache_fork_sequence_func_(kv_cache_, parent_seq_id, child_seq_id, fork_pos);
    prefilled_seq_ids_.insert(child_seq_id);
  }

  void RemoveSequence(int64_t seq_id) final {
    if (this->kind == KVStateKind::kNone) {
      return;
    }
    prefilled_seq_ids_.erase(seq_id);
    ft_.kv_cache_remove_sequence_func_(kv_cache_, seq_id);
  }

  void PopNFromKVCache(int64_t seq_id, int num_tokens) final {
    if (this->kind == KVStateKind::kNone) {
      return;
    }
    ft_.kv_cache_popn_func_(kv_cache_, seq_id, num_tokens);
  }

  void CommitAcceptedTokenTreeNodesToKVCache(
      const std::vector<int64_t>& seq_ids,
      const std::vector<int64_t>& accepted_leaf_indices) final {
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple accepted_leaf_indices_tuple(accepted_leaf_indices);
    ft_.kv_cache_commit_accepted_token_tree_nodes_func_(kv_cache_, seq_ids_tuple,
                                                        accepted_leaf_indices_tuple);
  }

  void EnableSlidingWindowForSeq(int64_t seq_id) final {
    if (this->kind == KVStateKind::kNone) {
      return;
    }
    if (sliding_window_size_ != -1) {
      ft_.kv_cache_enable_sliding_window_for_seq_(kv_cache_, seq_id, sliding_window_size_,
                                                  attention_sink_size_);
    }
  }

  IntTuple DisaggPrepareKVRecv(int64_t seq_id, int length) final {
    NVTXScopedRange nvtx_scope("DisaggPrepareKVRecv length=" + std::to_string(length));

    TVM_FFI_ICHECK(ft_.kv_cache_disagg_prepare_recv_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Run KV receive preparation.
    ObjectRef ret;
    ret = ft_.kv_cache_disagg_prepare_recv_func_(kv_cache_, seq_id, length).cast<ObjectRef>();
    IntTuple compressed_kv_append_metadata;
    if (ft_.use_disco) {
      compressed_kv_append_metadata = Downcast<DRef>(ret)->DebugGetFromRemote(0).cast<IntTuple>();
    } else {
      compressed_kv_append_metadata = Downcast<IntTuple>(ret);
    }

    return compressed_kv_append_metadata;
  }

  void DisaggMarkKVSend(int64_t seq_id, int begin_pos, IntTuple compressed_kv_append_metadata,
                        int dst_group_offset) final {
    NVTXScopedRange nvtx_scope("DisaggMarkKVSend seq_id=" + std::to_string(seq_id) +
                               " begin_pos=" + std::to_string(begin_pos));

    TVM_FFI_ICHECK(ft_.kv_cache_disagg_mark_send_func_.defined());
    TVM_FFI_ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Run KV send preparation.
    ft_.kv_cache_disagg_mark_send_func_(kv_cache_, seq_id, begin_pos, compressed_kv_append_metadata,
                                        dst_group_offset);
  }

  /************** Raw Info Query **************/

  ModelMetadata GetMetadata() const final { return ft_.model_metadata_; }

  int GetNumAvailablePages() const final {
    if (this->kind == KVStateKind::kRNNState || this->kind == KVStateKind::kNone) {
      // RNNState does not introduce new page at runtime
      return std::numeric_limits<int>::max();
    } else {
      return ft_.kv_cache_get_num_available_pages_func_(local_kv_cache_).cast<int>();
    }
  }

  int GetCurrentTotalSequenceLength() const final {
    if (this->kind == KVStateKind::kRNNState || this->kind == KVStateKind::kNone) {
      // RNNState does not have a total sequence length limit
      return 0;
    } else {
      return ft_.kv_cache_get_total_sequence_length_func_(local_kv_cache_).cast<int>();
    }
  }

  /*********************** Utilities  ***********************/

  void LoadParams() final { this->params_ = ft_.LoadParams(model_, device_); }

  void SetMaxNumSequence(int max_num_sequence) final {
    this->max_num_sequence_ = max_num_sequence;
    this->logit_pos_arr_ =
        Tensor::Empty({max_num_sequence}, DataType::Int(32), Device{DLDeviceType::kDLCPU, 0});
  }

  void SetPrefillChunkSize(int prefill_chunk_size) final {
    this->prefill_chunk_size_ = prefill_chunk_size;
    Device preferred_host_device = GetPreferredHostDevice(device_);
    memory::Allocator* allocator = memory::MemoryManager::GetOrCreateAllocator(
        preferred_host_device, memory::AllocatorType::kNaive);
    ICHECK_NOTNULL(allocator);
    token_ids_storage_ = memory::Storage(
        allocator->Alloc(preferred_host_device, {prefill_chunk_size_}, DataType::Int(32)),
        allocator);
    if (this->num_stages_ > 1) {
      // Create a remote Tensor for logits when pipeline parallelism is enabled.
      disco_logits_arr_ =
          ft_.Empty({prefill_chunk_size_, vocab_size_}, DataType::Float(32), device_,
                    /*worker0_only=*/true);
    }
  }

  LogitProcessor CreateLogitProcessor(int max_num_token,
                                      Optional<EventTraceRecorder> trace_recorder) final {
    return LogitProcessor(max_num_token, vocab_size_, &this->ft_, device_,
                          std::move(trace_recorder));
  }

  Sampler CreateSampler(int max_num_sample, int num_models,
                        Optional<EventTraceRecorder> trace_recorder) final {
    if (Sampler::SupportGPUSampler(device_)) {
      return Sampler::CreateGPUSampler(max_num_sample, vocab_size_, &this->ft_, device_,
                                       std::move(trace_recorder));
    } else {
      return Sampler::CreateCPUSampler(std::move(trace_recorder));
    }
  }

  int EstimateHostCPURequirement() const final {
    CHECK_NE(num_shards_, -1) << "The model has not been initialized";
    return num_shards_ > 1 ? num_shards_ : 0;
  }

  int GetSlidingWindowSize() const final { return sliding_window_size_; }

  int GetAttentionSinkSize() const final { return attention_sink_size_; }

  ObjectRef AllocEmbeddingTensor() final {
    if (!ft_.alloc_embedding_tensor_func_.defined()) {
      return ObjectRef{nullptr};
    }
    // Allocate the embedding tensor.
    ObjectRef embedding = ft_.alloc_embedding_tensor_func_().cast<ObjectRef>();
    // Get the shape of the embedding tensor for hidden size.
    Shape embedding_shape;
    if (ft_.use_disco) {
      TVM_FFI_ICHECK(embedding->IsInstance<DRefObj>());
      ObjectRef shape_ref = ft_.nd_get_shape_func_(embedding).cast<ObjectRef>();
      embedding_shape = Downcast<DRef>(shape_ref)->DebugGetFromRemote(0).cast<Shape>();
    } else {
      Tensor embedding_nd = Downcast<Tensor>(embedding);
      embedding_shape = embedding_nd.Shape();
    }
    TVM_FFI_ICHECK_NE(prefill_chunk_size_, -1);
    TVM_FFI_ICHECK_EQ(embedding_shape.size(), 2);
    TVM_FFI_ICHECK_GE(embedding_shape[0], prefill_chunk_size_);
    this->hidden_size_ = embedding_shape[1];
    return embedding;
  }

  ObjectRef AllocHiddenStatesTensor() final {
    if (!ft_.alloc_embedding_tensor_func_.defined()) {
      return ObjectRef{nullptr};
    }
    // Allocate the hidden_states tensor.
    // Use the same function as embeddings.
    ObjectRef hidden_states = ft_.alloc_embedding_tensor_func_().cast<ObjectRef>();
    Tensor hidden_states_nd{nullptr};
    // Get the shape of the hidden_states tensor for hidden size.
    if (ft_.use_disco) {
      TVM_FFI_ICHECK(hidden_states->IsInstance<DRefObj>());
      hidden_states_nd = Downcast<DRef>(hidden_states)->DebugGetFromRemote(0).cast<Tensor>();
    } else {
      hidden_states_nd = Downcast<Tensor>(hidden_states);
    }
    Shape hidden_states_shape = hidden_states_nd.Shape();
    TVM_FFI_ICHECK_NE(prefill_chunk_size_, -1);
    TVM_FFI_ICHECK_EQ(hidden_states_shape.size(), 2);
    TVM_FFI_ICHECK_GE(hidden_states_shape[0], prefill_chunk_size_);
    this->hidden_size_ = hidden_states_shape[1];
    this->hidden_states_dtype_ = hidden_states_nd->dtype;
    return hidden_states;
  }

  void Reset() final {
    // Reset the KV cache.
    if (kv_cache_.defined()) {
      ft_.reset_kv_cache_func_(kv_cache_);
    }
  }

  /********************** Utilities for speculative decoding **********************/

  DraftTokenWorkspaceManager CreateDraftTokenWorkspaceManager(int max_num_tokens) {
    return DraftTokenWorkspaceManager(max_num_tokens, vocab_size_, hidden_size_,
                                      hidden_states_dtype_, device_, ft_);
  }

  ObjectRef GatherHiddenStates(const ObjectRef& input, const std::vector<int>& indices,
                               ObjectRef* dst) final {
    ObjectRef dst_view{nullptr};
    Shape out_shape{static_cast<int64_t>(indices.size()), hidden_size_};
    if ((*dst)->IsInstance<DRefObj>()) {
      dst_view = ft_.nd_view_func_(*dst, out_shape).cast<ObjectRef>();
    } else {
      Tensor dst_nd = Downcast<Tensor>(*dst);
      dst_view = dst_nd.CreateView(out_shape, hidden_states_dtype_);
    }
    Tensor indices_nd =
        logit_pos_arr_.CreateView({static_cast<int64_t>(indices.size())}, DataType::Int(32));
    indices_nd.CopyFromBytes(indices.data(), indices.size() * sizeof(int));
    TVM_FFI_ICHECK_NE(max_num_sequence_, -1);
    ObjectRef indices_device = ft_.CopyToWorker0(indices_nd, "logit_pos", {max_num_sequence_});
    ft_.gather_hidden_states_func_(input, indices_device, dst_view);
    return dst_view;
  }

  void ScatterHiddenStates(const ObjectRef& input, const std::vector<int>& indices,
                           ObjectRef* dst) final {
    Tensor indices_nd =
        logit_pos_arr_.CreateView({static_cast<int64_t>(indices.size())}, DataType::Int(32));
    indices_nd.CopyFromBytes(indices.data(), indices.size() * sizeof(int));
    TVM_FFI_ICHECK_NE(max_num_sequence_, -1);
    ObjectRef indices_device = ft_.CopyToWorker0(indices_nd, "logit_pos", {max_num_sequence_});
    ft_.scatter_hidden_states_func_(input, indices_device, *dst);
  }

  Tensor GatherDraftProbs(const Tensor& input, const std::vector<int>& indices, Tensor* dst) final {
    Tensor dst_view =
        dst->CreateView({static_cast<int64_t>(indices.size()), vocab_size_}, DataType::Float(32));
    Tensor indices_nd =
        logit_pos_arr_.CreateView({static_cast<int64_t>(indices.size())}, DataType::Int(32));
    indices_nd.CopyFromBytes(indices.data(), indices.size() * sizeof(int));
    TVM_FFI_ICHECK_NE(max_num_sequence_, -1);
    ObjectRef indices_device =
        ft_.CopyToWorker0(indices_nd, "logit_pos_local", {max_num_sequence_}, /*local_only=*/true);
    ft_.gather_probs_func_(input, indices_device, dst_view);
    return dst_view;
  }

  void ScatterDraftProbs(const Tensor& input, const std::vector<int>& indices, Tensor* dst) final {
    Tensor indices_nd =
        logit_pos_arr_.CreateView({static_cast<int64_t>(indices.size())}, DataType::Int(32));
    indices_nd.CopyFromBytes(indices.data(), indices.size() * sizeof(int));
    TVM_FFI_ICHECK_NE(max_num_sequence_, -1);
    ObjectRef indices_device =
        ft_.CopyToWorker0(indices_nd, "logit_pos_local", {max_num_sequence_}, /*local_only=*/true);
    ft_.scatter_probs_func_(input, indices_device, *dst);
  }

  Array<Tensor> GetMedusaLogits(const ObjectRef& hidden_states) {
    ObjectRef result = ft_.get_logits_func_(hidden_states).cast<ObjectRef>();
    Array<Tensor> logits{nullptr};
    if (ft_.use_disco) {
      logits = Downcast<DRef>(result)->DebugGetFromRemote(0).cast<Array<Tensor>>();
    } else {
      logits = Downcast<Array<Tensor>>(result);
    }
    return logits;
  }

  /************** Debug/Profile **************/

  void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) final {
    ft_.DebugCallFuncOnAllAllWorker(func_name, func_args);
  }

 private:
  /*! \brief Load model configuration from JSON. */
  void LoadModelConfigJSON(const picojson::object& config) {
    this->sliding_window_size_ =
        json::LookupOrDefault<int64_t>(config, "sliding_window_size", this->sliding_window_size_);
    CHECK(sliding_window_size_ == -1 || sliding_window_size_ > 0)
        << "Sliding window should be either -1 (which means disabled) of positive";
    this->attention_sink_size_ =
        json::LookupOrDefault<int64_t>(config, "attention_sink_size", this->attention_sink_size_);
    this->attention_sink_size_ = std::max(this->attention_sink_size_, 0);
    this->vocab_size_ = json::Lookup<int64_t>(config, "vocab_size");
    this->model_type_ = json::Lookup<std::string>(config, "model_type");
  }

  //----------------------------
  // Model configurations
  //----------------------------
  std::string model_;
  int sliding_window_size_ = -1;
  int attention_sink_size_ = 0;
  int num_shards_ = -1;
  int num_stages_ = -1;
  int max_num_sequence_ = -1;
  int prefill_chunk_size_ = -1;
  int hidden_size_ = -1;
  DLDataType hidden_states_dtype_;
  int vocab_size_ = -1;
  int image_embed_size_ = -1;
  int seqlen_padding_factor_ = 1;
  std::string model_type_;
  //----------------------------
  // TVM related states
  //----------------------------
  // Packed function table
  FunctionTable ft_;
  // Paged KV cache.
  // - We use `kv_cache_` for general KV cache operations.
  // When tensor parallelism is enabled, `kv_cache_` is a DRef object.
  // - For efficient KV cache raw info query, we use `local_kv_cache`
  // as a local **reference** of `kv_cache_`. It is a pure mirror of `kv_cache_`
  // except that it is always a local object.
  ObjectRef kv_cache_{nullptr};
  ObjectRef local_kv_cache_{nullptr};
  // Runtime device
  Device device_;
  // Model parameters
  ObjectRef params_;
  // Shared Tensor
  memory::Storage token_ids_storage_{nullptr};
  Tensor logit_pos_arr_{nullptr};
  ObjectRef disco_logits_arr_{nullptr};
  // A boolean indicating if tracing is enabled.
  bool trace_enabled_;
  // An enum indicating whether it's RNN-based.
  KVStateKind kind;
  // A set of sequence IDs that have been prefilled.
  std::unordered_set<int64_t> prefilled_seq_ids_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "mlc.copy_embedding_to_offset", [](Tensor embedding, Tensor dst, int offset) {
        // embedding: (m, hidden_size)
        // dst: (prefill_chunk_size, hidden_size)
        TVM_FFI_ICHECK_EQ(embedding->ndim, 2);
        TVM_FFI_ICHECK_EQ(dst->ndim, 2);
        TVM_FFI_ICHECK_LE(embedding->shape[0] + offset, dst->shape[0]);
        TVM_FFI_ICHECK_EQ(embedding->shape[1], dst->shape[1]);
        const DLTensor& copy_src = *(embedding.operator->());
        const DLTensor* p_copy_dst = dst.operator->();
        DLTensor copy_dst = *p_copy_dst;
        copy_dst.shape = embedding->shape;
        copy_dst.byte_offset = offset * embedding->shape[1] *
                               ((embedding->dtype.bits * embedding->dtype.lanes + 7) / 8);
        Tensor::CopyFromTo(&copy_src, &copy_dst);
      });
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
