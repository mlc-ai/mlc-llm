/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.cc
 * \brief The implementation of runtime module of LLM functions (prefill/decode/etc.)
 */
#include "model.h"

#include <picojson.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include "logit_processor.h"

namespace mlc {
namespace llm {
namespace serve {

/*********************** Model Implementation ***********************/

class ModelImpl;

TVM_REGISTER_OBJECT_TYPE(ModelObj);

Model Model::Create(String reload_lib_path, String model_path, DLDevice device,
                    int max_num_sequence, bool trace_enabled) {
  return Model(
      make_object<ModelImpl>(reload_lib_path, model_path, device, max_num_sequence, trace_enabled));
}

class ModelImpl : public ModelObj {
 public:
  /*!
   * \brief Constructor of ModelImpl.
   * \sa Model::Create
   */
  explicit ModelImpl(String reload_lib_path, String model_path, DLDevice device,
                     int max_num_sequence, bool trace_enabled)
      : device_(device) {
    // Step 1. Process model config json string.
    picojson::object model_config;
    {
      std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
      std::ostringstream config_ostream;
      ICHECK(config_istream);
      config_ostream << config_istream.rdbuf();
      std::string config_str = config_ostream.str();
      model_config = LoadModelConfigJSON(config_str);
    }
    // Step 2. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    this->ft_.Init(reload_lib_path, device_, model_config);
    // Step 3. Load params in nd-array cache.
    this->params_ = ft_.LoadParams(model_path, device_);
    // Step 4. Set max_num_sequence
    this->max_num_sequence_ = max_num_sequence;
    // Step 5. Reset
    this->Reset();
    // Step 6. Initialize the shared NDArray.
    Device device_host{DLDeviceType::kDLCPU, 0};
    memory::Allocator* allocator =
        memory::MemoryManager::GetOrCreateAllocator(device_host, memory::AllocatorType::kNaive);
    ICHECK_NOTNULL(allocator);
    token_ids_storage_ = memory::Storage(
        allocator->Alloc(device_host, {prefill_chunk_size_}, DataType::Int(32)), allocator);
    this->logit_pos_arr_ = NDArray::Empty({max_num_sequence}, DataType::Int(32), device_host);
  }

  /*********************** Model Computation  ***********************/

  ObjectRef TokenEmbed(IntTuple token_ids, ObjectRef* dst, int offset) final {
    NVTXScopedRange nvtx_scope("TokenEmbed");
    int num_tokens = token_ids.size();
    // Copy input token ids to device.
    DLDataType dtype(DataType::Int(32));
    NDArray token_ids_nd;
    {
      NVTXScopedRange nvtx_scope("Allocate token_ids at offset");
      token_ids_nd = token_ids_storage_->AllocNDArray(offset * 4, {num_tokens}, dtype);
      int* p_token_ids = static_cast<int*>(token_ids_nd->data) + (token_ids_nd->byte_offset) / 4;
      for (int i = 0; i < num_tokens; ++i) {
        p_token_ids[i] = token_ids[i];
      }
    }
    ICHECK_EQ(token_ids_nd->ndim, 1);
    ICHECK_EQ(token_ids_nd->shape[0], num_tokens);
    auto token_ids_dref_or_nd = ft_.CopyToWorker0(token_ids_nd, "token_ids", {prefill_chunk_size_});

    ObjectRef embeddings = ft_.embed_func_(token_ids_dref_or_nd, params_);
    if (dst != nullptr) {
      CHECK(dst->defined());
      ft_.nd_copy_embedding_to_offset_func_(embeddings, *dst, offset);
      return *dst;
    } else {
      CHECK_EQ(offset, 0);
      return embeddings;
    }
  }

  ObjectRef ImageEmbed(const NDArray& image, ObjectRef* dst, int offset) final {
    NVTXScopedRange nvtx_scope("ImageEmbed");
    CHECK(ft_.image_embed_func_.defined()) << "`image_embed` function is not found in the model. ";
    auto image_dref_or_nd = ft_.CopyToWorker0(image, "image", image.Shape());
    ObjectRef embeddings = ft_.image_embed_func_(image_dref_or_nd, params_);
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

  NDArray GetLogits(const ObjectRef& last_hidden_states, int batch_size, int seq_len) final {
    NVTXScopedRange nvtx_scope("GetLogits");
    CHECK(ft_.get_logits_func_.defined()) << "`get_logits` function is not found in the model.";

    ObjectRef hidden_states_dref_or_nd;
    CHECK(!last_hidden_states->IsInstance<DRefObj>());
    // hidden_states: (b, s, h)
    NDArray hidden_states = Downcast<NDArray>(last_hidden_states);
    ICHECK_NE(hidden_size_, -1);
    ICHECK_EQ(hidden_states->ndim, 3);
    ICHECK_EQ(hidden_states->shape[0], batch_size);
    ICHECK_EQ(hidden_states->shape[1], seq_len);
    ICHECK_EQ(hidden_states->shape[2], hidden_size_);
    ICHECK_EQ(hidden_states->device.device_type, device_.device_type);
    ICHECK_EQ(hidden_states->device.device_id, device_.device_id);

    hidden_states_dref_or_nd =
        hidden_states.CreateView({batch_size * seq_len, hidden_size_}, hidden_states->dtype);

    ObjectRef ret = ft_.get_logits_func_(hidden_states_dref_or_nd, params_);
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }

    NDArray logits;
    logits = Downcast<NDArray>(ret);
    CHECK(logits.defined());
    // logits: (b * s, v)
    ICHECK_EQ(logits->ndim, 2);
    ICHECK_EQ(logits->shape[0], batch_size * seq_len);
    return logits.CreateView({batch_size, seq_len, logits->shape[1]}, logits->dtype);
  }

  NDArray BatchGetLogits(const ObjectRef& last_hidden_states, const std::vector<int64_t>& seq_ids,
                         const std::vector<int>& lengths) {
    NVTXScopedRange nvtx_scope("BatchGetLogits");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;

    int* p_logit_pos = static_cast<int*>(logit_pos_arr_->data);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      p_logit_pos[i] = total_length - 1;
    }
    NDArray logit_pos_nd = logit_pos_arr_.CreateView({num_sequences}, DataType::Int(32));
    ObjectRef logit_pos_dref_or_nd =
        ft_.CopyToWorker0(logit_pos_nd, "logit_pos", {max_num_sequence_});

    CHECK(ft_.batch_get_logits_func_.defined())
        << "`batch_get_logits` function is not found in the model.";

    ObjectRef hidden_states_dref_or_nd;
    CHECK(!last_hidden_states->IsInstance<DRefObj>());
    // hidden_states: (b, s, h)
    NDArray hidden_states = Downcast<NDArray>(last_hidden_states);
    ICHECK_NE(hidden_size_, -1);
    ICHECK_EQ(hidden_states->ndim, 3);
    ICHECK_EQ(hidden_states->shape[0], 1);
    ICHECK_EQ(hidden_states->shape[1], total_length);
    ICHECK_EQ(hidden_states->shape[2], hidden_size_);
    ICHECK_EQ(hidden_states->device.device_type, device_.device_type);
    ICHECK_EQ(hidden_states->device.device_id, device_.device_id);

    hidden_states_dref_or_nd =
        hidden_states.CreateView({total_length, hidden_size_}, hidden_states->dtype);

    ObjectRef ret =
        ft_.batch_get_logits_func_(hidden_states_dref_or_nd, logit_pos_dref_or_nd, params_);
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }

    NDArray logits;
    logits = Downcast<NDArray>(ret);
    CHECK(logits.defined());
    // logits: (b * s, v)
    ICHECK_EQ(logits->ndim, 2);
    ICHECK_EQ(logits->shape[0], num_sequences);
    return logits.CreateView({1, num_sequences, logits->shape[1]}, logits->dtype);
  }

  NDArray BatchSelectLastHidden(const ObjectRef& last_hidden_states,
                                const std::vector<int64_t>& seq_ids,
                                const std::vector<int>& lengths) {
    NVTXScopedRange nvtx_scope("BatchSelectLastHidden");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;

    int* p_logit_pos = static_cast<int*>(logit_pos_arr_->data);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      p_logit_pos[i] = total_length - 1;
    }
    NDArray logit_pos_nd = logit_pos_arr_.CreateView({num_sequences}, DataType::Int(32));
    ObjectRef logit_pos_dref_or_nd =
        ft_.CopyToWorker0(logit_pos_nd, "logit_pos", {max_num_sequence_});

    CHECK(ft_.batch_select_last_hidden_func_.defined())
        << "`batch_select_last_hidden_states` function is not found in the model.";

    ObjectRef hidden_states_dref_or_nd;
    CHECK(!last_hidden_states->IsInstance<DRefObj>());
    // hidden_states: (b, s, h)
    NDArray hidden_states = Downcast<NDArray>(last_hidden_states);
    ICHECK_NE(hidden_size_, -1);
    ICHECK_EQ(hidden_states->ndim, 3);
    ICHECK_EQ(hidden_states->shape[0], 1);
    ICHECK_EQ(hidden_states->shape[1], total_length);
    ICHECK_EQ(hidden_states->shape[2], hidden_size_);
    ICHECK_EQ(hidden_states->device.device_type, device_.device_type);
    ICHECK_EQ(hidden_states->device.device_id, device_.device_id);

    hidden_states_dref_or_nd =
        hidden_states.CreateView({total_length, hidden_size_}, hidden_states->dtype);

    ObjectRef ret =
        ft_.batch_select_last_hidden_func_(hidden_states_dref_or_nd, logit_pos_dref_or_nd, params_);
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }

    NDArray hidden;
    hidden = Downcast<NDArray>(ret);
    // hidden: (b * s, v)
    ICHECK_EQ(hidden->ndim, 2);
    ICHECK_EQ(hidden->shape[0], num_sequences);
    return hidden.CreateView({1, num_sequences, hidden->shape[1]}, hidden->dtype);
  }

  NDArray ConcatLastHidden(std::vector<NDArray>& hidden_states, ObjectRef* dst) final {
    NVTXScopedRange nvtx_scope("ConcatLastHidden");

    CHECK(dst->defined());

    int cum_length = 0;
    ICHECK_GE(hidden_states.size(), 1);
    for (auto hidden : hidden_states) {
      ICHECK_EQ(hidden->ndim, 1);
      // No ICHECK_EQ(hidden->shape[0], hidden_size_) here to allow different hidden_sizes.
      hidden = hidden.CreateView({1, hidden_size_}, hidden->dtype);
      // Reuse the copy embedding function
      ft_.nd_copy_embedding_to_offset_func_(hidden, *dst, cum_length);
      cum_length += 1;
    }
    NDArray ret = Downcast<NDArray>(*dst);
    ret = ret.CreateView({cum_length, hidden_size_}, hidden_states[0]->dtype);
    return ret;
  }

  ObjectRef FuseEmbedHidden(const ObjectRef& embeddings, const ObjectRef& previous_hidden_states,
                            int batch_size, int seq_len) final {
    NVTXScopedRange nvtx_scope("FuseEmbedHidden");

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (n, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_NE(hidden_size_, -1);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_GE(embeddings_nd->shape[0], batch_size * seq_len);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({batch_size * seq_len, hidden_size_}, embeddings_nd->dtype);

      if (!ft_.fuse_embed_hidden_func_.defined() || !previous_hidden_states.defined()) {
        // Model has no support for fuse_embed_hidden_states or this is the first model (base model)
        return embeddings_nd.CreateView({batch_size, seq_len, hidden_size_}, embeddings_nd->dtype);
      }
    } else {
      ShapeTuple embedding_shape{batch_size, seq_len, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape);

      if (!ft_.fuse_embed_hidden_func_.defined() || !previous_hidden_states.defined()) {
        // Model has no support for fuse_embed_hidden_states or this is the first model (base model)
        ShapeTuple embedding_shape{batch_size, seq_len, hidden_size_};
        return ft_.nd_view_func_(embeddings, embedding_shape);
      }
    }

    NDArray hidden_states = Downcast<NDArray>(previous_hidden_states);
    CHECK(hidden_states.defined());
    ICHECK_EQ(hidden_states->ndim, 3);
    ICHECK_EQ(hidden_states->shape[0], batch_size);
    ICHECK_EQ(hidden_states->shape[1], seq_len);
    ICHECK_EQ(hidden_states->shape[2], hidden_size_);
    ICHECK_EQ(hidden_states->device.device_type, device_.device_type);
    ICHECK_EQ(hidden_states->device.device_id, device_.device_id);
    NDArray hidden_states_2d =
        hidden_states.CreateView({batch_size * seq_len, hidden_size_}, hidden_states->dtype);
    auto hidden_states_dref_or_nd =
        ft_.CopyToWorker0(hidden_states_2d, "hidden_states_2d",
                          {max_num_sequence_ * prefill_chunk_size_, hidden_size_});

    ObjectRef ret =
        ft_.fuse_embed_hidden_func_(embeddings_dref_or_nd, hidden_states_dref_or_nd, params_);
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    if (!ret->IsInstance<DRefObj>()) {
      NDArray fused = Downcast<NDArray>(ret);
      return fused.CreateView({batch_size, seq_len, hidden_size_}, fused->dtype);
    } else {
      ShapeTuple fused_shape{batch_size, seq_len, hidden_size_};
      return ft_.nd_view_func_(ret, fused_shape);
    }
  }

  NDArray BatchPrefill(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                       const std::vector<int>& lengths) final {
    NVTXScopedRange nvtx_scope("BatchPrefill");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;

    int* p_logit_pos = static_cast<int*>(logit_pos_arr_->data);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      p_logit_pos[i] = total_length - 1;
    }
    NDArray logit_pos_nd = logit_pos_arr_.CreateView({num_sequences}, DataType::Int(32));

    CHECK(ft_.prefill_func_.defined())
        << "`prefill_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_NE(hidden_size_, -1);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_GE(embeddings_nd->shape[0], total_length);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape);
    }
    ObjectRef logit_pos_dref_or_nd =
        ft_.CopyToWorker0(logit_pos_nd, "logit_pos", {max_num_sequence_});
    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1) {
      ret = ft_.single_batch_prefill_func_(embeddings_dref_or_nd, kv_cache_, params_);
    } else {
      ret = ft_.prefill_func_(embeddings_dref_or_nd, logit_pos_dref_or_nd, kv_cache_, params_);
    }
    NDArray logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      logits = Downcast<NDArray>(result[0]);
    } else {
      logits = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, num_sequences, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], num_sequences);
    return logits;
  }

  NDArray BatchPrefillToLastHidden(const ObjectRef& hidden_states,
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

    ObjectRef hidden_states_dref_or_nd;
    if (!hidden_states->IsInstance<DRefObj>()) {
      // hidden_states: (1, n, h)
      NDArray hidden_states_nd = Downcast<NDArray>(hidden_states);
      ICHECK_EQ(hidden_states_nd->ndim, 3);
      ICHECK_EQ(hidden_states_nd->shape[0], 1);
      ICHECK_EQ(hidden_states_nd->shape[1], total_length);
      ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      hidden_states_dref_or_nd =
          hidden_states_nd.CreateView({1, total_length, hidden_size_}, hidden_states_nd->dtype);
    } else {
      ShapeTuple hidden_states_shape{1, total_length, hidden_size_};
      hidden_states_dref_or_nd = ft_.nd_view_func_(hidden_states, hidden_states_shape);
    }

    CHECK(ft_.prefill_to_last_hidden_func_.defined())
        << "`prefill_to_last_hidden_states` function is not found in the model.";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1) {
      CHECK(ft_.single_batch_prefill_to_last_hidden_func_.defined())
          << "`single_batch_prefill_to_last_hidden_states` function is not found in the model.";
      ret = ft_.single_batch_prefill_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_,
                                                          params_);
    } else {
      ret = ft_.prefill_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_, params_);
    }
    NDArray last_hidden_states;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      last_hidden_states = Downcast<NDArray>(result[0]);
    } else {
      last_hidden_states = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // hidden_states: (1, total_length, v)
    ICHECK_EQ(last_hidden_states->ndim, 3);
    ICHECK_EQ(last_hidden_states->shape[0], 1);
    ICHECK_EQ(last_hidden_states->shape[1], total_length);
    return last_hidden_states;
  }

  NDArray BatchDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids) final {
    NVTXScopedRange nvtx_scope("BatchDecode");
    int num_sequence = seq_ids.size();

    CHECK(ft_.decode_func_.defined())
        << "`decode_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(std::vector<int64_t>(/*n=*/seq_ids.size(), /*v=*/1));
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, b, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_NE(hidden_size_, -1);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_GE(embeddings_nd->shape[0], num_sequence);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({num_sequence, 1, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{num_sequence, 1, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape);
    }

    // args: embeddings, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1) {
      ret = ft_.single_batch_decode_func_(embeddings_dref_or_nd, kv_cache_, params_);
    } else {
      ret = ft_.decode_func_(embeddings_dref_or_nd, kv_cache_, params_);
    }
    NDArray logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      logits = Downcast<NDArray>(result[0]);
    } else {
      logits = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (b, 1, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], num_sequence);
    ICHECK_EQ(logits->shape[1], 1);
    return logits;
  }

  NDArray BatchDecodeToLastHidden(const ObjectRef& hidden_states,
                                  const std::vector<int64_t>& seq_ids) final {
    NVTXScopedRange nvtx_scope("BatchDecodeToLastHidden");
    int num_sequence = seq_ids.size();

    CHECK(ft_.decode_to_last_hidden_func_.defined())
        << "`batch_decode_to_last_hidden_states` function is not found in the model.";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    ObjectRef hidden_states_dref_or_nd;
    if (!hidden_states->IsInstance<DRefObj>()) {
      // hidden_states: (1, n, h)
      NDArray hidden_states_nd = Downcast<NDArray>(hidden_states);
      ICHECK_EQ(hidden_states_nd->ndim, 3);
      ICHECK_EQ(hidden_states_nd->shape[0], num_sequence);
      ICHECK_EQ(hidden_states_nd->shape[1], 1);
      ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      hidden_states_dref_or_nd =
          hidden_states_nd.CreateView({num_sequence, 1, hidden_size_}, hidden_states_nd->dtype);
    } else {
      ShapeTuple hidden_states_shape{num_sequence, 1, hidden_size_};
      hidden_states_dref_or_nd = ft_.nd_view_func_(hidden_states, hidden_states_shape);
    }

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(std::vector<int64_t>(/*n=*/seq_ids.size(), /*v=*/1));
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    // args: embeddings, kv_cache, params
    ObjectRef ret;
    if (seq_ids.size() == 1) {
      CHECK(ft_.single_batch_decode_to_last_hidden_func_.defined())
          << "`decode_to_last_hidden_states` function is not found in the model.";
      ret = ft_.single_batch_decode_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_,
                                                         params_);
    } else {
      ret = ft_.decode_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_, params_);
    }
    NDArray last_hidden_states;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      last_hidden_states = Downcast<NDArray>(result[0]);
    } else {
      last_hidden_states = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // hidden_states: (b, 1, v)
    ICHECK_EQ(last_hidden_states->ndim, 3);
    ICHECK_EQ(last_hidden_states->shape[0], num_sequence);
    ICHECK_EQ(last_hidden_states->shape[1], 1);
    return last_hidden_states;
  }

  NDArray BatchVerify(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                      const std::vector<int>& lengths) final {
    NVTXScopedRange nvtx_scope("BatchVerify");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
    }

    CHECK(ft_.verify_func_.defined())
        << "`verify_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (1, n, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_NE(hidden_size_, -1);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_GE(embeddings_nd->shape[0], total_length);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.nd_view_func_(embeddings, embedding_shape);
    }
    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret = ft_.verify_func_(embeddings_dref_or_nd, kv_cache_, params_);
    NDArray logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      logits = Downcast<NDArray>(result[0]);
    } else {
      logits = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, total_length, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], total_length);
    return logits;
  }

  NDArray BatchVerifyToLastHidden(const ObjectRef& hidden_states,
                                  const std::vector<int64_t>& seq_ids,
                                  const std::vector<int>& lengths) final {
    NVTXScopedRange nvtx_scope("BatchVerifyToLastHidden");
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
    }

    CHECK(ft_.verify_to_last_hidden_func_.defined())
        << "`batch_verify_to_last_hidden_states` function is not found in the model.";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    ObjectRef hidden_states_dref_or_nd;
    if (!hidden_states->IsInstance<DRefObj>()) {
      // hidden_states: (1, n, h)
      NDArray hidden_states_nd = Downcast<NDArray>(hidden_states);
      ICHECK_EQ(hidden_states_nd->ndim, 3);
      ICHECK_EQ(hidden_states_nd->shape[0], 1);
      ICHECK_EQ(hidden_states_nd->shape[1], total_length);
      ICHECK_EQ(hidden_states_nd->shape[2], hidden_size_);
      hidden_states_dref_or_nd =
          hidden_states_nd.CreateView({1, total_length, hidden_size_}, hidden_states_nd->dtype);
    } else {
      ShapeTuple hidden_states_shape{1, total_length, hidden_size_};
      hidden_states_dref_or_nd = ft_.nd_view_func_(hidden_states, hidden_states_shape);
    }

    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(lengths.begin(), lengths.end());
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret = ft_.verify_to_last_hidden_func_(hidden_states_dref_or_nd, kv_cache_, params_);
    NDArray last_hidden_states;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      last_hidden_states = Downcast<NDArray>(result[0]);
    } else {
      last_hidden_states = Downcast<Array<NDArray>>(ret)[0];
    }
    if (trace_enabled_) {
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    }
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // hidden_states: (1, total_length, v)
    ICHECK_EQ(last_hidden_states->ndim, 3);
    ICHECK_EQ(last_hidden_states->shape[0], 1);
    ICHECK_EQ(last_hidden_states->shape[1], total_length);
    return last_hidden_states;
  }

  /*********************** KV Cache Management  ***********************/

  void CreateKVCache(int page_size, int max_num_sequence, int max_total_sequence_length,
                     int prefill_chunk_size) final {
    IntTuple max_num_sequence_tuple{max_num_sequence};
    IntTuple max_total_sequence_length_tuple{max_total_sequence_length};
    IntTuple prefill_chunk_size_tuple{prefill_chunk_size};
    IntTuple page_size_tuple{page_size};
    IntTuple support_sliding_window{sliding_window_size_ != -1};
    kv_cache_ = ft_.create_kv_cache_func_(max_num_sequence_tuple, max_total_sequence_length_tuple,
                                          prefill_chunk_size_tuple, page_size_tuple,
                                          support_sliding_window);
    local_kv_cache_ = ft_.use_disco ? Downcast<DRef>(kv_cache_)->DebugGetFromRemote(0) : kv_cache_;
  }

  void AddNewSequence(int64_t seq_id) final { ft_.kv_cache_add_sequence_func_(kv_cache_, seq_id); }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos) final {
    ft_.kv_cache_fork_sequence_func_(kv_cache_, parent_seq_id, child_seq_id, fork_pos);
  }

  void RemoveSequence(int64_t seq_id) final {
    ft_.kv_cache_remove_sequence_func_(kv_cache_, seq_id);
  }

  void PopNFromKVCache(int64_t seq_id, int num_tokens) final {
    ft_.kv_cache_popn_func_(kv_cache_, seq_id, num_tokens);
  }

  void EnableSlidingWindowForSeq(int64_t seq_id) final {
    if (sliding_window_size_ != -1) {
      ft_.kv_cache_enable_sliding_window_for_seq_(kv_cache_, seq_id, sliding_window_size_,
                                                  attention_sink_size_);
    }
  }

  /************** Raw Info Query **************/

  int GetNumAvailablePages() const final {
    return ft_.kv_cache_get_num_available_pages_func_(local_kv_cache_);
  }

  int GetCurrentTotalSequenceLength() const final {
    return ft_.kv_cache_get_total_sequence_length_func_(local_kv_cache_);
  }

  /*********************** Utilities  ***********************/

  LogitProcessor CreateLogitProcessor(int max_num_token,
                                      Optional<EventTraceRecorder> trace_recorder) {
    return LogitProcessor(max_num_token, vocab_size_, &this->ft_, device_,
                          std::move(trace_recorder));
  }

  Sampler CreateSampler(int max_num_sample, int num_models,
                        Optional<EventTraceRecorder> trace_recorder) {
    if (num_models > 1) {  // speculative decoding uses cpu sampler
      auto gpu_sampler = Sampler::CreateGPUSampler(max_num_sample, vocab_size_, &this->ft_, device_,
                                                   std::move(trace_recorder));
      auto cpu_sampler = Sampler::CreateCPUSampler(trace_recorder);
      gpu_sampler->sub_sampler_ = cpu_sampler;
      cpu_sampler->sub_sampler_ = gpu_sampler;
      // return gpu_sampler;
      return cpu_sampler;
    } else
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

  int GetMaxWindowSize() const final {
    // Being "-1" means there is no limit on the window size.
    return max_window_size_ != -1 ? max_window_size_ : std::numeric_limits<int>::max();
  }

  ObjectRef AllocEmbeddingTensor() final {
    // Allocate the embedding tensor.
    ObjectRef embedding = ft_.alloc_embedding_tensor_func_();
    // Get the shape of the embedding tensor for hidden size.
    ShapeTuple embedding_shape;
    if (ft_.use_disco) {
      ICHECK(embedding->IsInstance<DRefObj>());
      ObjectRef shape_ref = ft_.nd_get_shape_func_(embedding);
      embedding_shape = Downcast<DRef>(shape_ref)->DebugGetFromRemote(0);
    } else {
      NDArray embedding_nd = Downcast<NDArray>(embedding);
      embedding_shape = embedding_nd.Shape();
    }
    ICHECK_EQ(embedding_shape.size(), 2);
    ICHECK_GE(embedding_shape[0], prefill_chunk_size_);
    this->hidden_size_ = embedding_shape[1];
    return embedding;
  }

  ObjectRef AllocHiddenStatesTensor() final {
    // Allocate the hidden_states tensor.
    // Use the same function as embeddings.
    ObjectRef hidden_states = ft_.alloc_embedding_tensor_func_();
    // Get the shape of the hidden_states tensor for hidden size.
    ShapeTuple hidden_states_shape;
    if (ft_.use_disco) {
      ICHECK(hidden_states->IsInstance<DRefObj>());
      ObjectRef shape_ref = ft_.nd_get_shape_func_(hidden_states);
      hidden_states_shape = Downcast<DRef>(shape_ref)->DebugGetFromRemote(0);
    } else {
      NDArray hidden_states_nd = Downcast<NDArray>(hidden_states);
      hidden_states_shape = hidden_states_nd.Shape();
    }
    ICHECK_EQ(hidden_states_shape.size(), 2);
    ICHECK_EQ(hidden_states_shape[0], prefill_chunk_size_);
    this->hidden_size_ = hidden_states_shape[1];
    return hidden_states;
  }

  void Reset() final {
    // Reset the KV cache.
    if (kv_cache_.defined()) {
      ft_.reset_kv_cache_func_(kv_cache_);
    }
  }

  /************** Debug/Profile **************/

  void DebugCallFuncOnAllAllWorker(const String& func_name) final {
    ft_.DebugCallFuncOnAllAllWorker(func_name);
  }

 private:
  /*! \brief Load model configuration from JSON. */
  picojson::object LoadModelConfigJSON(const std::string& config_str) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
    }

    // Get json fields.
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("context_window_size")) {
      CHECK(config["context_window_size"].is<int64_t>());
      this->max_window_size_ = config["context_window_size"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"context_window_size\" not found.";
    }
    if (config.count("sliding_window_size")) {
      CHECK(config["sliding_window_size"].is<int64_t>());
      this->sliding_window_size_ = config["sliding_window_size"].get<int64_t>();
      CHECK(sliding_window_size_ == -1 || sliding_window_size_ > 0)
          << "Sliding window should be either -1 (which means disabled) of positive";
    }
    if (config.count("attention_sink_size")) {
      CHECK(config["attention_sink_size"].is<int64_t>());
      this->attention_sink_size_ = config["attention_sink_size"].get<int64_t>();
      this->attention_sink_size_ = std::max(this->attention_sink_size_, 0);
    }
    if (config.count("tensor_parallel_shards")) {
      CHECK(config["tensor_parallel_shards"].is<int64_t>());
      this->num_shards_ = config["tensor_parallel_shards"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"tensor_parallel_shards\" not found.";
    }
    if (config.count("prefill_chunk_size")) {
      CHECK(config["prefill_chunk_size"].is<int64_t>());
      this->prefill_chunk_size_ = config["prefill_chunk_size"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"prefill_chunk_size\" not found.";
    }
    if (config.count("vocab_size")) {
      CHECK(config["vocab_size"].is<int64_t>());
      this->vocab_size_ = config["vocab_size"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"vocab_size\" not found.";
    }

    return config;
  }

  //----------------------------
  // Model configurations
  //----------------------------
  int max_window_size_ = -1;
  int sliding_window_size_ = -1;
  int attention_sink_size_ = 0;
  int num_shards_ = -1;
  int max_num_sequence_ = -1;
  int prefill_chunk_size_ = -1;
  int hidden_size_ = -1;
  int vocab_size_ = -1;
  int image_embed_size_ = -1;
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
  // Shared NDArray
  memory::Storage token_ids_storage_{nullptr};
  NDArray logit_pos_arr_{nullptr};
  // A boolean indicating if tracing is enabled.
  bool trace_enabled_;
};

TVM_REGISTER_GLOBAL("mlc.copy_embedding_to_offset")
    .set_body_typed([](NDArray embedding, NDArray dst, int offset) {
      // embedding: (m, hidden_size)
      // dst: (prefill_chunk_size, hidden_size)
      ICHECK_EQ(embedding->ndim, 2);
      ICHECK_EQ(dst->ndim, 2);
      ICHECK_LE(embedding->shape[0] + offset, dst->shape[0]);
      ICHECK_EQ(embedding->shape[1], dst->shape[1]);
      const DLTensor& copy_src = *(embedding.operator->());
      const DLTensor* p_copy_dst = dst.operator->();
      DLTensor copy_dst = *p_copy_dst;
      copy_dst.shape = embedding->shape;
      copy_dst.byte_offset =
          offset * embedding->shape[1] * ((embedding->dtype.bits * embedding->dtype.lanes + 7) / 8);
      NDArray::CopyFromTo(&copy_src, &copy_dst);
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
