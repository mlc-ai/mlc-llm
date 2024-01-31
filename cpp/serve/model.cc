/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.cc
 * \brief The implementation of runtime module of LLM functions (prefill/decode/etc.)
 */
#include "model.h"

#include <picojson.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

namespace mlc {
namespace llm {
namespace serve {

/*********************** Utils ***********************/

/*! \brief Utility function that copies input array to the device. */
template <typename T>
NDArray CopyArrayToDevice(const std::vector<T>& array, NDArray* dst, DLDataType dtype,
                          int default_init_size, Device device) {
  ICHECK(!array.empty());
  ICHECK(dst != nullptr);
  ICHECK(!dst->defined() || (*dst)->ndim == 1);
  int64_t init_size = dst->defined() ? (*dst)->shape[0] : default_init_size;
  while (init_size < static_cast<int64_t>(array.size())) {
    init_size *= 2;
  }
  if (!dst->defined() || init_size != (*dst)->shape[0]) {
    (*dst) = NDArray::Empty({init_size}, dtype, device);
  }
  ICHECK_LE(static_cast<int64_t>(array.size()), (*dst)->shape[0]);
  NDArray view = dst->CreateView(ShapeTuple({static_cast<int64_t>(array.size())}), dtype);

  DLTensor copy_dst = *(view.operator->());
  DLTensor copy_src;
  copy_src.data = const_cast<T*>(array.data());
  copy_src.device = Device{kDLCPU, 0};
  copy_src.ndim = 1;
  copy_src.dtype = view->dtype;
  copy_src.shape = view->shape;
  copy_src.strides = nullptr;
  copy_src.byte_offset = 0;
  NDArray::CopyFromTo(&copy_src, &copy_dst);
  return view;
}

/*********************** Model Implementation ***********************/

class ModelImpl;

TVM_REGISTER_OBJECT_TYPE(ModelObj);

Model Model::Create(TVMArgValue reload_lib, String model_path, DLDevice device,
                    int max_num_sequence) {
  return Model(make_object<ModelImpl>(reload_lib, model_path, device, max_num_sequence));
}

class ModelImpl : public ModelObj {
 public:
  /*!
   * \brief Constructor of ModelImpl.
   * \sa Model::Create
   */
  explicit ModelImpl(TVMArgValue reload_lib, String model_path, DLDevice device,
                     int max_num_sequence)
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
    this->ft_.Init(reload_lib, device_, model_config);
    // Step 3. Load params in nd-array cache.
    this->params_ = ft_.LoadParams(model_path, device_);
    // Step 4. Set max_num_sequence
    this->max_num_sequence_ = max_num_sequence;
    // Step 5. Reset
    this->Reset();
  }

  /*********************** Model Computation  ***********************/

  ObjectRef TokenEmbed(IntTuple token_ids, ObjectRef dst, int offset) final {
    int num_tokens = token_ids.size();
    std::vector<int32_t> vec_token_ids(token_ids->data, token_ids->data + num_tokens);
    // Copy input token ids to device.
    DLDataType dtype(DataType::Int(32));
    NDArray token_ids_nd =
        CopyArrayToDevice(vec_token_ids, &input_token_ids_, dtype, max_window_size_, device_);
    ICHECK_EQ(token_ids_nd->ndim, 1);
    ICHECK_EQ(token_ids_nd->shape[0], num_tokens);

    CHECK(ft_.embed_func_.defined())
        << "`embed` function is not found in the model. Please make sure the model is compiled "
           "with flag `--sep-embed` and `--enable-batching`";
    auto token_ids_dref_or_nd = ft_.CopyToWorker0(token_ids_nd, "token_ids", {max_window_size_});
    if (!dst->IsInstance<DRefObj>()) {
      NDArray embeddings_nd = Downcast<NDArray>(dst);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_LE(offset + num_tokens, embeddings_nd->shape[0]);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      dst = ft_.CopyToWorker0(Downcast<NDArray>(dst), "embed_dst",
                              {prefill_chunk_size_, hidden_size_});
    }

    IntTuple offset_tuple{offset};
    return ft_.embed_func_(token_ids_dref_or_nd, dst, offset_tuple, params_);
  }

  NDArray BatchPrefill(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                       const std::vector<int>& lengths) final {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    std::vector<int> logit_pos;
    logit_pos.reserve(num_sequences);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      logit_pos.push_back(total_length - 1);
    }

    NDArray logit_pos_nd =
        CopyArrayToDevice(logit_pos, &logit_pos_arr_, DataType::Int(32), 32, device_);

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
      // embeddings: (n, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_EQ(embeddings_nd->shape[0], total_length);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.view_func_(embeddings, embedding_shape);
    }
    ObjectRef logit_pos_dref_or_nd =
        ft_.CopyToWorker0(logit_pos_nd, "logit_pos", {max_num_sequence_});
    // args: embeddings, logit_pos, kv_cache, params
    ObjectRef ret =
        ft_.prefill_func_(embeddings_dref_or_nd, logit_pos_dref_or_nd, kv_cache_, params_);
    NDArray logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      logits = Downcast<NDArray>(result[0]);
    } else {
      logits = Downcast<Array<NDArray>>(ret)[0];
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, num_sequences, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], num_sequences);
    return logits;
  }

  NDArray BatchDecode(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids) final {
    int num_sequences = seq_ids.size();
    CHECK(ft_.decode_func_.defined())
        << "`decode_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.kv_cache_begin_forward_func_.defined());
    ICHECK(ft_.kv_cache_end_forward_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    // Begin forward with the sequence ids and new lengths.
    IntTuple seq_ids_tuple(seq_ids);
    IntTuple lengths_tuple(std::vector<int64_t>(/*n=*/num_sequences, /*v=*/1));
    ft_.kv_cache_begin_forward_func_(kv_cache_, seq_ids_tuple, lengths_tuple);

    ObjectRef embeddings_dref_or_nd;
    if (!embeddings->IsInstance<DRefObj>()) {
      // embeddings: (b, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_EQ(embeddings_nd->shape[0], num_sequences);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({num_sequences, 1, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{num_sequences, 1, hidden_size_};
      embeddings_dref_or_nd = ft_.view_func_(embeddings, embedding_shape);
    }

    // args: embeddings, kv_cache, params
    ObjectRef ret = ft_.decode_func_(embeddings_dref_or_nd, kv_cache_, params_);
    NDArray logits;
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      logits = Downcast<NDArray>(result[0]);
    } else {
      logits = Downcast<Array<NDArray>>(ret)[0];
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (b, 1, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], num_sequences);
    ICHECK_EQ(logits->shape[1], 1);
    return logits;
  }

  NDArray BatchVerify(const ObjectRef& embeddings, const std::vector<int64_t>& seq_ids,
                      const std::vector<int>& lengths) final {
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
      // embeddings: (n, h)
      NDArray embeddings_nd = Downcast<NDArray>(embeddings);
      ICHECK_EQ(embeddings_nd->ndim, 2);
      ICHECK_EQ(embeddings_nd->shape[0], total_length);
      ICHECK_EQ(embeddings_nd->shape[1], hidden_size_);
      ICHECK_EQ(embeddings_nd->device.device_type, device_.device_type);
      ICHECK_EQ(embeddings_nd->device.device_id, device_.device_id);
      embeddings_dref_or_nd =
          embeddings_nd.CreateView({1, total_length, hidden_size_}, embeddings_nd->dtype);
    } else {
      ShapeTuple embedding_shape{1, total_length, hidden_size_};
      embeddings_dref_or_nd = ft_.view_func_(embeddings, embedding_shape);
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
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    ft_.kv_cache_end_forward_func_(kv_cache_);

    // logits: (1, total_length, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], total_length);
    return logits;
  }

  NDArray SoftmaxWithTemperature(NDArray logits, Array<GenerationConfig> generation_cfg) final {
    // logits: (b, n, v)
    CHECK_EQ(logits->ndim, 3);
    CHECK_EQ(logits->shape[0], generation_cfg.size());
    CHECK_EQ(logits->device.device_type, device_.device_type);
    CHECK_EQ(logits->device.device_id, device_.device_id);

    int batch_size = logits->shape[0];
    std::vector<float> temperatures;
    temperatures.reserve(batch_size);
    for (GenerationConfig cfg : generation_cfg) {
      temperatures.push_back(cfg->temperature);
    }
    NDArray temperatures_nd =
        CopyArrayToDevice(temperatures, &temperature_arr_, logits->dtype, 32, device_);
    ICHECK_EQ(temperatures_nd->ndim, 1);
    ICHECK_EQ(temperatures_nd->shape[0], batch_size);

    NDArray probs = ft_.softmax_func_(logits, temperatures_nd);
    ICHECK_EQ(probs->ndim, 3);
    ICHECK_EQ(probs->shape[0], logits->shape[0]);
    ICHECK_EQ(probs->shape[1], logits->shape[1]);
    ICHECK_EQ(probs->shape[2], logits->shape[2]);
    return probs;
  }

  /*********************** KV Cache Management  ***********************/

  void CreateKVCache(KVCacheConfig kv_cache_config) final {
    IntTuple max_num_sequence{kv_cache_config->max_num_sequence};
    IntTuple max_total_sequence_length{kv_cache_config->max_total_sequence_length};
    IntTuple prefill_chunk_size{kv_cache_config->prefill_chunk_size};
    IntTuple page_size{kv_cache_config->page_size};
    kv_cache_ = ft_.create_kv_cache_func_(max_num_sequence, max_total_sequence_length,
                                          prefill_chunk_size, page_size);
  }

  void AddNewSequence(int64_t seq_id) final { ft_.kv_cache_add_sequence_func_(kv_cache_, seq_id); }

  /*! \brief Remove the given sequence from the KV cache in the model. */
  void RemoveSequence(int64_t seq_id) final {
    ft_.kv_cache_remove_sequence_func_(kv_cache_, seq_id);
  }

  /*! \brief Get the number of available pages in KV cache. */
  int GetNumAvailablePages() const final {
    if (!ft_.use_disco) {
      return ft_.kv_cache_get_num_available_pages_func_(kv_cache_);
    } else {
      DRef ret = ft_.kv_cache_get_num_available_pages_func_(kv_cache_);
      return ret->DebugGetFromRemote(0);
    }
  }

  /*! \brief Pop out N pages from KV cache. */
  void PopNFromKVCache(int seq_id, int num_tokens) final {
    ft_.kv_cache_popn_func_(kv_cache_, seq_id, num_tokens);
  }

  /*********************** Utilities  ***********************/

  int GetMaxWindowSize() const final {
    CHECK_NE(max_window_size_, -1) << "The model has not been initialized";
    return max_window_size_;
  }

  NDArray GetEmbeddingArray(int length) final {
    if (!embeddings_.defined()) {
      // Initialize the embedding array.
      embeddings_ = NDArray::Empty({prefill_chunk_size_, hidden_size_},
                                   ft_.model_metadata_.model_dtype, device_);
    } else {
      ICHECK_EQ(embeddings_->ndim, 2);
      ICHECK_EQ(embeddings_->shape[0], prefill_chunk_size_);
      ICHECK_EQ(embeddings_->shape[1], hidden_size_);
    }
    CHECK_LE(length, prefill_chunk_size_)
        << "The required length \"" << length << "\" exceeds the supported prefill chunk size.";
    return embeddings_.CreateView({length, hidden_size_}, embeddings_->dtype);
  }

  void Reset() final {
    // Reset the KV cache.
    if (kv_cache_.defined()) {
      ft_.reset_kv_cache_func_(kv_cache_);
    }
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
    if (config.count("prefill_chunk_size")) {
      CHECK(config["prefill_chunk_size"].is<int64_t>());
      this->prefill_chunk_size_ = config["prefill_chunk_size"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"prefill_chunk_size\" not found.";
    }
    CHECK(config.count("model_config") && config["model_config"].is<picojson::object>())
        << "The mlc-chat-config.json does not contain the required `model_config` field.";
    picojson::object model_meta_config = config["model_config"].get<picojson::object>();
    CHECK(model_meta_config.count("hidden_size") && model_meta_config["hidden_size"].is<int64_t>())
        << "The `model_config` field does not contain the required `hidden_size` field.";
    this->hidden_size_ = model_meta_config["hidden_size"].get<int64_t>();
    return config;
  }

  //----------------------------
  // Model configurations
  //----------------------------
  int max_window_size_ = -1;
  int hidden_size_ = -1;
  int prefill_chunk_size_ = -1;
  int max_num_sequence_ = -1;
  //----------------------------
  // TVM related states
  //----------------------------
  // Packed function table
  FunctionTable ft_;
  // Paged KV cache
  ObjectRef kv_cache_{nullptr};
  // Runtime device
  Device device_;
  // Model parameters
  ObjectRef params_;
  // Shared NDArray
  NDArray input_token_ids_{nullptr};
  NDArray embeddings_{nullptr};
  NDArray logit_pos_arr_{nullptr};
  NDArray temperature_arr_{nullptr};
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc
