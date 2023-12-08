/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.cc
 * \brief The implementation of runtime module of LLM functions (prefill/decode/etc.)
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "model.h"

#include <picojson.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

namespace mlc {
namespace llm {
namespace serve {

/*********************** Utils ***********************/

/*!
 * \brief Concatenate the input embeddings along the sequence dimension.
 * Store the concatenation result into the input destination NDarray.
 * Return concatenation result as an NDArray view of the destination array.
 * \param embedding_arr The array of embeddings to concatenate.
 * \param total_length The total length of the input embeddings along the sequence dim.
 * \param device The device where the embeddings locate.
 * \param initial_seq_len The initial sequence length to allocate for embeddings.
 * \param dst The destination of the concatenation
 * \return The concatenated embeddings.
 */
NDArray ConcatEmbeddings(Array<NDArray> embedding_arr, int64_t total_length, DLDevice device,
                         int initial_seq_len, NDArray* dst) {
  ICHECK(!embedding_arr.empty());
  ICHECK_NOTNULL(dst);
  int hidden_size = -1;
  DataType dtype;
  for (NDArray inp_embeddings : embedding_arr) {
    // inp_embedding: (1, n, h)
    CHECK_EQ(inp_embeddings->ndim, 3);
    CHECK_EQ(inp_embeddings->shape[0], 1);
    CHECK_EQ(inp_embeddings->device.device_type, device.device_type);
    CHECK_EQ(inp_embeddings->device.device_id, device.device_id);
    if (hidden_size == -1) {
      hidden_size = inp_embeddings->shape[2];
      dtype = inp_embeddings.DataType();
    } else {
      CHECK_EQ(inp_embeddings->shape[2], hidden_size);
      CHECK_EQ(inp_embeddings.DataType(), dtype);
    }
  }

  // - Resize the shared embedding array.
  if (dst->defined()) {
    ICHECK_EQ((*dst)->ndim, 3);
    ICHECK_EQ((*dst)->shape[0], 1);
    ICHECK_EQ((*dst)->shape[2], hidden_size);
  }
  int64_t init_size = dst->defined() ? (*dst)->shape[1] : initial_seq_len;
  while (init_size < total_length) {
    init_size *= 2;
  }
  if (!dst->defined() || init_size != (*dst)->shape[1]) {
    *dst = NDArray::Empty({1, init_size, hidden_size}, dtype, device);
  }

  // - Copy input embeddings.
  int64_t start_pos = 0;
  for (NDArray inp_embeddings : embedding_arr) {
    int64_t length = inp_embeddings->shape[1];
    CHECK_LE(start_pos + length, total_length);

    DLTensor copy_dst = *(dst->operator->());
    copy_dst.byte_offset = start_pos * hidden_size * dtype.bytes();
    copy_dst.shape = inp_embeddings->shape;
    NDArray::CopyFromTo(inp_embeddings.operator->(), &copy_dst);

    start_pos += length;
  }
  CHECK_EQ(start_pos, total_length);
  return dst->CreateView({1, total_length, hidden_size}, dtype);
}

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
  view.CopyFromBytes(array.data(), array.size() * sizeof(T));
  return view;
}

/*********************** Model Implementation ***********************/

class ModelImpl;

TVM_REGISTER_OBJECT_TYPE(ModelObj);

Model Model::Create(TVMArgValue reload_lib, String model_path, DLDevice device) {
  return Model(make_object<ModelImpl>(reload_lib, model_path, device));
}

class ModelImpl : public ModelObj {
 public:
  /*!
   * \brief Constructor of ModelImpl.
   * \sa Model::Create
   */
  explicit ModelImpl(TVMArgValue reload_lib, String model_path, DLDevice device) : device_(device) {
    // Step 1. Process model config json string.
    {
      std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
      std::ostringstream config_ostream;
      ICHECK(config_istream);
      config_ostream << config_istream.rdbuf();
      std::string config_str = config_ostream.str();
      LoadModelConfigJSON(config_str);
    }
    // Step 2. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    this->ft_.Init(reload_lib, device_, num_shards_);
    // Step 3. Load params in nd-array cache.
    this->params_ = ft_.LoadParams(model_path, device_);
    // Step 4. Reset
    this->Reset();
  }

  /*********************** Model Computation  ***********************/

  NDArray TokenEmbed(IntTuple token_ids) final {
    int num_tokens = token_ids.size();
    std::vector<int32_t> vec_token_ids(token_ids->data, token_ids->data + num_tokens);
    // Copy input token ids to device.
    DLDataType dtype(DataType::Int(32));
    NDArray token_ids_nd =
        CopyArrayToDevice(vec_token_ids, &input_token_ids_, dtype, max_window_size_, device_);
    ICHECK_EQ(token_ids_nd->ndim, 1);
    ICHECK_EQ(token_ids_nd->shape[0], num_tokens);
    token_ids_nd = token_ids_nd.CreateView({1, num_tokens}, dtype);

    CHECK(ft_.embed_func_.defined())
        << "`embed` function is not found in the model. Please make sure the model is compiled "
           "with flag `--sep-embed` and `--enable-batching`";
    auto token_ids_dref_or_nd = ft_.CopyToWorker0(token_ids_nd);

    ObjectRef embeddings = ft_.embed_func_(token_ids_dref_or_nd, params_);
    NDArray embeddings_ndarray;
    if (ft_.use_disco) {
      embeddings_ndarray = Downcast<DRef>(embeddings)->DebugGetFromRemote(0);
    } else {
      embeddings_ndarray = Downcast<NDArray>(embeddings);
    }
    // embeddings: (1, total_length, hidden_size)
    ICHECK_EQ(embeddings_ndarray->ndim, 3);
    ICHECK_EQ(embeddings_ndarray->shape[0], 1);
    ICHECK_EQ(embeddings_ndarray->shape[1], num_tokens);
    return embeddings_ndarray;
  }

  NDArray BatchPrefill(Array<NDArray> embedding_arr, std::vector<int> seq_ids,
                       std::vector<int> lengths) final {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    std::vector<int> logit_pos;
    logit_pos.reserve(num_sequences);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      logit_pos.push_back(total_length - 1);
      if (i > 0) {
        CHECK_GT(seq_ids[i], seq_ids[i - 1]) << "The input sequence ids must be non-decreasing.";
      }
    }

    // embeddings: (1, n, h)
    NDArray embeddings = ConcatEmbeddings(std::move(embedding_arr), total_length, device_,
                                          max_window_size_, &embeddings_);
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], total_length);
    ICHECK_EQ(embeddings->device.device_type, device_.device_type);
    ICHECK_EQ(embeddings->device.device_id, device_.device_id);

    NDArray logit_pos_nd =
        CopyArrayToDevice(logit_pos, &logit_pos_arr_, DataType::Int(32), 32, device_);

    CHECK(ft_.prefill_func_.defined())
        << "`prefill_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.reset_append_length_kv_cache_func_.defined());
    ICHECK(ft_.reserve_length_in_kv_cache_func_.defined());
    ICHECK(ft_.sync_device_kv_cache_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the length of the input.
    ft_.reset_append_length_kv_cache_func_(kv_cache_);
    for (int i = 0; i < num_sequences; ++i) {
      ft_.reserve_length_in_kv_cache_func_(kv_cache_, seq_ids[i], lengths[i]);
    }
    ft_.sync_device_kv_cache_func_(kv_cache_);

    ObjectRef embeddings_dref_or_nd = ft_.CopyToWorker0(embeddings);
    ObjectRef logit_pos_dref_or_nd = ft_.CopyToWorker0(logit_pos_nd);
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
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], num_sequences);
    return logits;
  }

  NDArray BatchDecode(NDArray embeddings) final {
    // embeddings: (b, 1, h)
    CHECK_EQ(embeddings->ndim, 3);
    CHECK_EQ(embeddings->shape[1], 1);
    CHECK_EQ(embeddings->device.device_type, device_.device_type);
    CHECK_EQ(embeddings->device.device_id, device_.device_id);

    CHECK(ft_.decode_func_.defined())
        << "`decode_with_embed` function is not found in the model. Please make sure the model is "
           "compiled with flag `--sep-embed` and `--enable-batching`";
    ICHECK(ft_.reset_append_length_kv_cache_func_.defined());
    ICHECK(ft_.reserve_length_in_kv_cache_func_.defined());
    ICHECK(ft_.sync_device_kv_cache_func_.defined());
    ICHECK(kv_cache_.defined()) << "KV cache has not been initialized.";

    // Reserve in KV cache for the lengths of the input.
    ft_.reset_append_length_kv_cache_func_(kv_cache_);
    for (int64_t seq_id = 0; seq_id < embeddings->shape[0]; ++seq_id) {
      ft_.reserve_length_in_kv_cache_func_(kv_cache_, seq_id, /*length=*/1);
    }
    ft_.sync_device_kv_cache_func_(kv_cache_);
    ObjectRef embeddings_dref_or_nd = ft_.CopyToWorker0(embeddings);

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
    // logits: (b, 1, v)
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);
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
    IntTuple kv_cache_config_shape({kv_cache_config->max_num_sequence,
                                    kv_cache_config->max_total_sequence_length,
                                    kv_cache_config->page_size});
    kv_cache_ = ft_.create_kv_cache_func_(kv_cache_config_shape);
  }

  /*! \brief Add a new sequence to the KV cache. Return the in-cache id of the sequence. */
  int AddNewSequence() final {
    if (!ft_.use_disco) {
      return ft_.add_sequence_to_kv_cache_func_(kv_cache_);
    } else {
      DRef in_cache_id = ft_.add_sequence_to_kv_cache_func_(kv_cache_);
      return in_cache_id->DebugGetFromRemote(0);
    }
  }

  /*! \brief Remove the given sequence from the KV cache in the model. */
  void RemoveSequence(int seq_id) final { ft_.remove_from_kv_cache_func_(kv_cache_, seq_id); }

  /*! \brief Get the number of available pages in KV cache. */
  int GetNumAvailablePages() const final {
    if (!ft_.use_disco) {
      return ft_.get_num_available_pages_kv_cache_func_(kv_cache_);
    } else {
      DRef ret = ft_.get_num_available_pages_kv_cache_func_(kv_cache_);
      return ret->DebugGetFromRemote(0);
    }
  }

  /*********************** Utilities  ***********************/

  int GetMaxWindowSize() const final {
    CHECK_NE(max_window_size_, -1) << "The model has not been initialized";
    return max_window_size_;
  }

  void Reset() final {
    // Reset the KV cache.
    if (kv_cache_.defined()) {
      ft_.reset_kv_cache_func_(kv_cache_);
    }
  }

 private:
  /*! \brief Load model configuration from JSON. */
  void LoadModelConfigJSON(const std::string& config_str) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
    }

    // Get json fields.
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("num_shards")) {
      CHECK(config["num_shards"].is<int64_t>());
      this->num_shards_ = config["num_shards"].get<int64_t>();
    } else {
      this->num_shards_ = 1;
    }
    if (config.count("model_name")) {
      CHECK(config["model_name"].is<std::string>());
      this->model_name_ = config["model_name"].get<std::string>();
    } else {
      LOG(FATAL) << "Key \"model_name\" not found.";
    }
    if (config.count("max_window_size")) {
      CHECK(config["max_window_size"].is<int64_t>());
      this->max_window_size_ = config["max_window_size"].get<int64_t>();
    } else {
      LOG(FATAL) << "Key \"max_window_size\" not found.";
    }
  }

  //----------------------------
  // Model configurations
  //----------------------------
  std::string model_name_;
  int num_shards_ = -1;
  int max_window_size_ = -1;

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
