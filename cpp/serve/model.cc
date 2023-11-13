/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.cc
 * \brief The implementation of runtime module of LLM functions (prefill/decode/etc.)
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "model.h"

#include <picojson.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include "config.h"
#include "function_table.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The runtime module for LLM functions.
 * It runs an LLM, and has an internal KV cache that maintains
 * the history KV values of all processed tokens.
 *
 * It contains the following functions:
 *
 * Model related:
 * - "token_embed": take token ids as input and return the embeddings,
 * - "single_seq_prefill": take embedding of a single sequence
 * as input, forward the embedding through LLM and return the logits,
 * - "decode": take the embeddings of the last-committed token of an
 * entire batch as input, forward through LLM and return the logits
 * for all sequences in the batch,
 * - "softmax_with_temperature": take logits and temperatures, return
 * probabilities.
 *
 * KV cache related:
 * - "create_kv_cache": create the KV cache for this module,
 * - "add_new_sequence": add (declare) a new sequence in the KV cache,
 * - "remove_sequence": remove a sequence from KV cache.
 *
 * ... and more other auxiliary functions.
 */
class ModelModule : public ModuleNode {
 public:
  explicit ModelModule(TVMArgValue reload_lib, String model_path, DLDevice device)
      : device_(device) {
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

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "token_embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        *rv = TokenEmbed(args[0]);
      });
    } else if (name == "batch_prefill") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3);
        *rv = BatchPrefill(args[0], args[1], args[2]);
      });
    } else if (name == "decode") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        *rv = Decode(args[0]);
      });
    } else if (name == "softmax_with_temperature") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 2);
        *rv = SoftmaxWithTemperature(args[0], args[1]);
      });
    } else if (name == "create_kv_cache") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1);
        KVCacheConfig kv_cache_config = args[0];
        kv_cache_ = ft_.create_kv_cache_func_(
            ShapeTuple({kv_cache_config->max_num_sequence,
                        kv_cache_config->max_total_sequence_length, kv_cache_config->page_size}));
      });
    } else if (name == "add_new_sequence") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        *rv = ft_.add_sequence_to_kv_cache_func_(kv_cache_);
      });
    } else if (name == "remove_sequence") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1);
        ft_.remove_from_kv_cache_func_(kv_cache_, args[0]);
      });
    } else if (name == "reset") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        Reset();
      });
    } else if (name == "get_num_available_pages") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        ICHECK(kv_cache_.defined());
        *rv = ft_.get_num_available_pages_kv_cache_func_(kv_cache_);
      });
    } else if (name == "get_max_window_size") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        CHECK_NE(max_window_size_, -1) << "The model has not been initialized";
        *rv = max_window_size_;
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  const char* type_key() const final { return "mlc.serve.Model"; }

 private:
  /*!
   * \brief Compute embeddings for the input token ids.
   * \param batch_token_ids The batch of token ids to compute embedding for.
   * \return The computed embeddings.
   * \note This function will **flatten** the input batch token ids,
   * and return the NDArray flattened on the batch/sequence dimension.
   * The caller side can decide whether to reshape the returned
   * NDArray into some other shape or not.
   * This brings the convenience for batched prefill and speculation
   * verification where input sequences / draft outputs might can
   * have different lengths, and we forward the flattened embeddings
   * to prefill/verification.
   */
  NDArray TokenEmbed(Array<ShapeTuple> batch_token_ids) {
    // Flatten input tokens.
    int total_length = 0;
    std::vector<int32_t> flattened_token_ids;
    for (ShapeTuple token_ids : batch_token_ids) {
      flattened_token_ids.insert(flattened_token_ids.end(), token_ids->data,
                                 token_ids->data + token_ids.size());
      total_length += token_ids.size();
    }
    // Copy input token ids to device.
    DLDataType dtype(DataType::Int(32));
    NDArray token_ids_nd =
        CopyArrayToDevice(flattened_token_ids, &input_token_ids_, dtype, max_window_size_);
    ICHECK_EQ(token_ids_nd->ndim, 1);
    ICHECK_EQ(token_ids_nd->shape[0], total_length);
    token_ids_nd = token_ids_nd.CreateView({1, total_length}, dtype);

    CHECK(ft_.embed_func_.defined())
        << "`embed` function is not found in the model. Please make sure the model is compiled "
           "with flag `--sep-embed` and `--enable-batching`";

    NDArray embeddings = ft_.embed_func_(ft_.CopyToWorker0(token_ids_nd), params_);

    // embeddings: (1, total_length, hidden_size)
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], total_length);
    return embeddings;
  }

  /*!
   * \brief Single-sequence prefill function. Embedding in, logits out.
   * \param embeddings The embedding of the input to be prefilled.
   * \param seq_id The id of the sequence in the KV cache.
   * \param lengths The length of each sequence to prefill.
   * \return The logits for the next token.
   */
  NDArray BatchPrefill(Array<NDArray> embedding_arr, ShapeTuple seq_ids, ShapeTuple lengths) {
    CHECK(!seq_ids.empty());
    CHECK_EQ(seq_ids.size(), lengths.size());
    int num_sequences = seq_ids.size();
    int total_length = 0;
    std::vector<int> logit_pos;
    logit_pos.reserve(num_sequences);
    for (int i = 0; i < num_sequences; ++i) {
      total_length += lengths[i];
      logit_pos.push_back(total_length);
      if (i > 0) {
        CHECK_GT(seq_ids[i], seq_ids[i - 1]) << "The input sequence ids must be non-decreasing.";
      }
    }

    // embeddings: (1, n, h)
    NDArray embeddings = ConcatEmbeddings(std::move(embedding_arr), total_length);
    ICHECK_EQ(embeddings->ndim, 3);
    ICHECK_EQ(embeddings->shape[0], 1);
    ICHECK_EQ(embeddings->shape[1], total_length);
    ICHECK_EQ(embeddings->device.device_type, device_.device_type);
    ICHECK_EQ(embeddings->device.device_id, device_.device_id);

    NDArray logit_pos_nd = CopyArrayToDevice(logit_pos, &logit_pos_arr_, DataType::Int(32), 32);

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

    // args: embeddings, logit_pos, kv_cache, params
    Array<ObjectRef> ret =
        ft_.prefill_func_(ft_.CopyToWorker0(embeddings), logit_pos_nd, kv_cache_, params_);

    // logits: (1, num_sequences, v)
    NDArray logits = Downcast<NDArray>(ret[0]);
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], num_sequences);
    return logits;
  }

  /*!
   * \brief Batch decode function. Embedding in, logits out.
   * \param embeddings The embedding of last generated token in the entire batch.
   * \return The logits for the next token for each sequence in the batch.
   * \note The function runs for **every** sequence in the batch.
   * That is to say, it does not accept "running a decode step for a subset
   * of the full batch".
   */
  NDArray Decode(NDArray embeddings) {
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

    // args: embeddings, kv_cache, params
    Array<ObjectRef> ret = ft_.decode_func_(ft_.CopyToWorker0(embeddings), kv_cache_, params_);

    // logits: (b, 1, v)
    NDArray logits = Downcast<NDArray>(ret[0]);
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
    ICHECK_EQ(logits->shape[1], 1);
    return logits;
  }

  /*!
   * \brief Computing probabilities from logits with softmax and temperatures.
   * \param logits The logits to compute from.
   * \param generation_cfg The generation config which contains the temperatures.
   * \return The computed probabilities distribution.
   */
  NDArray SoftmaxWithTemperature(NDArray logits, Array<GenerationConfig> generation_cfg) {
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
    NDArray temperatures_nd = CopyArrayToDevice(temperatures, &temperature_arr_, logits->dtype, 32);
    ICHECK_EQ(temperatures_nd->ndim, 1);
    ICHECK_EQ(temperatures_nd->shape[0], batch_size);

    NDArray probs = ft_.softmax_func_(logits, temperatures_nd);
    ICHECK_EQ(probs->ndim, 3);
    ICHECK_EQ(probs->shape[0], logits->shape[0]);
    ICHECK_EQ(probs->shape[1], logits->shape[1]);
    ICHECK_EQ(probs->shape[2], logits->shape[2]);
    return probs;
  }

  /*! \brief Copy input array to the device. */
  template <typename T>
  NDArray CopyArrayToDevice(const std::vector<T>& array, NDArray* dst, DLDataType dtype,
                            int default_init_size) {
    ICHECK(!array.empty());
    ICHECK(dst != nullptr);
    ICHECK(!dst->defined() || (*dst)->ndim == 1);
    int64_t init_size = dst->defined() ? (*dst)->shape[0] : default_init_size;
    while (init_size < static_cast<int64_t>(array.size())) {
      init_size *= 2;
    }
    if (!dst->defined() || init_size != (*dst)->shape[0]) {
      (*dst) = NDArray::Empty({init_size}, dtype, device_);
    }
    ICHECK_LE(static_cast<int64_t>(array.size()), (*dst)->shape[0]);
    NDArray view = dst->CreateView(ShapeTuple({static_cast<int64_t>(array.size())}), dtype);
    view.CopyFromBytes(array.data(), array.size() * sizeof(T));
    return view;
  }

  /*! \brief Concatenate the input embeddings. */
  NDArray ConcatEmbeddings(Array<NDArray> embedding_arr, int64_t total_length) {
    ICHECK(!embedding_arr.empty());
    int hidden_size = -1;
    DataType dtype;
    for (NDArray inp_embeddings : embedding_arr) {
      // inp_embedding: (1, n, h)
      CHECK_EQ(inp_embeddings->ndim, 3);
      CHECK_EQ(inp_embeddings->shape[0], 1);
      CHECK_EQ(inp_embeddings->device.device_type, device_.device_type);
      CHECK_EQ(inp_embeddings->device.device_id, device_.device_id);
      if (hidden_size == -1) {
        hidden_size = inp_embeddings->shape[2];
        dtype = inp_embeddings.DataType();
      } else {
        CHECK_EQ(inp_embeddings->shape[2], hidden_size);
        CHECK_EQ(inp_embeddings.DataType(), dtype);
      }
    }

    // - Resize the shared embedding array.
    if (embeddings_.defined()) {
      ICHECK_EQ(embeddings_->ndim, 3);
      ICHECK_EQ(embeddings_->shape[0], 1);
      ICHECK_EQ(embeddings_->shape[2], hidden_size);
    }
    int64_t init_size = embeddings_.defined() ? embeddings_->shape[1] : max_window_size_;
    while (init_size < total_length) {
      init_size *= 2;
    }
    if (!embeddings_.defined() || init_size != embeddings_->shape[1]) {
      embeddings_ = NDArray::Empty({1, init_size, hidden_size}, dtype, device_);
    }

    // - Copy input embeddings.
    int64_t start_pos = 0;
    for (NDArray inp_embeddings : embedding_arr) {
      int64_t length = inp_embeddings->shape[1];
      CHECK_LE(start_pos + length, total_length);

      DLTensor copy_dst = *(embeddings_.operator->());
      copy_dst.byte_offset = start_pos * hidden_size * dtype.bytes();
      copy_dst.shape = inp_embeddings->shape;
      NDArray::CopyFromTo(inp_embeddings.operator->(), &copy_dst);

      start_pos += length;
    }
    CHECK_EQ(start_pos, total_length);
    return embeddings_.CreateView({1, total_length, hidden_size}, dtype);
  }

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

  /*! \brief reset the runtime states. */
  void Reset() {
    // Reset the KV cache.
    if (kv_cache_.defined()) {
      ft_.reset_kv_cache_func_(kv_cache_);
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
  // runtime device
  Device device_;
  // model params
  ObjectRef params_;
  // Shared NDArray
  NDArray input_token_ids_{nullptr};
  NDArray embeddings_{nullptr};
  NDArray logit_pos_arr_{nullptr};
  NDArray temperature_arr_{nullptr};
};

tvm::runtime::Module CreateModelModule(TVMArgValue reload_lib, String model_path, DLDevice device) {
  ObjectPtr<ModelModule> n = make_object<ModelModule>(reload_lib, std::move(model_path), device);
  return Module(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
