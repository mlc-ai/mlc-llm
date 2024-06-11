/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler/gpu_sampler.cc
 * \brief The implementation for GPU sampler functions.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/packed_func.h>

#include "../../support/random.h"
#include "sampler.h"

namespace mlc {
namespace llm {
namespace serve {

inline bool FlashInferSamplingAvailable(Device device) {
  // Device must be CUDA, and FlashInfer must be enabled.
  if (device.device_type != DLDeviceType::kDLCUDA ||
      Registry::Get("flashinfer.sampling.parallel_sampling_from_prob") == nullptr) {
    return false;
  }
  // Compute version must be at least 8.0
  TVMRetValue rv;
  DeviceAPI::Get(device)->GetAttr(device, kComputeVersion, &rv);
  std::string compute_version = rv;
  std::string major_version = compute_version.substr(0, compute_version.find('.'));
  return std::stoi(major_version) >= 8;
}

inline void CopyArray(NDArray src, NDArray dst, TVMStreamHandle copy_stream) {
  DLTensor dl_dst = *(dst.operator->());
  NDArray::CopyFromTo(src.operator->(), &dl_dst, copy_stream);
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

/*********************** GPU Sampler ***********************/

class GPUSampler : public SamplerObj {
 public:
  explicit GPUSampler(int max_num_sample, int vocab_size, FunctionTable* ft, DLDevice device,
                      Optional<EventTraceRecorder> trace_recorder)
      : max_num_sample_(max_num_sample),
        vocab_size_(vocab_size),
        flashinfer_sampling_available_(FlashInferSamplingAvailable(device)),
        device_(device),
        gpu_multinomial_from_uniform_func_(ft->gpu_multinomial_from_uniform_func_),
        gpu_argsort_probs_func_(ft->gpu_argsort_probs_func_),
        gpu_sample_with_top_p_func_(ft->gpu_sample_with_top_p_func_),
        gpu_sampler_take_probs_func_(ft->gpu_sampler_take_probs_func_),
        gpu_verify_draft_tokens_func_(ft->gpu_verify_draft_tokens_func_),
        gpu_renormalize_by_top_p_func_(ft->gpu_renormalize_by_top_p_func_),
        trace_recorder_(std::move(trace_recorder)) {
    ICHECK(gpu_multinomial_from_uniform_func_.defined());
    ICHECK(gpu_argsort_probs_func_.defined());
    ICHECK(gpu_sample_with_top_p_func_.defined());
    ICHECK(gpu_sampler_take_probs_func_.defined());

    flashinfer_multinomial_sample_func_ =
        Registry::Get("flashinfer.sampling.parallel_sampling_from_prob");

    Device preferred_host_device = GetPreferredHostDevice(device);
    // We support at most 5 top prob results for each sequence.
    // Initialize auxiliary arrays on CPU.
    uniform_samples_host_ = NDArray::Empty({max_num_sample}, dtype_f32_, preferred_host_device);
    sample_indices_host_ = NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    top_p_host_ = NDArray::Empty({max_num_sample}, dtype_f32_, preferred_host_device);
    top_p_init_pivots_host_ = NDArray::Empty({max_num_sample, num_top_p_cutoff_pivots_}, dtype_f32_,
                                             preferred_host_device);
    top_prob_offsets_host_ =
        NDArray::Empty({max_num_sample * 5}, dtype_i32_, preferred_host_device);
    draft_tokens_host_ = NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    token_tree_first_child_host_ =
        NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    token_tree_next_sibling_host_ =
        NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    token_tree_parent_ptr_host_ =
        NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    sampled_token_ids_host_ = NDArray::Empty({max_num_sample}, dtype_i32_, preferred_host_device);
    sampled_probs_host_ = NDArray::Empty({max_num_sample}, dtype_f32_, preferred_host_device);
    top_prob_probs_host_ = NDArray::Empty({max_num_sample * 5}, dtype_f32_, preferred_host_device);
    top_prob_indices_host_ =
        NDArray::Empty({max_num_sample * 5}, dtype_i32_, preferred_host_device);
    // Initialize auxiliary arrays on GPU.
    uniform_samples_device_ = NDArray::Empty({max_num_sample}, dtype_f32_, device);
    sample_indices_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);
    top_p_device_ = NDArray::Empty({max_num_sample}, dtype_f32_, device);
    top_p_init_pivots_device_ =
        NDArray::Empty({max_num_sample, num_top_p_cutoff_pivots_}, dtype_f32_, device);
    top_prob_offsets_device_ = NDArray::Empty({max_num_sample * 5}, dtype_i32_, device);
    draft_tokens_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);
    token_tree_first_child_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);
    token_tree_next_sibling_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);
    token_tree_parent_ptr_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);
    sampled_token_ids_device_ = NDArray::Empty({max_num_sample}, dtype_i32_, device);

    // If the device is CUDA/ROCm, we create a standalone copy stream, in
    // purpose to hide the latency of auxiliary stream copy.
    if (device.device_type == DLDeviceType::kDLCUDA ||
        device.device_type == DLDeviceType::kDLROCM) {
      // The compute stream is the default stream.
      compute_stream_ = DeviceAPI::Get(device)->GetCurrentStream(device);
      copy_stream_ = DeviceAPI::Get(device)->CreateStream(device);
    }
  }

  ~GPUSampler() {
    // Free the copy stream if defined.
    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, copy_stream_);
    }
  }

  NDArray BatchRenormalizeProbsByTopP(NDArray probs_on_device,                 //
                                      const std::vector<int>& sample_indices,  //
                                      const Array<String>& request_ids,        //
                                      const Array<GenerationConfig>& generation_cfg) final {
    NVTXScopedRange nvtx_scope("BatchRenormalizeProbsByTopP");
    // probs_on_device: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start renormalization by top p");
    CHECK_EQ(probs_on_device->ndim, 2);
    int num_samples = sample_indices.size();
    int num_probs = probs_on_device->shape[0];
    int vocab_size = probs_on_device->shape[1];
    ICHECK_LE(num_probs, max_num_sample_);
    ICHECK_EQ(request_ids.size(), num_samples);
    ICHECK_EQ(generation_cfg.size(), num_samples);

    // - Check if there is need for applying top p.
    bool need_top_p = CheckTopP(generation_cfg, sample_indices, num_probs, num_samples, vocab_size);
    if (!need_top_p) {
      return probs_on_device;
    }

    // - Copy auxiliary array for top-p and initial pivots.
    NDArray top_p_host = top_p_host_.CreateView({num_probs}, dtype_f32_);
    NDArray top_p_device = top_p_device_.CreateView({num_probs}, dtype_f32_);
    CopyArray(/*src=*/top_p_host, /*dst=*/top_p_device, copy_stream_);

    NDArray top_p_init_pivots_host =
        top_p_init_pivots_host_.CreateView({num_probs, num_top_p_cutoff_pivots_}, dtype_f32_);
    NDArray top_p_init_pivots_device =
        top_p_init_pivots_device_.CreateView({num_probs, num_top_p_cutoff_pivots_}, dtype_f32_);
    const float* p_top_p = static_cast<const float*>(top_p_host->data);
    float* p_top_p_init_pivots = static_cast<float*>(top_p_init_pivots_host->data);
    for (int i = 0; i < num_probs; ++i) {
      if (1 - p_top_p[i] >= 0.02) {
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_] =
            std::min(1 - p_top_p[i], static_cast<float>(0.5));
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_ + 1] = 0.02;
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_ + 2] = 0.01;
      } else {
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_] = 1 - p_top_p[i];
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_ + 1] = (1 - p_top_p[i]) / 2;
        p_top_p_init_pivots[i * num_top_p_cutoff_pivots_ + 2] = (1 - p_top_p[i]) / 4;
      }
    }
    CopyArray(/*src=*/top_p_init_pivots_host, /*dst=*/top_p_init_pivots_device, copy_stream_);
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    // - Renormalize the prob with top p.
    NDArray renormed_probs_on_device =
        gpu_renormalize_by_top_p_func_(probs_on_device, top_p_device, top_p_init_pivots_device);

    RECORD_EVENT(trace_recorder_, request_ids, "finish renormalization by top p");
    return renormed_probs_on_device;
  }

  std::vector<SampleResult> BatchSampleTokensWithProbBeforeTopP(
      NDArray probs_on_device,                        //
      const std::vector<int>& sample_indices,         //
      const Array<String>& request_ids,               //
      const Array<GenerationConfig>& generation_cfg,  //
      const std::vector<RandomGenerator*>& rngs) final {
    NVTXScopedRange nvtx_scope("BatchSampleTokensWithProbBeforeTopP");
    return BatchSampleTokensImpl(std::move(probs_on_device), sample_indices, request_ids,
                                 generation_cfg, rngs, /*top_p_applied=*/false);
  }

  std::vector<SampleResult> BatchSampleTokensWithProbAfterTopP(
      NDArray probs_on_device,                        //
      const std::vector<int>& sample_indices,         //
      const Array<String>& request_ids,               //
      const Array<GenerationConfig>& generation_cfg,  //
      const std::vector<RandomGenerator*>& rngs) final {
    NVTXScopedRange nvtx_scope("BatchSampleTokensWithProbAfterTopP");
    return BatchSampleTokensImpl(std::move(probs_on_device), sample_indices, request_ids,
                                 generation_cfg, rngs, /*top_p_applied=*/true);
  }

  std::vector<std::vector<SampleResult>> BatchVerifyDraftTokensWithProbAfterTopP(
      NDArray probs_on_device, const Array<String>& request_ids,
      const std::vector<int>& cum_verify_lengths, const Array<GenerationConfig>& generation_cfg,
      const std::vector<RandomGenerator*>& rngs,
      const std::vector<std::vector<SampleResult>>& draft_output_tokens,
      const std::vector<int64_t>& token_tree_parent_ptr, NDArray draft_probs_on_device) final {
    NVTXScopedRange nvtx_scope("BatchVerifyDraftTokensWithProbAfterTopP");
    std::vector<std::vector<SampleResult>> sample_results;
    // probs_on_device: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start draft verification");
    CHECK_EQ(probs_on_device->ndim, 2);

    int num_sequence = static_cast<int>(cum_verify_lengths.size()) - 1;
    CHECK_EQ(rngs.size(), num_sequence);
    CHECK_EQ(draft_output_tokens.size(), num_sequence);
    sample_results.resize(num_sequence);

    int num_nodes = cum_verify_lengths.back();
    ICHECK(num_nodes <= max_num_sample_);
    CHECK_EQ(draft_probs_on_device->shape[0], num_nodes);
    NDArray uniform_samples_device = GenerateUniformSamples(rngs, cum_verify_lengths);
    NDArray draft_tokens_host = draft_tokens_host_.CreateView({num_nodes}, dtype_i32_);
    NDArray draft_tokens_device = draft_tokens_device_.CreateView({num_nodes}, dtype_i32_);

    // Copy draft tokens to GPU
    int* p_draft_tokens_host = static_cast<int*>(draft_tokens_host->data);
    for (int i = 0; i < num_sequence; i++) {
      const std::vector<SampleResult>& draft_output_tokens_i = draft_output_tokens[i];
      int start = cum_verify_lengths[i];
      int end = cum_verify_lengths[i + 1];
      // start/end is the range of the sequence i in probs_on_device, which includes the prob dist
      // of the draft tokens and the last committed token
      ICHECK_EQ(draft_output_tokens_i.size() + 1, end - start);
      for (int j = 0; j < end - start - 1; j++) {
        // Copy sampled token id
        p_draft_tokens_host[start + j + 1] = draft_output_tokens_i[j].GetTokenId();
      }
    }
    CopyArray(draft_tokens_host, draft_tokens_device, copy_stream_);

    NDArray token_tree_first_child_host =
        token_tree_first_child_host_.CreateView({num_nodes}, dtype_i32_);
    NDArray token_tree_first_child_device =
        token_tree_first_child_device_.CreateView({num_nodes}, dtype_i32_);
    NDArray token_tree_next_sibling_host =
        token_tree_next_sibling_host_.CreateView({num_nodes}, dtype_i32_);
    NDArray token_tree_next_sibling_device =
        token_tree_next_sibling_device_.CreateView({num_nodes}, dtype_i32_);
    NDArray token_tree_parent_ptr_host =
        token_tree_parent_ptr_host_.CreateView({num_sequence}, dtype_i32_);
    NDArray token_tree_parent_ptr_device =
        token_tree_parent_ptr_device_.CreateView({num_sequence}, dtype_i32_);
    std::vector<int> token_tree_child_to_parent(/*n=*/num_nodes);

    int* token_tree_first_child_ptr_host = static_cast<int*>(token_tree_first_child_host->data);
    int* token_tree_next_sibling_ptr_host = static_cast<int*>(token_tree_next_sibling_host->data);
    // Build the tree structure on CPU
    for (int i = 0; i < num_sequence; i++) {
      // Assuming no tree structure for now
      int start = cum_verify_lengths[i];
      int end = cum_verify_lengths[i + 1];
      ICHECK_GE(end - start, 2);
      for (int j = 0; j < end - start; j++) {
        int cur_node = j + start;
        int parent_node =
            token_tree_parent_ptr[cur_node] != -1 ? token_tree_parent_ptr[cur_node] + start : -1;
        token_tree_first_child_ptr_host[cur_node] = -1;
        if (parent_node != -1 && token_tree_first_child_ptr_host[parent_node] == -1) {
          token_tree_first_child_ptr_host[parent_node] = cur_node;
        }
        token_tree_child_to_parent[cur_node] = parent_node;
        if (cur_node + 1 < end && token_tree_parent_ptr[cur_node - start + 1] ==
                                      token_tree_parent_ptr[cur_node - start]) {
          token_tree_next_sibling_ptr_host[cur_node] = cur_node + 1;
        } else {
          token_tree_next_sibling_ptr_host[cur_node] = -1;
        }
      }
      static_cast<int*>(token_tree_parent_ptr_host->data)[i] = start;  // point to the root
    }
    // Copy token tree structure to GPU
    CopyArray(token_tree_first_child_host, token_tree_first_child_device, copy_stream_);
    CopyArray(token_tree_next_sibling_host, token_tree_next_sibling_device, copy_stream_);
    CopyArray(token_tree_parent_ptr_host, token_tree_parent_ptr_device, copy_stream_);

    SyncCopyStream(device_, compute_stream_, copy_stream_);

    gpu_verify_draft_tokens_func_(draft_probs_on_device, draft_tokens_device, probs_on_device,
                                  token_tree_first_child_device, token_tree_next_sibling_device,
                                  uniform_samples_device, token_tree_parent_ptr_device);

    DeviceAPI::Get(device_)->SyncStreamFromTo(device_, compute_stream_, copy_stream_);
    CopyArray(token_tree_parent_ptr_device, token_tree_parent_ptr_host, copy_stream_);

    std::vector<SampleResult> additional_sample_result;
    {
      additional_sample_result.reserve(num_sequence);
      // Sample one additional token for each sequence using the probablity at the last accepted
      // token.
      uniform_samples_device = GenerateUniformSamples(rngs, num_sequence);
      const NDArray& sample_indices_device = token_tree_parent_ptr_device;
      // Check need_prob_values
      bool need_prob_values = false;
      for (int i = 0; i < num_sequence; i++) {
        need_prob_values |= generation_cfg[i]->logprobs;
      }
      std::vector<int> top_prob_offset_indptr;
      if (!need_prob_values) {
        top_prob_offset_indptr.resize(num_sequence + 1, 0);
      } else {
        // Slow path: if any of the generation config requires prob values, we need to copy
        // sample_indices to host to compute top_prob_offset_indptr.
        TVMSynchronize(device_.device_type, device_.device_id, copy_stream_);
        std::vector<int> sample_indices;
        sample_indices.reserve(num_sequence);
        const int* p_token_tree_parent_ptr = static_cast<int*>(token_tree_parent_ptr_host->data);
        for (int i = 0; i < num_sequence; i++) {
          sample_indices.push_back(p_token_tree_parent_ptr[i]);
        }
        CheckProbValues(generation_cfg, sample_indices, num_nodes, num_sequence, vocab_size_,
                        &top_prob_offset_indptr);
      }
      auto device_arrays =
          SampleOnGPU(probs_on_device, uniform_samples_device, sample_indices_device,
                      /*need_top_p=*/false, need_prob_values, num_nodes, top_prob_offset_indptr);
      auto host_arrays = CopyArraysToCPU(device_arrays, num_sequence, need_prob_values,
                                         top_prob_offset_indptr.back());
      additional_sample_result =
          CollectSampleResult(host_arrays, num_sequence, need_prob_values, top_prob_offset_indptr);
    }

    for (int i = 0; i < num_sequence; i++) {
      int start = cum_verify_lengths[i];
      int end = cum_verify_lengths[i + 1];
      int last_accepted = static_cast<int*>(token_tree_parent_ptr_host->data)[i];
      int num_accepted = 0;
      for (int cur_node = last_accepted; cur_node != start;
           cur_node = token_tree_child_to_parent[cur_node]) {
        sample_results[i].push_back(draft_output_tokens[i][cur_node - start - 1]);
        num_accepted++;
      }
      std::reverse(sample_results[i].rbegin(), sample_results[i].rbegin() + num_accepted);
    }

    // Append the additional sample result to the sample_results
    ICHECK_EQ(additional_sample_result.size(), num_sequence);
    for (int i = 0; i < num_sequence; i++) {
      sample_results[i].push_back(additional_sample_result[i]);
    }

    RECORD_EVENT(trace_recorder_, request_ids, "finish draft verification");
    return sample_results;
  }

 private:
  std::vector<SampleResult> BatchSampleTokensImpl(NDArray probs_on_device,                        //
                                                  const std::vector<int>& sample_indices,         //
                                                  const Array<String>& request_ids,               //
                                                  const Array<GenerationConfig>& generation_cfg,  //
                                                  const std::vector<RandomGenerator*>& rngs,      //
                                                  bool top_p_applied) {
    // probs_on_device: (n, v)
    RECORD_EVENT(trace_recorder_, request_ids, "start sampling");
    CHECK_EQ(probs_on_device->ndim, 2);
    CHECK_EQ(probs_on_device->device.device_id, device_.device_id);
    CHECK_EQ(probs_on_device->device.device_type, device_.device_type);
    int num_samples = sample_indices.size();
    int num_probs = probs_on_device->shape[0];
    int vocab_size = probs_on_device->shape[1];
    if (num_samples == 0) {
      // This synchronization is necessary for making sure that this round
      // of model forward is finished.
      TVMSynchronize(device_.device_type, device_.device_id, compute_stream_);
      return {};
    }
    ICHECK_EQ(request_ids.size(), num_samples);
    ICHECK_EQ(generation_cfg.size(), num_samples);
    ICHECK_EQ(rngs.size(), num_samples);

    // Since `num_samples` may be larger than `max_num_sample_` in some cases,
    // we apply chunking to support large `num_samples`.
    std::vector<SampleResult> sample_results;
    if (num_samples <= max_num_sample_) {
      sample_results = ChunkSampleTokensImpl(probs_on_device, sample_indices, generation_cfg, rngs,
                                             top_p_applied);
    } else {
      for (int chunk_start = 0; chunk_start < num_samples; chunk_start += max_num_sample_) {
        int chunk_end = std::min(chunk_start + max_num_sample_, num_samples);
        std::vector<int> sample_indices_chunk(sample_indices.begin() + chunk_start,
                                              sample_indices.begin() + chunk_end);
        Array<GenerationConfig> generation_cfg_chunk(generation_cfg.begin() + chunk_start,
                                                     generation_cfg.begin() + chunk_end);
        std::vector<RandomGenerator*> rngs_chunk(rngs.begin() + chunk_start,
                                                 rngs.begin() + chunk_end);
        std::vector<SampleResult> sample_results_chunk = ChunkSampleTokensImpl(
            probs_on_device, sample_indices_chunk, generation_cfg_chunk, rngs_chunk, top_p_applied);
        sample_results.insert(sample_results.end(), sample_results_chunk.begin(),
                              sample_results_chunk.end());
      }
    }

    RECORD_EVENT(trace_recorder_, request_ids, "finish sampling");
    return sample_results;
  }

  /*! \brief Collect the sampling results from the computed NDArray results. */
  std::vector<SampleResult> CollectSampleResult(const std::vector<NDArray>& host_arrays,
                                                int num_samples, bool need_prob_values,
                                                const std::vector<int> top_prob_offset_indptr) {
    const int* p_sampled_token_ids = static_cast<const int*>(host_arrays[0]->data);
    const float* p_sampled_probs = nullptr;
    const float* p_top_prob_probs = nullptr;
    const int* p_top_prob_indices = nullptr;
    if (need_prob_values) {
      p_sampled_probs = static_cast<const float*>(host_arrays[1]->data);
      p_top_prob_probs = static_cast<const float*>(host_arrays[2]->data);
      p_top_prob_indices = static_cast<const int*>(host_arrays[3]->data);
    }
    std::vector<SampleResult> sample_results;
    sample_results.reserve(num_samples);
    ICHECK_EQ(top_prob_offset_indptr.size(), num_samples + 1);
    for (int i = 0; i < num_samples; ++i) {
      // Note: we set the probability in SampleResult to 1.0 since prob value is not needed.
      float sampled_prob = need_prob_values ? p_sampled_probs[i] : 1.0;
      std::vector<TokenProbPair> top_prob_tokens;
      top_prob_tokens.reserve(top_prob_offset_indptr[i + 1] - top_prob_offset_indptr[i]);
      for (int j = top_prob_offset_indptr[i]; j < top_prob_offset_indptr[i + 1]; ++j) {
        top_prob_tokens.emplace_back(p_top_prob_indices[j], p_top_prob_probs[j]);
      }
      sample_results.push_back(
          SampleResult{{p_sampled_token_ids[i], sampled_prob}, top_prob_tokens});
    }
    return sample_results;
  }

  std::vector<SampleResult> ChunkSampleTokensImpl(NDArray probs_on_device,                        //
                                                  const std::vector<int>& sample_indices,         //
                                                  const Array<GenerationConfig>& generation_cfg,  //
                                                  const std::vector<RandomGenerator*>& rngs,      //
                                                  bool top_p_applied) {
    // probs_on_device: (n, v)
    int num_samples = sample_indices.size();
    int num_probs = probs_on_device->shape[0];
    int vocab_size = probs_on_device->shape[1];

    // - Generate random numbers.
    //   Copy the random numbers and sample indices.
    auto uniform_samples_device = GenerateUniformSamples(rngs, num_samples);
    auto sample_indices_device = CopySampleIndicesToGPU(sample_indices);

    // - Check if there is need for applying top p or prob values,
    //   so that argsort is needed.
    bool need_top_p = false;
    if (!top_p_applied) {
      need_top_p = CheckTopP(generation_cfg, sample_indices, num_probs, num_samples, vocab_size);
    }
    // The indptr array of the number of top probs for each sample.
    std::vector<int> top_prob_offset_indptr;
    bool need_prob_values = CheckProbValues(generation_cfg, sample_indices, num_probs, num_samples,
                                            vocab_size, &top_prob_offset_indptr);

    // - Sample tokens on GPU, and take out the probability values if needed.
    std::vector<NDArray> device_arrays =
        SampleOnGPU(probs_on_device, uniform_samples_device, sample_indices_device, need_top_p,
                    need_prob_values, num_probs, top_prob_offset_indptr);

    // - Copy the GPU sampling function results to CPU.
    std::vector<NDArray> host_arrays = CopyArraysToCPU(device_arrays, num_samples, need_prob_values,
                                                       top_prob_offset_indptr.back());

    // - Collect the sampling results.
    return CollectSampleResult(host_arrays, num_samples, need_prob_values, top_prob_offset_indptr);
  }

  /*! \brief Generate num_samples uniform random numbers, and copy them to GPU. */
  NDArray GenerateUniformSamples(const std::vector<RandomGenerator*>& rngs, int num_samples) {
    float* p_uniform_samples = static_cast<float*>(uniform_samples_host_->data);
    for (int i = 0; i < num_samples; ++i) {
      p_uniform_samples[i] = rngs[i]->GetRandomNumber();
    }
    NDArray uniform_samples_host = uniform_samples_host_.CreateView({num_samples}, dtype_f32_);
    NDArray uniform_samples_device = uniform_samples_device_.CreateView({num_samples}, dtype_f32_);
    CopyArray(/*src=*/uniform_samples_host, /*dst=*/uniform_samples_device, copy_stream_);
    return uniform_samples_device;
  }

  /*! \brief Generate uniform random numbers, and copy the numbers and sample indices to GPU. The
   * number of samples for each random generator is given by `cum_num_samples`. */
  NDArray GenerateUniformSamples(const std::vector<RandomGenerator*>& rngs,
                                 const std::vector<int>& cum_num_samples) {
    float* p_uniform_samples = static_cast<float*>(uniform_samples_host_->data);
    int total_samples = cum_num_samples.back();
    for (int i = 0; i + 1 < static_cast<int>(cum_num_samples.size()); ++i) {
      for (int j = cum_num_samples[i]; j < cum_num_samples[i + 1]; ++j) {
        p_uniform_samples[j] = rngs[i]->GetRandomNumber();
      }
    }
    NDArray uniform_samples_host = uniform_samples_host_.CreateView({total_samples}, dtype_f32_);
    NDArray uniform_samples_device =
        uniform_samples_device_.CreateView({total_samples}, dtype_f32_);
    CopyArray(/*src=*/uniform_samples_host, /*dst=*/uniform_samples_device, copy_stream_);
    return uniform_samples_device;
  }

  /*! \brief Generate uniform random numbers, and copy the numbers and sample indices to GPU. */
  NDArray CopySampleIndicesToGPU(const std::vector<int>& sample_indices) {
    int* p_sample_indices = static_cast<int*>(sample_indices_host_->data);
    std::copy(sample_indices.begin(), sample_indices.end(), p_sample_indices);
    // Copy the sample indices to GPU.
    int num_samples = static_cast<int>(sample_indices.size());
    NDArray sample_indices_host = sample_indices_host_.CreateView({num_samples}, dtype_i32_);
    NDArray sample_indices_device = sample_indices_device_.CreateView({num_samples}, dtype_i32_);
    CopyArray(/*src=*/sample_indices_host, /*dst=*/sample_indices_device, copy_stream_);
    return sample_indices_device;
  }

  /*! \brief Check if top p is needed. Update host top p array in place. */
  bool CheckTopP(const Array<GenerationConfig>& generation_cfg,
                 const std::vector<int>& sample_indices, int num_probs, int num_samples,
                 int vocab_size) {
    // Initialize top p values with -1.
    float* p_top_p = static_cast<float*>(top_p_host_->data);
    for (int i = 0; i < num_probs; ++i) {
      p_top_p[i] = -1.0;
    }
    bool need_top_p = false;
    for (int i = 0; i < num_samples; ++i) {
      if (p_top_p[sample_indices[i]] == -1.0) {
        p_top_p[sample_indices[i]] = generation_cfg[i]->top_p;
        need_top_p |= generation_cfg[i]->top_p != 1.0;
      } else {
        CHECK(fabs(p_top_p[sample_indices[i]] - generation_cfg[i]->top_p) < eps_)
            << "GPU sampler requires the top_p values for each prob distribution are the same.";
      }
    }
    for (int i = 0; i < num_probs; ++i) {
      p_top_p[i] = std::max(p_top_p[i], eps_);
    }
    return need_top_p;
  }

  /*! \brief Check whether prob values are needed, and collect info when necessary. */
  bool CheckProbValues(const Array<GenerationConfig>& generation_cfg,
                       const std::vector<int>& sample_indices, int num_probs, int num_samples,
                       int vocab_size, std::vector<int>* top_prob_offset_indptr) {
    top_prob_offset_indptr->reserve(num_samples + 1);
    top_prob_offset_indptr->push_back(0);
    int* p_top_prob_offsets = static_cast<int*>(top_prob_offsets_host_->data);
    int num_top_probs = 0;
    bool need_prob_values = false;
    for (int i = 0; i < num_samples; ++i) {
      need_prob_values |= generation_cfg[i]->logprobs;
      for (int j = 0; j < generation_cfg[i]->top_logprobs; ++j) {
        p_top_prob_offsets[num_top_probs++] = sample_indices[i] * vocab_size + j;
      }
      top_prob_offset_indptr->push_back(top_prob_offset_indptr->back() +
                                        generation_cfg[i]->top_logprobs);
    }
    ICHECK_EQ(num_top_probs, top_prob_offset_indptr->back());
    return need_prob_values;
  }

  /*! \brief Sample tokens on GPU. Take out the probability values when needed. */
  std::vector<NDArray> SampleOnGPU(NDArray probs_on_device, NDArray uniform_samples_device,
                                   NDArray sample_indices_device,  //
                                   bool need_top_p, bool need_prob_values, int num_probs,
                                   const std::vector<int>& top_prob_offset_indptr) {
    NDArray sampled_token_ids_device{nullptr};
    NDArray sampled_probs_device{nullptr};
    NDArray top_prob_probs_device{nullptr};
    NDArray top_prob_indices_device{nullptr};

    if (!need_top_p && !need_prob_values) {
      // - Short path: If top_p and prob values are not needed, we directly sample from multinomial.
      SyncCopyStream(device_, compute_stream_, copy_stream_);
      if (flashinfer_sampling_available_) {
        sampled_token_ids_device =
            sampled_token_ids_device_.CreateView({sample_indices_device->shape[0]}, dtype_i32_);
        (*flashinfer_multinomial_sample_func_)(probs_on_device, uniform_samples_device,
                                               sample_indices_device, sampled_token_ids_device);
      } else {
        sampled_token_ids_device = gpu_multinomial_from_uniform_func_(
            probs_on_device, uniform_samples_device, sample_indices_device);
      }
      return {sampled_token_ids_device, sampled_probs_device, top_prob_probs_device,
              top_prob_indices_device};
    }

    // - Argsort the probability.
    Array<NDArray> argsort_results = gpu_argsort_probs_func_(probs_on_device);
    ICHECK_EQ(argsort_results.size(), 2);
    NDArray sorted_probs_on_device = argsort_results[0];
    NDArray sorted_indices_on_device = argsort_results[1];

    // - Copy auxiliary array for top-p and prob values in ahead.
    NDArray top_p_device;
    NDArray top_prob_offsets_device;
    if (need_top_p) {
      NDArray top_p_host = top_p_host_.CreateView({num_probs}, dtype_f32_);
      top_p_device = top_p_device_.CreateView({num_probs}, dtype_f32_);
      CopyArray(/*src=*/top_p_host, /*dst=*/top_p_device, copy_stream_);
    }
    if (need_prob_values) {
      int num_top_probs = top_prob_offset_indptr.back();
      NDArray top_prob_offsets_host =
          top_prob_offsets_host_.CreateView({num_top_probs}, dtype_i32_);
      top_prob_offsets_device = top_prob_offsets_device_.CreateView({num_top_probs}, dtype_i32_);
      CopyArray(/*src=*/top_prob_offsets_host, /*dst=*/top_prob_offsets_device, copy_stream_);
    }
    SyncCopyStream(device_, compute_stream_, copy_stream_);

    if (need_top_p) {
      // - Sample with top_p applied.
      sampled_token_ids_device =
          gpu_sample_with_top_p_func_(sorted_probs_on_device, sorted_indices_on_device,
                                      uniform_samples_device, sample_indices_device, top_p_device);
    } else {
      // - Sample without top_p.
      if (flashinfer_sampling_available_) {
        sampled_token_ids_device =
            sampled_token_ids_device_.CreateView({sample_indices_device->shape[0]}, dtype_i32_);
        (*flashinfer_multinomial_sample_func_)(probs_on_device, uniform_samples_device,
                                               sample_indices_device, sampled_token_ids_device);
      } else {
        sampled_token_ids_device = gpu_multinomial_from_uniform_func_(
            probs_on_device, uniform_samples_device, sample_indices_device);
      }
    }

    if (need_prob_values) {
      // - Take the probability values.
      Array<NDArray> prob_value_results = gpu_sampler_take_probs_func_(
          probs_on_device, sorted_indices_on_device, sample_indices_device,
          sampled_token_ids_device, top_prob_offsets_device);
      sampled_probs_device = prob_value_results[0];
      top_prob_probs_device = prob_value_results[1];
      top_prob_indices_device = prob_value_results[2];
    }

    return {sampled_token_ids_device, sampled_probs_device, top_prob_probs_device,
            top_prob_indices_device};
  }

  /*! \brief Copy the results of GPU sampling functions back to CPU. */
  std::vector<NDArray> CopyArraysToCPU(const std::vector<NDArray>& device_arrays,  //
                                       int num_samples, bool need_prob_values, int num_top_probs) {
    NDArray sampled_token_ids_device = device_arrays[0];
    NDArray sampled_probs_device = device_arrays[1];
    NDArray top_prob_probs_device = device_arrays[2];
    NDArray top_prob_indices_device = device_arrays[3];
    ICHECK(sampled_token_ids_device.defined());
    ICHECK_EQ(sampled_token_ids_device->ndim, 1);
    ICHECK_EQ(sampled_token_ids_device->shape[0], num_samples);
    NDArray sampled_token_ids_host = sampled_token_ids_host_.CreateView({num_samples}, dtype_i32_);
    CopyArray(/*src=*/sampled_token_ids_device, /*dst=*/sampled_token_ids_host, compute_stream_);

    NDArray sampled_probs_host{nullptr};
    NDArray top_prob_probs_host{nullptr};
    NDArray top_prob_indices_host{nullptr};
    if (need_prob_values) {
      ICHECK(sampled_probs_device.defined());
      ICHECK(top_prob_probs_device.defined());
      ICHECK(top_prob_indices_device.defined());
      ICHECK_EQ(sampled_probs_device->ndim, 1);
      ICHECK_EQ(top_prob_probs_device->ndim, 1);
      ICHECK_EQ(top_prob_indices_device->ndim, 1);
      ICHECK_EQ(sampled_probs_device->shape[0], num_samples);
      ICHECK_EQ(top_prob_probs_device->shape[0], num_top_probs);
      ICHECK_EQ(top_prob_indices_device->shape[0], num_top_probs);
      sampled_probs_host = sampled_probs_host_.CreateView({num_samples}, dtype_i32_);
      top_prob_probs_host = top_prob_probs_host_.CreateView({num_top_probs}, dtype_f32_);
      top_prob_indices_host = top_prob_indices_host_.CreateView({num_top_probs}, dtype_i32_);
      CopyArray(/*src=*/sampled_probs_device, /*dst=*/sampled_probs_host, compute_stream_);
      if (num_top_probs > 0) {
        CopyArray(/*src=*/top_prob_probs_device, /*dst=*/top_prob_probs_host, compute_stream_);
        CopyArray(/*src=*/top_prob_indices_device, /*dst=*/top_prob_indices_host, compute_stream_);
      }
    }

    // Synchronize for CPU to get the correct array results.
    TVMSynchronize(device_.device_type, device_.device_id, compute_stream_);

    return {sampled_token_ids_host, sampled_probs_host, top_prob_probs_host, top_prob_indices_host};
  }

  // Model configurations
  const int max_num_sample_;
  const int vocab_size_;
  const DLDataType dtype_i32_ = DataType::Int(32);
  const DLDataType dtype_f32_ = DataType::Float(32);
  const bool flashinfer_sampling_available_;
  // Functions for sampling on GPU.
  Device device_;
  PackedFunc gpu_multinomial_from_uniform_func_;
  PackedFunc gpu_argsort_probs_func_;
  PackedFunc gpu_sample_with_top_p_func_;
  PackedFunc gpu_sampler_take_probs_func_;
  PackedFunc gpu_verify_draft_tokens_func_;
  PackedFunc gpu_renormalize_by_top_p_func_;
  const PackedFunc* flashinfer_multinomial_sample_func_;
  // Auxiliary NDArrays on CPU
  NDArray uniform_samples_host_;
  NDArray sample_indices_host_;
  NDArray top_p_host_;
  NDArray top_p_init_pivots_host_;
  NDArray top_prob_offsets_host_;
  NDArray draft_tokens_host_;
  NDArray token_tree_first_child_host_;
  NDArray token_tree_next_sibling_host_;
  NDArray token_tree_parent_ptr_host_;
  NDArray sampled_token_ids_host_;
  NDArray sampled_probs_host_;
  NDArray top_prob_probs_host_;
  NDArray top_prob_indices_host_;
  // Auxiliary NDArrays on GPU
  NDArray uniform_samples_device_;
  NDArray sample_indices_device_;
  NDArray top_p_device_;
  NDArray top_p_init_pivots_device_;
  NDArray top_prob_offsets_device_;
  NDArray draft_tokens_device_;
  NDArray token_tree_first_child_device_;
  NDArray token_tree_next_sibling_device_;
  NDArray token_tree_parent_ptr_device_;
  NDArray sampled_token_ids_device_;
  // The event trace recorder for requests. */
  Optional<EventTraceRecorder> trace_recorder_;
  // The device stream for the default computation operations.
  TVMStreamHandle compute_stream_ = nullptr;
  // The device stream for copying auxiliary data structure to GPU.
  TVMStreamHandle copy_stream_ = nullptr;
  const float eps_ = 1e-5;
  const int num_top_p_cutoff_pivots_ = 3;
};

Sampler Sampler::CreateGPUSampler(int max_num_sample, int vocab_size, FunctionTable* ft,
                                  DLDevice device, Optional<EventTraceRecorder> trace_recorder) {
  return Sampler(
      make_object<GPUSampler>(max_num_sample, vocab_size, ft, device, std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
