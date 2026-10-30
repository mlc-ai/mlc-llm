/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file cpp/support/nvtx.h
 * \brief NVTX scoped range utility (header-only).
 *
 * Provides NVTXScopedRange: a lightweight RAII wrapper over
 * nvtxRangePush/Pop.  When MLC_NVTX_ENABLED is not defined or is 0,
 * all methods are no-ops compiled away by the optimizer.
 */
#ifndef MLC_LLM_SUPPORT_NVTX_H_
#define MLC_LLM_SUPPORT_NVTX_H_

#include <string>

#ifndef MLC_NVTX_ENABLED
#define MLC_NVTX_ENABLED 0
#endif

#if MLC_NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif  // MLC_NVTX_ENABLED

namespace mlc {
namespace llm {

/*!
 * \brief A class to create a NVTX range. No-op if MLC is not built against NVTX.
 */
class NVTXScopedRange {
 public:
  /*! \brief Enter an NVTX scoped range */
#if MLC_NVTX_ENABLED
  explicit NVTXScopedRange(const char* name) { nvtxRangePush(name); }
#else
  explicit NVTXScopedRange(const char* name) {}
#endif  // MLC_NVTX_ENABLED
  /*! \brief Enter an NVTX scoped range */
  explicit NVTXScopedRange(const std::string& name) : NVTXScopedRange(name.c_str()) {}
  /*! \brief Exit an NVTX scoped range */
#if MLC_NVTX_ENABLED
  ~NVTXScopedRange() { nvtxRangePop(); }
#else
  ~NVTXScopedRange() {}
#endif  // MLC_NVTX_ENABLED
  NVTXScopedRange(const NVTXScopedRange& other) = delete;
  NVTXScopedRange(NVTXScopedRange&& other) = delete;
  NVTXScopedRange& operator=(const NVTXScopedRange& other) = delete;
  NVTXScopedRange& operator=(NVTXScopedRange&& other) = delete;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_NVTX_H_
