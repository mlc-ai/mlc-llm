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
#ifndef MLC_LLM_SUPPORT_DEVICE_H_
#define MLC_LLM_SUPPORT_DEVICE_H_

#include <dlpack/dlpack.h>

namespace mlc {
namespace llm {

/*!
 * \brief Get the preferred host device from the input device.
 * - For CUDA and ROCm, CUDAHost and ROCMHost will be returned for pinned memory,
 * since pinned memory reduces copy overhead.
 * - For other devices, CPU is returned as a fallback.
 */
inline DLDevice GetPreferredHostDevice(DLDevice device) {
  if (device.device_type == DLDeviceType::kDLCUDA) {
    return DLDevice{DLDeviceType::kDLCUDAHost, 0};
  } else if (device.device_type == DLDeviceType::kDLROCM) {
    return DLDevice{DLDeviceType::kDLROCMHost, 0};
  } else {
    // Fallback to CPU.
    return DLDevice{DLDeviceType::kDLCPU, 0};
  }
}

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_DEVICE_H_
