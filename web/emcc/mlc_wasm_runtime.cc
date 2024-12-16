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

/*
 * \file mlc_wasm_runtime.cc
 * \brief MLC wasm runtime library pack.
 */

// configurations for tvm logging
#define TVM_LOG_STACK_TRACE 0
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1

// Pass in COMPILE_MLC_WASM_RUNTIME so unsupported code would not be compiled in to the .bc file
#define COMPILE_MLC_WASM_RUNTIME 1
#define __STDC_FORMAT_MACROS 1
#define PICOJSON_USE_INT64

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
