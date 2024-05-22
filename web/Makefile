# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

TVM_ROOT=$(TVM_SOURCE_DIR)
MLC_LLM_ROOT=$(shell cd ..; pwd)

INCLUDE_FLAGS = -I$(TVM_ROOT) -I$(TVM_ROOT)/include\
	-I$(TVM_ROOT)/3rdparty/dlpack/include -I$(TVM_ROOT)/3rdparty/dmlc-core/include\
	-I$(TVM_ROOT)/3rdparty/compiler-rt -I$(TVM_ROOT)/3rdparty/picojson\
	-I$(MLC_LLM_ROOT)/3rdparty/tokenizers-cpp\
	-I$(MLC_LLM_ROOT)/3rdparty/tokenizers-cpp/include -I$(MLC_LLM_ROOT)/cpp

.PHONY: clean all rmtypedep preparetest

all: dist/wasm/mlc_wasm_runtime.wasm

EMCC = emcc

EMCC_CFLAGS = $(INCLUDE_FLAGS) -O3 -std=c++17 -Wno-ignored-attributes

EMCC_LDFLAGS = --no-entry -s WASM_BIGINT=1 -s ALLOW_MEMORY_GROWTH=1 -s STANDALONE_WASM=1\
 -s ERROR_ON_UNDEFINED_SYMBOLS=0

dist/wasm/mlc_wasm_runtime.bc: emcc/mlc_wasm_runtime.cc
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -c -MM -MT dist/wasm/mlc_wasm_runtime.bc emcc/mlc_wasm_runtime.cc >dist/wasm/mlc_wasm_runtime.d
	$(EMCC) $(EMCC_CFLAGS) -emit-llvm -c -o dist/wasm/mlc_wasm_runtime.bc emcc/mlc_wasm_runtime.cc

# Compile to wasm here so that errors can be caught earlier (rather than during export_library)
dist/wasm/mlc_wasm_runtime.wasm: dist/wasm/mlc_wasm_runtime.bc
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -o dist/wasm/mlc_wasm_runtime.wasm $+ $(EMCC_LDFLAGS)

clean:
	@rm -rf dist/wasm lib

-include dist/wasm/*.d
