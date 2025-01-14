<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MLC-LLM WebAssembly Runtime

This folder contains MLC-LLM WebAssembly Runtime.

Please refer to https://llm.mlc.ai/docs/install/emcc.html.

The main step is running `make` under this folder, a step included in `web/prep_emcc_deps.sh`.

`make` creates `web/dist/wasm/mlc_wasm_runtime.bc`, which will be included in the model library wasm
when we compile the model. Thus during runtime, runtimes like WebLLM can directly reuse source
code from MLC-LLM.
