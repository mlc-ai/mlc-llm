(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('perf_hooks'), require('ws')) :
	typeof define === 'function' && define.amd ? define(['exports', 'perf_hooks', 'ws'], factory) :
	(global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.tvmjs = {}, global.perf_hooks, global.ws));
})(this, (function (exports, perf_hooks, ws) { 'use strict';

	function _interopDefaultLegacy (e) { return e && typeof e === 'object' && 'default' in e ? e : { 'default': e }; }

	var perf_hooks__default = /*#__PURE__*/_interopDefaultLegacy(perf_hooks);
	var ws__default = /*#__PURE__*/_interopDefaultLegacy(ws);

	var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

	function unwrapExports (x) {
		return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, 'default') ? x['default'] : x;
	}

	function createCommonjsModule(fn, module) {
		return module = { exports: {} }, fn(module, module.exports), module.exports;
	}

	var support = createCommonjsModule(function (module, exports) {
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
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.wasmPath = exports.assert = exports.Uint8ArrayToString = exports.StringToUint8Array = void 0;
	/**
	 * Convert string to Uint8array.
	 * @param str The string.
	 * @returns The corresponding Uint8Array.
	 */
	function StringToUint8Array(str) {
	    const arr = new Uint8Array(str.length + 1);
	    for (let i = 0; i < str.length; ++i) {
	        arr[i] = str.charCodeAt(i);
	    }
	    arr[str.length] = 0;
	    return arr;
	}
	exports.StringToUint8Array = StringToUint8Array;
	/**
	 * Convert Uint8array to string.
	 * @param array The array.
	 * @returns The corresponding string.
	 */
	function Uint8ArrayToString(arr) {
	    const ret = [];
	    for (const ch of arr) {
	        ret.push(String.fromCharCode(ch));
	    }
	    return ret.join("");
	}
	exports.Uint8ArrayToString = Uint8ArrayToString;
	/**
	 * Internal assert helper
	 * @param condition condition The condition to fail.
	 * @param msg msg The message.
	 */
	function assert(condition, msg) {
	    if (!condition) {
	        throw new Error("AssertError:" + (msg || ""));
	    }
	}
	exports.assert = assert;
	/**
	 * Get the path to the wasm library in nodejs.
	 * @return The wasm path.
	 */
	function wasmPath() {
	    return __dirname + "/wasm";
	}
	exports.wasmPath = wasmPath;

	});

	unwrapExports(support);
	support.wasmPath;
	support.assert;
	support.Uint8ArrayToString;
	support.StringToUint8Array;

	var memory = createCommonjsModule(function (module, exports) {
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.CachedCallStack = exports.Memory = void 0;

	/**
	 * Wasm Memory wrapper to perform JS side raw memory access.
	 */
	class Memory {
	    constructor(memory) {
	        this.wasm32 = true;
	        this.memory = memory;
	        this.buffer = this.memory.buffer;
	        this.viewU8 = new Uint8Array(this.buffer);
	        this.viewU16 = new Uint16Array(this.buffer);
	        this.viewI32 = new Int32Array(this.buffer);
	        this.viewU32 = new Uint32Array(this.buffer);
	        this.viewF32 = new Float32Array(this.buffer);
	        this.viewF64 = new Float64Array(this.buffer);
	    }
	    loadU8(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewU8[ptr >> 0];
	    }
	    loadU16(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewU16[ptr >> 1];
	    }
	    loadU32(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewU32[ptr >> 2];
	    }
	    loadI32(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewI32[ptr >> 2];
	    }
	    loadI64(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        const base = ptr >> 2;
	        // assumes little endian, for now truncate high.
	        return this.viewI32[base];
	    }
	    loadF32(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewF32[ptr >> 2];
	    }
	    loadF64(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        return this.viewF64[ptr >> 3];
	    }
	    loadPointer(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        if (this.wasm32) {
	            return this.loadU32(ptr);
	        }
	        else {
	            return this.loadI64(ptr);
	        }
	    }
	    loadUSize(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        if (this.wasm32) {
	            return this.loadU32(ptr);
	        }
	        else {
	            return this.loadI64(ptr);
	        }
	    }
	    sizeofPtr() {
	        return this.wasm32 ? 4 /* I32 */ : 8 /* I64 */;
	    }
	    /**
	     * Load raw bytes from ptr.
	     * @param ptr The head address
	     * @param numBytes The number
	     */
	    loadRawBytes(ptr, numBytes) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        const result = new Uint8Array(numBytes);
	        result.set(this.viewU8.slice(ptr, ptr + numBytes));
	        return result;
	    }
	    /**
	     * Load TVMByteArray from ptr.
	     *
	     * @param ptr The address of the header.
	     */
	    loadTVMBytes(ptr) {
	        const data = this.loadPointer(ptr);
	        const length = this.loadUSize(ptr + this.sizeofPtr());
	        return this.loadRawBytes(data, length);
	    }
	    /**
	     * Load null-terminated C-string from ptr.
	     * @param ptr The head address
	     */
	    loadCString(ptr) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        // NOTE: the views are still valid for read.
	        const ret = [];
	        let ch = 1;
	        while (ch != 0) {
	            ch = this.viewU8[ptr];
	            if (ch != 0) {
	                ret.push(String.fromCharCode(ch));
	            }
	            ++ptr;
	        }
	        return ret.join("");
	    }
	    /**
	     * Store raw bytes to the ptr.
	     * @param ptr The head address.
	     * @param bytes The bytes content.
	     */
	    storeRawBytes(ptr, bytes) {
	        if (this.buffer != this.memory.buffer) {
	            this.updateViews();
	        }
	        this.viewU8.set(bytes, ptr);
	    }
	    /**
	     * Update memory view after the memory growth.
	     */
	    updateViews() {
	        this.buffer = this.memory.buffer;
	        this.viewU8 = new Uint8Array(this.buffer);
	        this.viewU16 = new Uint16Array(this.buffer);
	        this.viewI32 = new Int32Array(this.buffer);
	        this.viewU32 = new Uint32Array(this.buffer);
	        this.viewF32 = new Float32Array(this.buffer);
	        this.viewF64 = new Float64Array(this.buffer);
	    }
	}
	exports.Memory = Memory;
	/**
	 * Auxiliary call stack for the FFI calls.
	 *
	 * Lifecyle of a call stack.
	 * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
	 * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
	 *   can still call into storeXX
	 * - Calls into commitToWasmMemory once.
	 * - reset.
	 */
	class CachedCallStack {
	    constructor(memory, allocSpace, freeSpace) {
	        /** List of temporay arguments that can be disposed during reset. */
	        this.tempArgs = [];
	        this.stackTop = 0;
	        this.basePtr = 0;
	        this.addressToSetTargetValue = [];
	        const initCallStackSize = 128;
	        this.memory = memory;
	        this.cAllocSpace = allocSpace;
	        this.cFreeSpace = freeSpace;
	        this.buffer = new ArrayBuffer(initCallStackSize);
	        this.basePtr = this.cAllocSpace(initCallStackSize);
	        this.viewU8 = new Uint8Array(this.buffer);
	        this.viewI32 = new Int32Array(this.buffer);
	        this.viewU32 = new Uint32Array(this.buffer);
	        this.viewF64 = new Float64Array(this.buffer);
	        this.updateViews();
	    }
	    dispose() {
	        if (this.basePtr != 0) {
	            this.cFreeSpace(this.basePtr);
	            this.basePtr = 0;
	        }
	    }
	    /**
	     * Rest the call stack so that it can be reused again.
	     */
	    reset() {
	        this.stackTop = 0;
	        support.assert(this.addressToSetTargetValue.length == 0);
	        while (this.tempArgs.length != 0) {
	            this.tempArgs.pop().dispose();
	        }
	    }
	    /**
	     * Commit all the cached data to WasmMemory.
	     * This function can only be called once.
	     * No further store function should be called.
	     *
	     * @param nbytes Number of bytes to be stored.
	     */
	    commitToWasmMemory(nbytes = this.stackTop) {
	        // commit all pointer values.
	        while (this.addressToSetTargetValue.length != 0) {
	            const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop();
	            this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
	        }
	        this.memory.storeRawBytes(this.basePtr, this.viewU8.slice(0, nbytes));
	    }
	    /**
	     * Allocate space by number of bytes
	     * @param nbytes Number of bytes.
	     * @note This function always allocate space that aligns to 64bit.
	     */
	    allocRawBytes(nbytes) {
	        // always aligns to 64bit
	        nbytes = ((nbytes + 7) >> 3) << 3;
	        if (this.stackTop + nbytes > this.buffer.byteLength) {
	            const newSize = Math.max(this.buffer.byteLength * 2, this.stackTop + nbytes);
	            const oldU8 = this.viewU8;
	            this.buffer = new ArrayBuffer(newSize);
	            this.updateViews();
	            this.viewU8.set(oldU8);
	            if (this.basePtr != 0) {
	                this.cFreeSpace(this.basePtr);
	            }
	            this.basePtr = this.cAllocSpace(newSize);
	        }
	        const retOffset = this.stackTop;
	        this.stackTop += nbytes;
	        return retOffset;
	    }
	    /**
	     * Allocate space for pointers.
	     * @param count Number of pointers.
	     * @returns The allocated pointer array.
	     */
	    allocPtrArray(count) {
	        return this.allocRawBytes(this.memory.sizeofPtr() * count);
	    }
	    /**
	     * Get the real pointer from offset values.
	     * Note that the returned value becomes obsolete if alloc is called on the stack.
	     * @param offset The allocated offset.
	     */
	    ptrFromOffset(offset) {
	        return this.basePtr + offset;
	    }
	    // Store APIs
	    storePtr(offset, value) {
	        if (this.memory.wasm32) {
	            this.storeU32(offset, value);
	        }
	        else {
	            this.storeI64(offset, value);
	        }
	    }
	    storeUSize(offset, value) {
	        if (this.memory.wasm32) {
	            this.storeU32(offset, value);
	        }
	        else {
	            this.storeI64(offset, value);
	        }
	    }
	    storeI32(offset, value) {
	        this.viewI32[offset >> 2] = value;
	    }
	    storeU32(offset, value) {
	        this.viewU32[offset >> 2] = value;
	    }
	    storeI64(offset, value) {
	        // For now, just store as 32bit
	        // NOTE: wasm always uses little endian.
	        const low = value & 0xffffffff;
	        const base = offset >> 2;
	        this.viewI32[base] = low;
	        this.viewI32[base + 1] = 0;
	    }
	    storeF64(offset, value) {
	        this.viewF64[offset >> 3] = value;
	    }
	    storeRawBytes(offset, bytes) {
	        this.viewU8.set(bytes, offset);
	    }
	    /**
	     * Allocate then set C-String pointer to the offset.
	     * This function will call into allocBytes to allocate necessary data.
	     * The address won't be set immediately(because the possible change of basePtr)
	     * and will be filled when we commit the data.
	     *
	     * @param offset The offset to set ot data pointer.
	     * @param data The string content.
	     */
	    allocThenSetArgString(offset, data) {
	        const strOffset = this.allocRawBytes(data.length + 1);
	        this.storeRawBytes(strOffset, support.StringToUint8Array(data));
	        this.addressToSetTargetValue.push([offset, strOffset]);
	    }
	    /**
	     * Allocate then set the argument location with a TVMByteArray.
	     * Allocate new temporary space for bytes.
	     *
	     * @param offset The offset to set ot data pointer.
	     * @param data The string content.
	     */
	    allocThenSetArgBytes(offset, data) {
	        // Note: size of size_t equals sizeof ptr.
	        const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
	        const dataOffset = this.allocRawBytes(data.length);
	        this.storeRawBytes(dataOffset, data);
	        this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
	        this.addressToSetTargetValue.push([offset, headerOffset]);
	        this.addressToSetTargetValue.push([headerOffset, dataOffset]);
	    }
	    /**
	     * Update internal cache views.
	     */
	    updateViews() {
	        this.viewU8 = new Uint8Array(this.buffer);
	        this.viewI32 = new Int32Array(this.buffer);
	        this.viewU32 = new Uint32Array(this.buffer);
	        this.viewF64 = new Float64Array(this.buffer);
	    }
	}
	exports.CachedCallStack = CachedCallStack;

	});

	unwrapExports(memory);
	memory.CachedCallStack;
	memory.Memory;

	var environment = createCommonjsModule(function (module, exports) {
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.Environment = void 0;

	/**
	 * Detect library provider from the importObject.
	 *
	 * @param importObject The import object.
	 */
	function detectLibraryProvider(importObject) {
	    if (importObject["wasmLibraryProvider"] &&
	        importObject["wasmLibraryProvider"]["start"] &&
	        importObject["wasmLibraryProvider"]["imports"] !== undefined) {
	        const item = importObject;
	        // create provider so that we capture imports in the provider.
	        return {
	            imports: item.wasmLibraryProvider.imports,
	            start: (inst) => {
	                item.wasmLibraryProvider.start(inst);
	            },
	        };
	    }
	    else if (importObject["imports"] && importObject["start"] !== undefined) {
	        return importObject;
	    }
	    else if (importObject["wasiImport"] && importObject["start"] !== undefined) {
	        // WASI
	        return {
	            imports: {
	                "wasi_snapshot_preview1": importObject["wasiImport"],
	            },
	            start: (inst) => {
	                importObject["start"](inst);
	            }
	        };
	    }
	    else {
	        return undefined;
	    }
	}
	/**
	 * Environment to impelement most of the JS library functions.
	 */
	class Environment {
	    constructor(importObject = {}, logger = console.log) {
	        /**
	         * Maintains a table of FTVMWasmPackedCFunc that the C part
	         * can call via TVMWasmPackedCFunc.
	         *
	         * We maintain a separate table so that we can have un-limited amount
	         * of functions that do not maps to the address space.
	         */
	        this.packedCFuncTable = [
	            undefined,
	        ];
	        /**
	         * Free table index that can be recycled.
	         */
	        this.packedCFuncTableFreeId = [];
	        this.logger = logger;
	        this.libProvider = detectLibraryProvider(importObject);
	        // get imports from the provider
	        if (this.libProvider !== undefined) {
	            this.imports = this.libProvider.imports;
	        }
	        else {
	            this.imports = importObject;
	        }
	        // update with more functions
	        this.imports.env = this.environment(this.imports.env);
	    }
	    /** Mark the start of the instance. */
	    start(inst) {
	        if (this.libProvider !== undefined) {
	            this.libProvider.start(inst);
	        }
	    }
	    environment(initEnv) {
	        // default env can be be overriden by libraries.
	        const defaultEnv = {
	            "__cxa_thread_atexit": () => { },
	            // eslint-disable-next-line @typescript-eslint/no-unused-vars
	            "emscripten_notify_memory_growth": (index) => { }
	        };
	        const wasmPackedCFunc = (args, typeCodes, nargs, ret, resourceHandle) => {
	            const cfunc = this.packedCFuncTable[resourceHandle];
	            support.assert(cfunc !== undefined);
	            return cfunc(args, typeCodes, nargs, ret, resourceHandle);
	        };
	        const wasmPackedCFuncFinalizer = (resourceHandle) => {
	            this.packedCFuncTable[resourceHandle] = undefined;
	            this.packedCFuncTableFreeId.push(resourceHandle);
	        };
	        const newEnv = {
	            TVMWasmPackedCFunc: wasmPackedCFunc,
	            TVMWasmPackedCFuncFinalizer: wasmPackedCFuncFinalizer,
	            "__console_log": (msg) => {
	                this.logger(msg);
	            }
	        };
	        return Object.assign(defaultEnv, initEnv, newEnv);
	    }
	}
	exports.Environment = Environment;

	});

	unwrapExports(environment);
	environment.Environment;

	var webgpu = createCommonjsModule(function (module, exports) {
	var __awaiter = (commonjsGlobal && commonjsGlobal.__awaiter) || function (thisArg, _arguments, P, generator) {
	    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
	    return new (P || (P = Promise))(function (resolve, reject) {
	        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
	        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
	        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
	        step((generator = generator.apply(thisArg, _arguments || [])).next());
	    });
	};
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.WebGPUContext = exports.detectGPUDevice = void 0;
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

	/**
	 * DetectGPU device in the environment.
	 */
	function detectGPUDevice() {
	    return __awaiter(this, void 0, void 0, function* () {
	        if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
	            const adapter = yield navigator.gpu.requestAdapter({ "powerPreference": "high-performance" });
	            if (adapter == null) {
	                throw Error("Cannot find adapter that matches the request");
	            }
	            const computeMB = (value) => {
	                return Math.ceil(value / (1 << 20)) + "MB";
	            };
	            // more detailed error message
	            const requiedMaxBufferSize = 1 << 30;
	            if (requiedMaxBufferSize > adapter.limits.maxBufferSize) {
	                throw Error(`Cannot initialize runtime because of requested maxBufferSize ` +
	                    `exceeds limit. requested=${computeMB(requiedMaxBufferSize)}, ` +
	                    `limit=${computeMB(adapter.limits.maxBufferSize)}. ` +
	                    `This error may be caused by an older version of the browser (e.g. Chrome 112). ` +
	                    `You can try to upgrade your browser to Chrome 113 or later.`);
	            }
	            const requiredMaxStorageBufferBindingSize = 1 << 30;
	            if (requiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
	                throw Error(`Cannot initialize runtime because of requested maxStorageBufferBindingSize ` +
	                    `exceeds limit. requested=${computeMB(requiredMaxStorageBufferBindingSize)}, ` +
	                    `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. `);
	            }
	            const requiredMaxComputeWorkgroupStorageSize = 32 << 10;
	            if (requiredMaxComputeWorkgroupStorageSize > adapter.limits.maxComputeWorkgroupStorageSize) {
	                throw Error(`Cannot initialize runtime because of requested maxComputeWorkgroupStorageSize ` +
	                    `exceeds limit. requested=${requiredMaxComputeWorkgroupStorageSize}, ` +
	                    `limit=${adapter.limits.maxComputeWorkgroupStorageSize}. `);
	            }
	            const adapterInfo = yield adapter.requestAdapterInfo();
	            const device = yield adapter.requestDevice({
	                requiredLimits: {
	                    maxBufferSize: requiedMaxBufferSize,
	                    maxStorageBufferBindingSize: requiredMaxStorageBufferBindingSize,
	                    maxComputeWorkgroupStorageSize: requiredMaxComputeWorkgroupStorageSize,
	                }
	            });
	            return {
	                adapter: adapter,
	                adapterInfo: adapterInfo,
	                device: device
	            };
	        }
	        else {
	            return undefined;
	        }
	    });
	}
	exports.detectGPUDevice = detectGPUDevice;
	const canvasRenderWGSL = `
@group(0) @binding(0) var my_sampler : sampler;
@group(0) @binding(1) var my_texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.position = vec4(pos[vidx], 0.0, 1.0);
  output.uv = uv[vidx];
  return output;
}

@fragment
fn fragment_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(my_texture, my_sampler, uv);
}

@fragment
fn fragment_clear(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return vec4(1.0, 1.0, 1.0, 1.0);
}
`;
	class CanvaRenderManager {
	    constructor(device, canvas) {
	        this.device = device;
	        const ctx = canvas.getContext("webgpu");
	        if (ctx == null) {
	            throw Error("Cannot bind WebGPU context");
	        }
	        this.canvasContext = ctx;
	        this.canvasTextureFormat = navigator.gpu.getPreferredCanvasFormat();
	        this.canvasContext.configure({
	            device: this.device,
	            format: this.canvasTextureFormat,
	            alphaMode: "opaque",
	        });
	        this.renderPipeline = device.createRenderPipeline({
	            layout: "auto",
	            vertex: {
	                module: device.createShaderModule({
	                    code: canvasRenderWGSL,
	                }),
	                entryPoint: "vertex_main",
	            },
	            fragment: {
	                module: device.createShaderModule({
	                    code: canvasRenderWGSL,
	                }),
	                entryPoint: "fragment_main",
	                targets: [{
	                        format: this.canvasTextureFormat,
	                    }],
	            },
	            primitive: {
	                topology: "triangle-list",
	            },
	        });
	        this.clearPipeline = device.createRenderPipeline({
	            layout: "auto",
	            vertex: {
	                module: device.createShaderModule({
	                    code: canvasRenderWGSL,
	                }),
	                entryPoint: "vertex_main",
	            },
	            fragment: {
	                module: device.createShaderModule({
	                    code: canvasRenderWGSL,
	                }),
	                entryPoint: "fragment_clear",
	                targets: [{
	                        format: this.canvasTextureFormat,
	                    }],
	            },
	            primitive: {
	                topology: "triangle-list",
	            },
	        });
	        this.renderSampler = device.createSampler({
	            magFilter: "linear",
	            minFilter: "linear",
	        });
	        // staging texture always be in RGBA
	        this.stagingTexture = device.createTexture({
	            size: [canvas.height, canvas.width, 1],
	            format: "rgba8unorm",
	            usage: GPUTextureUsage.TEXTURE_BINDING |
	                GPUTextureUsage.COPY_DST |
	                GPUTextureUsage.RENDER_ATTACHMENT,
	        });
	    }
	    clear() {
	        const commandEncoder = this.device.createCommandEncoder();
	        const passEncoder = commandEncoder.beginRenderPass({
	            colorAttachments: [
	                {
	                    view: this.canvasContext.getCurrentTexture().createView(),
	                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
	                    loadOp: "clear",
	                    storeOp: "store",
	                },
	            ],
	        });
	        passEncoder.setPipeline(this.clearPipeline);
	        const renderBindingGroup = this.device.createBindGroup({
	            layout: this.renderPipeline.getBindGroupLayout(0),
	            entries: [
	                { binding: 0, resource: this.renderSampler },
	                { binding: 1, resource: this.stagingTexture.createView() },
	            ],
	        });
	        passEncoder.setBindGroup(0, renderBindingGroup);
	        passEncoder.draw(6, 1, 0, 0);
	        passEncoder.end();
	        this.device.queue.submit([commandEncoder.finish()]);
	    }
	    draw(buffer, height, width) {
	        // resize the staging texture
	        if (height != this.stagingTexture.height || width != this.stagingTexture.width) {
	            this.stagingTexture.destroy();
	            this.stagingTexture = this.device.createTexture({
	                size: [height, width, 1],
	                format: "rgba8unorm",
	                usage: GPUTextureUsage.TEXTURE_BINDING |
	                    GPUTextureUsage.COPY_DST |
	                    GPUTextureUsage.RENDER_ATTACHMENT,
	            });
	        }
	        const commandEncoder = this.device.createCommandEncoder();
	        commandEncoder.copyBufferToTexture({
	            buffer: buffer,
	            offset: 0,
	            bytesPerRow: this.stagingTexture.width * 4
	        }, {
	            texture: this.stagingTexture
	        }, {
	            width: this.stagingTexture.width,
	            height: this.stagingTexture.height
	        });
	        const passEncoder = commandEncoder.beginRenderPass({
	            colorAttachments: [
	                {
	                    view: this.canvasContext.getCurrentTexture().createView(),
	                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
	                    loadOp: "clear",
	                    storeOp: "store",
	                },
	            ],
	        });
	        passEncoder.setPipeline(this.renderPipeline);
	        const renderBindingGroup = this.device.createBindGroup({
	            layout: this.renderPipeline.getBindGroupLayout(0),
	            entries: [
	                { binding: 0, resource: this.renderSampler },
	                { binding: 1, resource: this.stagingTexture.createView() },
	            ],
	        });
	        passEncoder.setBindGroup(0, renderBindingGroup);
	        passEncoder.draw(6, 1, 0, 0);
	        passEncoder.end();
	        this.device.queue.submit([commandEncoder.finish()]);
	    }
	    dispose() {
	        this.stagingTexture.destroy();
	    }
	}
	/**
	 * WebGPU context
	 * Manages all the webgpu resources here.
	 */
	class WebGPUContext {
	    constructor(memory, device) {
	        // internal data
	        this.bufferTable = [undefined];
	        this.bufferTableFreeId = [];
	        this.podArgStagingBuffers = [];
	        this.canvasRenderManager = undefined;
	        // number of pod arg staging buffers
	        this.maxNumPodArgsStagingBuffers = 2;
	        // flags for debugging
	        // stats of the runtime.
	        // peak allocation
	        this.peakAllocatedBytes = 0;
	        // current allocation
	        this.currAllocatedBytes = 0;
	        // all allocation(ignoring free)
	        this.allAllocatedBytes = 0;
	        // shader submit counter
	        this.shaderSubmitCounter = 0;
	        // limite number of shaders to be submitted, useful for debugging, default to -1
	        this.debugShaderSubmitLimit = -1;
	        // log and sync each step
	        this.debugLogFinish = false;
	        this.memory = memory;
	        this.device = device;
	    }
	    /**
	     * Dispose context.
	     */
	    dispose() {
	        var _a, _b, _c;
	        (_a = this.canvasRenderManager) === null || _a === void 0 ? void 0 : _a.dispose();
	        this.bufferTableFreeId = [];
	        while (this.bufferTable.length != 0) {
	            (_b = this.bufferTable.pop()) === null || _b === void 0 ? void 0 : _b.destroy();
	        }
	        while (this.podArgStagingBuffers.length != 0) {
	            (_c = this.podArgStagingBuffers.pop()) === null || _c === void 0 ? void 0 : _c.destroy();
	        }
	        this.device.destroy();
	    }
	    /**
	     * Wait for all pending GPU tasks to complete
	     */
	    sync() {
	        return __awaiter(this, void 0, void 0, function* () {
	            yield this.device.queue.onSubmittedWorkDone();
	        });
	    }
	    /**
	     * Obtain the runtime information in readable format.
	     */
	    runtimeStatsText() {
	        let info = "peak-memory=" + Math.ceil(this.peakAllocatedBytes / (1 << 20)) + " MB";
	        info += ", all-memory=" + Math.ceil(this.allAllocatedBytes / (1 << 20)) + " MB";
	        info += ", shader-submissions=" + this.shaderSubmitCounter;
	        return info;
	    }
	    /**
	     * Draw image from data in storage buffer.
	     * @param ptr The GPU ptr
	     * @param height The height of the image.
	     * @param width The width of the image.
	     */
	    drawImageFromBuffer(ptr, height, width) {
	        if (this.canvasRenderManager == undefined) {
	            throw Error("Do not have a canvas context, call bindCanvas first");
	        }
	        this.canvasRenderManager.draw(this.gpuBufferFromPtr(ptr), height, width);
	    }
	    /**
	     * Copy raw bytes into buffer ptr.
	     *
	     * @param rawBytes The raw bytes
	     * @param toPtr The target gpu buffer ptr
	     * @param toOffset The beginning offset
	     * @param nbytes Number of bytes
	     */
	    copyRawBytesToBuffer(rawBytes, toPtr, toOffset, nbytes) {
	        // Perhaps it would be more useful to use a staging buffer?
	        this.device.queue.writeBuffer(this.gpuBufferFromPtr(toPtr), toOffset, rawBytes, 0, nbytes);
	    }
	    /**
	     * Clear canvas
	     */
	    clearCanvas() {
	        var _a;
	        (_a = this.canvasRenderManager) === null || _a === void 0 ? void 0 : _a.clear();
	    }
	    /**
	     * Bind a canvas element to the runtime.
	     * @param canvas The HTML canvas/
	     */
	    bindCanvas(canvas) {
	        this.canvasRenderManager = new CanvaRenderManager(this.device, canvas);
	    }
	    /**
	     * Create a PackedFunc that runs the given shader
	     * via createComputePipeline
	     *
	     * @param info The function information already parsed as a record.
	     * @param code The shader data(in WGSL)
	     * @returns The shader
	     */
	    createShader(finfo, code) {
	        return this.createShadeInternal(finfo, code, false);
	    }
	    /**
	     * Create a PackedFunc that runs the given shader asynchrously
	     * via createComputePipelineAsync
	     *
	     * @param info The function information already parsed as a record.
	     * @param code The shader data(in WGSL)
	     * @returns The shader
	     */
	    createShaderAsync(finfo, code) {
	        return __awaiter(this, void 0, void 0, function* () {
	            return yield this.createShadeInternal(finfo, code, true);
	        });
	    }
	    /**
	     * Get the pod arg staging buffer
	     * \param nbytes The minimum size.
	     * \return The allocated buffer
	     */
	    getPodArgsBuffer(nbytes) {
	        let buffer = undefined;
	        if (this.podArgStagingBuffers.length >= this.maxNumPodArgsStagingBuffers) {
	            buffer = this.podArgStagingBuffers.shift();
	        }
	        // minimum of 16 bytes
	        let allocSize = 16;
	        if (buffer !== undefined) {
	            allocSize = buffer.size;
	            if (buffer.size < nbytes) {
	                buffer.destroy();
	                buffer = undefined;
	            }
	        }
	        while (allocSize < nbytes) {
	            allocSize *= 2;
	        }
	        if (buffer == undefined) {
	            // create uniform buffer
	            buffer = this.device.createBuffer({
	                size: allocSize,
	                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	            });
	        }
	        support.assert(nbytes <= buffer.size);
	        return buffer;
	    }
	    /**
	     * Internal impl of createShader for both async and sync mode.
	     *
	     * @param info The function information already parsed as a record.
	     * @param code The shader data(in WGSL)
	     * @param asyncMode Whether use async mode.
	     * @returns The shader function or promise of shader func.
	     */
	    createShadeInternal(finfo, code, asyncMode) {
	        const dispatchToDim = [];
	        let paramWriteAccess = [];
	        for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
	            const tag = finfo.launch_param_tags[i];
	            if (tag.startsWith("blockIdx.")) {
	                const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
	                support.assert(target >= 0 && target < 3);
	                dispatchToDim.push(target);
	            }
	            else if (tag.startsWith("threadIdx.")) {
	                const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
	                support.assert(target >= 0 && target < 3);
	                dispatchToDim.push(target + 3);
	            }
	            else if (tag.startsWith("paramWriteAccess:")) {
	                paramWriteAccess = JSON.parse(tag.substring(17));
	            }
	            else {
	                throw new Error("Cannot handle thread_axis " + tag);
	            }
	        }
	        const layoutEntries = [];
	        const bufferArgIndices = [];
	        const podArgIndices = [];
	        for (let i = 0; i < finfo.arg_types.length; ++i) {
	            const dtype = finfo.arg_types[i];
	            if (dtype == "handle") {
	                layoutEntries.push({
	                    binding: bufferArgIndices.length,
	                    visibility: GPUShaderStage.COMPUTE,
	                    buffer: {
	                        type: paramWriteAccess[bufferArgIndices.length] ? "storage" : "read-only-storage"
	                    }
	                });
	                bufferArgIndices.push(i);
	            }
	            else if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype.startsWith("float")) {
	                podArgIndices.push(i);
	            }
	            else {
	                throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
	            }
	        }
	        support.assert(paramWriteAccess.length == bufferArgIndices.length);
	        // POD arguments are pass in the end
	        layoutEntries.push({
	            binding: bufferArgIndices.length,
	            visibility: GPUShaderStage.COMPUTE,
	            buffer: {
	                type: "uniform"
	            }
	        });
	        const bindGroupLayout = this.device.createBindGroupLayout({
	            entries: layoutEntries
	        });
	        const pipelineLayout = this.device.createPipelineLayout({
	            bindGroupLayouts: [bindGroupLayout]
	        });
	        // Function to create the pipeline.
	        const createShaderFunc = (pipeline) => {
	            const submitShader = (...args) => {
	                if (this.debugShaderSubmitLimit != -1 &&
	                    this.shaderSubmitCounter >= this.debugShaderSubmitLimit) {
	                    this.shaderSubmitCounter += 1;
	                    return;
	                }
	                const commandEncoder = this.device.createCommandEncoder();
	                const compute = commandEncoder.beginComputePass();
	                compute.setPipeline(pipeline);
	                const bindGroupEntries = [];
	                const numBufferOrPodArgs = bufferArgIndices.length + podArgIndices.length;
	                support.assert(args.length == numBufferOrPodArgs + dispatchToDim.length);
	                const workDim = [1, 1, 1, 1, 1, 1];
	                for (let i = 0; i < dispatchToDim.length; ++i) {
	                    workDim[dispatchToDim[i]] = args[numBufferOrPodArgs + i];
	                }
	                // get around 65535 restriction of blockIdx.x
	                if (workDim[2] != 1) {
	                    throw Error("WebGPU: blockIdx.z is reserved for internal use");
	                }
	                const packDimX = workDim[0];
	                // spread thinsg out into blockIdx.z
	                if (workDim[0] >= (1 << 16)) {
	                    let wl_x = workDim[0];
	                    let wl_z = workDim[2];
	                    while (wl_x >= (1 << 16)) {
	                        if (wl_x % 2 == 0) {
	                            wl_x = wl_x / 2;
	                        }
	                        else {
	                            // pad up
	                            wl_x = (wl_x + 1) / 2;
	                        }
	                        wl_z *= 2;
	                    }
	                    workDim[0] = wl_x;
	                    workDim[2] = wl_z;
	                    support.assert(wl_x * wl_z >= packDimX);
	                }
	                for (let i = 0; i < bufferArgIndices.length; ++i) {
	                    bindGroupEntries.push({
	                        binding: i,
	                        resource: {
	                            buffer: this.gpuBufferFromPtr(args[bufferArgIndices[i]])
	                        }
	                    });
	                }
	                // push pod buffer
	                const sizeOfI32 = 4;
	                const podArgBuffer = this.getPodArgsBuffer((podArgIndices.length + 1) * sizeOfI32);
	                const i32View = new Int32Array(podArgIndices.length + 1);
	                const u32View = new Uint32Array(i32View.buffer);
	                const f32View = new Float32Array(i32View.buffer);
	                for (let i = 0; i < podArgIndices.length; ++i) {
	                    const value = args[podArgIndices[i]];
	                    const dtype = finfo.arg_types[podArgIndices[i]];
	                    if (dtype.startsWith("int")) {
	                        i32View[i] = value;
	                    }
	                    else if (dtype.startsWith("uint")) {
	                        u32View[i] = value;
	                    }
	                    else if (dtype.startsWith("float")) {
	                        f32View[i] = value;
	                    }
	                    else {
	                        throw Error("Unknown pod dtype " + dtype);
	                    }
	                }
	                // always pass in dim z launching grid size in
	                u32View[podArgIndices.length] = packDimX;
	                this.device.queue.writeBuffer(podArgBuffer, 0, i32View.buffer);
	                bindGroupEntries.push({
	                    binding: bufferArgIndices.length,
	                    resource: {
	                        buffer: podArgBuffer,
	                        size: i32View.buffer.byteLength
	                    }
	                });
	                compute.setBindGroup(0, this.device.createBindGroup({
	                    layout: bindGroupLayout,
	                    entries: bindGroupEntries
	                }));
	                compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);
	                compute.end();
	                const command = commandEncoder.finish();
	                this.device.queue.submit([command]);
	                if (this.debugLogFinish) {
	                    const currCounter = this.shaderSubmitCounter;
	                    this.device.queue.onSubmittedWorkDone().then(() => {
	                        console.log("[" + currCounter + "][Debug] finish shader" + finfo.name);
	                    });
	                }
	                this.shaderSubmitCounter += 1;
	            };
	            return submitShader;
	        };
	        const shaderModule = this.device.createShaderModule({
	            code: code,
	            hints: {
	                main: {
	                    layout: pipelineLayout
	                }
	            }
	        });
	        if (asyncMode) {
	            return this.device.createComputePipelineAsync({
	                layout: pipelineLayout,
	                compute: {
	                    module: shaderModule,
	                    entryPoint: finfo.name
	                }
	            }).then((pipeline) => {
	                return createShaderFunc(pipeline);
	            });
	        }
	        else {
	            const pipeline = this.device.createComputePipeline({
	                layout: pipelineLayout,
	                compute: {
	                    module: shaderModule,
	                    entryPoint: finfo.name
	                }
	            });
	            return createShaderFunc(pipeline);
	        }
	    }
	    /**
	     * Get the device API according to its name
	     * @param The name of the API.
	     * @returns The corresponding device api.
	     */
	    getDeviceAPI(name) {
	        if (name == "deviceAllocDataSpace") {
	            return (nbytes) => {
	                return this.deviceAllocDataSpace(nbytes);
	            };
	        }
	        else if (name == "deviceFreeDataSpace") {
	            return (ptr) => {
	                return this.deviceFreeDataSpace(ptr);
	            };
	        }
	        else if (name == "deviceCopyToGPU") {
	            return (from, to, toOffset, nbytes) => {
	                this.deviceCopyToGPU(from, to, toOffset, nbytes);
	            };
	        }
	        else if (name == "deviceCopyFromGPU") {
	            return (from, fromOffset, to, nbytes) => {
	                this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
	            };
	        }
	        else if (name == "deviceCopyWithinGPU") {
	            return (from, fromOffset, to, toOffset, nbytes) => {
	                this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
	            };
	        }
	        else {
	            throw new Error("Unknown DeviceAPI function " + name);
	        }
	    }
	    // DeviceAPI
	    deviceAllocDataSpace(nbytes) {
	        // allocate 0 bytes buffer as 1 bytes buffer.
	        if (nbytes == 0) {
	            nbytes = 1;
	        }
	        const buffer = this.device.createBuffer({
	            size: nbytes,
	            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	        });
	        this.currAllocatedBytes += nbytes;
	        this.allAllocatedBytes += nbytes;
	        if (this.currAllocatedBytes > this.peakAllocatedBytes) {
	            this.peakAllocatedBytes = this.currAllocatedBytes;
	        }
	        const ptr = this.attachToBufferTable(buffer);
	        return ptr;
	    }
	    deviceFreeDataSpace(ptr) {
	        const idx = ptr;
	        const buffer = this.bufferTable[idx];
	        this.bufferTable[idx] = undefined;
	        support.assert(buffer !== undefined);
	        this.bufferTableFreeId.push(idx);
	        this.currAllocatedBytes -= buffer.size;
	        buffer.destroy();
	    }
	    deviceCopyToGPU(from, to, toOffset, nbytes) {
	        // Perhaps it would be more useful to use a staging buffer?
	        const rawBytes = this.memory.loadRawBytes(from, nbytes);
	        this.device.queue.writeBuffer(this.gpuBufferFromPtr(to), toOffset, rawBytes, 0, nbytes);
	    }
	    deviceCopyFromGPU(from, fromOffset, to, nbytes) {
	        // Perhaps it would be more useful to resuse a staging buffer?
	        const gpuTemp = this.device.createBuffer({
	            size: nbytes,
	            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	        });
	        const copyEncoder = this.device.createCommandEncoder();
	        copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, gpuTemp, 0, nbytes);
	        const copyCommands = copyEncoder.finish();
	        this.device.queue.submit([copyCommands]);
	        gpuTemp.mapAsync(GPUMapMode.READ).then(() => {
	            const data = gpuTemp.getMappedRange();
	            this.memory.storeRawBytes(to, new Uint8Array(data));
	            gpuTemp.destroy();
	        });
	    }
	    deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {
	        const copyEncoder = this.device.createCommandEncoder();
	        copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, this.gpuBufferFromPtr(to), toOffset, nbytes);
	        const copyCommands = copyEncoder.finish();
	        this.device.queue.submit([copyCommands]);
	    }
	    gpuBufferFromPtr(ptr) {
	        const buffer = this.bufferTable[ptr];
	        support.assert(buffer !== undefined);
	        return buffer;
	    }
	    attachToBufferTable(buffer) {
	        if (this.bufferTableFreeId.length != 0) {
	            const idx = this.bufferTableFreeId.pop();
	            this.bufferTable[idx] = buffer;
	            return idx;
	        }
	        else {
	            const idx = this.bufferTable.length;
	            this.bufferTable.push(buffer);
	            return idx;
	        }
	    }
	}
	exports.WebGPUContext = WebGPUContext;

	});

	unwrapExports(webgpu);
	webgpu.WebGPUContext;
	webgpu.detectGPUDevice;

	var compact = createCommonjsModule(function (module, exports) {
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
	/** NodeJS and Web compact layer */
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.createWebSocket = exports.getPerformance = void 0;
	/**
	 * Get performance measurement.
	 */
	function getPerformance() {
	    if (typeof performance == "undefined") {
	        // eslint-disable-next-line @typescript-eslint/no-var-requires
	        const performanceNode = perf_hooks__default["default"];
	        return performanceNode.performance;
	    }
	    else {
	        return performance;
	    }
	}
	exports.getPerformance = getPerformance;
	/**
	 * Create a new websocket for a given URL
	 * @param url The url.
	 */
	function createWebSocket(url) {
	    if (typeof WebSocket == "undefined") {
	        // eslint-disable-next-line @typescript-eslint/no-var-requires
	        const WebSocket = ws__default["default"];
	        return new WebSocket(url);
	    }
	    else {
	        return new WebSocket(url);
	    }
	}
	exports.createWebSocket = createWebSocket;

	});

	unwrapExports(compact);
	compact.createWebSocket;
	compact.getPerformance;

	var runtime = createCommonjsModule(function (module, exports) {
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
	var __awaiter = (commonjsGlobal && commonjsGlobal.__awaiter) || function (thisArg, _arguments, P, generator) {
	    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
	    return new (P || (P = Promise))(function (resolve, reject) {
	        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
	        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
	        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
	        step((generator = generator.apply(thisArg, _arguments || [])).next());
	    });
	};
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.instantiate = exports.Instance = exports.VirtualMachine = exports.TVMArray = exports.TVMObject = exports.Module = exports.NDArray = exports.DLDataType = exports.DLDevice = exports.Scalar = void 0;





	/**
	 * @internal
	 * FFI Library wrapper, maintains most runtime states.
	 */
	class FFILibrary {
	    constructor(wasmInstance, imports) {
	        this.recycledCallStacks = [];
	        this.wasmInstance = wasmInstance;
	        this.memory = new memory.Memory(this.detectWasmMemory(this.wasmInstance, imports));
	        support.assert(this.wasmInstance.exports !== undefined, "Expect the library module contains exports");
	        this.exports = this.wasmInstance.exports;
	        this.wasm32 = this.memory.wasm32;
	        this.validateInstance();
	    }
	    dispose() {
	        var _a;
	        while (this.recycledCallStacks.length != 0) {
	            this.recycledCallStacks.pop().dispose();
	        }
	        (_a = this.webGPUContext) === null || _a === void 0 ? void 0 : _a.dispose();
	    }
	    sizeofPtr() {
	        return this.memory.sizeofPtr();
	    }
	    checkCall(code) {
	        if (code != 0) {
	            const msgPtr = this.exports
	                .TVMGetLastError();
	            throw new Error("TVMError: " + this.memory.loadCString(msgPtr));
	        }
	    }
	    getOrAllocCallStack() {
	        if (this.recycledCallStacks.length != 0) {
	            return this.recycledCallStacks.pop();
	        }
	        return new memory.CachedCallStack(this.memory, this.exports.TVMWasmAllocSpace, this.exports.TVMWasmFreeSpace);
	    }
	    recycleCallStack(callstack) {
	        callstack.reset();
	        this.recycledCallStacks.push(callstack);
	    }
	    validateInstance() {
	        this.checkExports(["TVMWasmAllocSpace", "TVMWasmFreeSpace", "TVMFuncFree"]);
	    }
	    checkExports(funcNames) {
	        const missList = [];
	        for (const name of funcNames) {
	            const f = this.exports[name];
	            if (!(f instanceof Function)) {
	                missList.push(name);
	            }
	        }
	        if (missList.length != 0) {
	            throw new Error("Cannot find " + missList + " in exports");
	        }
	    }
	    detectWasmMemory(instance, imports) {
	        if (instance.exports.memory instanceof WebAssembly.Memory) {
	            return instance.exports.memory;
	        }
	        if (imports.env && imports.env.memory instanceof WebAssembly.Memory) {
	            return imports.env.memory;
	        }
	        throw new Error("Cannt detect wasm memory from imports " +
	            imports +
	            " or exports" +
	            instance.exports);
	    }
	}
	/**
	 * @internal
	 * Manages extra runtime context for the runtime.
	 */
	class RuntimeContext {
	    constructor(getGlobalFunc) {
	        this.autoDisposeScope = [];
	        this.arrayGetItem = getGlobalFunc("runtime.ArrayGetItem");
	        this.arrayGetSize = getGlobalFunc("runtime.ArraySize");
	        this.arrayMake = getGlobalFunc("runtime.Array");
	        this.getSysLib = getGlobalFunc("runtime.SystemLib");
	        this.arrayCacheGet = getGlobalFunc("vm.builtin.ndarray_cache.get");
	        this.arrayCacheRemove = getGlobalFunc("vm.builtin.ndarray_cache.remove");
	        this.arrayCacheUpdate = getGlobalFunc("vm.builtin.ndarray_cache.update");
	        this.arrayCacheClear = getGlobalFunc("vm.builtin.ndarray_cache.clear");
	        this.arrayDecodeStorage = getGlobalFunc("tvmjs.array.decode_storage");
	        this.paramModuleFromCache = getGlobalFunc("vm.builtin.param_module_from_cache");
	        this.makeShapeTuple = getGlobalFunc("runtime.ShapeTuple");
	        this.ndarrayCreateView = getGlobalFunc("runtime.TVMArrayCreateView");
	        this.sampleTopPFromLogits = getGlobalFunc("vm.builtin.sample_top_p_from_logits");
	    }
	    dispose() {
	        // call array cache clear to clear all cached items
	        this.arrayCacheClear.dispose();
	        this.arrayGetItem.dispose();
	        this.arrayGetSize.dispose();
	        this.arrayMake.dispose();
	        this.arrayCacheGet.dispose();
	        this.arrayCacheRemove.dispose();
	        this.arrayCacheUpdate.dispose();
	        this.arrayCacheClear.dispose();
	        this.arrayDecodeStorage.dispose();
	        this.paramModuleFromCache.dispose();
	        this.makeShapeTuple.dispose();
	        this.ndarrayCreateView.dispose();
	        this.sampleTopPFromLogits.dispose();
	    }
	    beginScope() {
	        this.autoDisposeScope.push([]);
	    }
	    endScope() {
	        if (this.autoDisposeScope.length == 0) {
	            throw Error("tvm.endScope called when the stack is empty.");
	        }
	        // automatically dispose all the tracked values in the current scope.
	        const currScope = this.autoDisposeScope.pop();
	        for (let i = 0; i < currScope.length; ++i) {
	            const val = currScope[i];
	            if (val !== undefined) {
	                val.dispose();
	            }
	        }
	    }
	    /**
	     * Track object for dispose in current scope.
	     *
	     * @param obj The object to be tracked.
	     * @returns the same object.
	     * @note This function only needs to be called for raw system C API values.
	     *       The return value of PackedFunc will be automatically tracked.
	     */
	    attachToCurrentScope(obj) {
	        if (this.autoDisposeScope.length == 0) {
	            throw Error("Must call beginScope to use functions that returns TVM objects");
	        }
	        const currScope = this.autoDisposeScope[this.autoDisposeScope.length - 1];
	        currScope.push(obj);
	        return obj;
	    }
	    moveToParentScope(obj) {
	        this.detachFromCurrentScope(obj);
	        if (this.autoDisposeScope.length < 2) {
	            throw Error("moveToParentScope: Parent scope do not exist");
	        }
	        const parentScope = this.autoDisposeScope[this.autoDisposeScope.length - 2];
	        parentScope.push(obj);
	        return obj;
	    }
	    detachFromCurrentScope(obj) {
	        const currScope = this.autoDisposeScope[this.autoDisposeScope.length - 1];
	        let occurance = 0;
	        for (let i = 0; i < currScope.length; ++i) {
	            if (currScope[i] === obj) {
	                occurance += 1;
	                currScope[i] = undefined;
	            }
	        }
	        if (occurance == 0) {
	            throw Error("Cannot find obj in the current auto conversion pool");
	        }
	        if (occurance > 1) {
	            throw Error("Value attached to scope multiple times");
	        }
	        return obj;
	    }
	}
	/**
	 * A typed scalar constant used to represent a typed number
	 * argument to PackedFunc calls.
	 */
	class Scalar {
	    constructor(value, dtype) {
	        this.value = value;
	        this.dtype = dtype;
	    }
	}
	exports.Scalar = Scalar;
	/**
	 * Cell holds the PackedFunc object.
	 */
	class PackedFuncCell {
	    constructor(handle, lib) {
	        this.handle = handle;
	        this.lib = lib;
	    }
	    dispose() {
	        if (this.handle != 0) {
	            this.lib.checkCall(this.lib.exports.TVMFuncFree(this.handle));
	            this.handle = 0;
	        }
	    }
	    getHandle(requireNotNull = true) {
	        if (requireNotNull && this.handle == 0) {
	            throw Error("PackedFunc has already been disposed");
	        }
	        return this.handle;
	    }
	}
	const DeviceEnumToStr = {
	    1: "cpu",
	    2: "cuda",
	    4: "opencl",
	    8: "metal",
	    15: "webgpu"
	};
	const DeviceStrToEnum = {
	    cpu: 1,
	    cuda: 2,
	    cl: 4,
	    opencl: 4,
	    vulkan: 7,
	    metal: 8,
	    webgpu: 15
	};
	/**
	 * Represent a runtime context where a NDArray can reside.
	 */
	class DLDevice {
	    constructor(deviceType, deviceId, lib) {
	        const tp = typeof deviceType;
	        if (tp == "string") {
	            this.deviceType = DeviceStrToEnum[deviceType];
	            if (this.deviceType == undefined) {
	                throw new Error("Cannot recogonize deviceType " + deviceType);
	            }
	        }
	        else if (tp == "number") {
	            this.deviceType = deviceType;
	        }
	        else {
	            throw new Error("Cannot take type " + tp + " as deviceType");
	        }
	        this.deviceId = deviceId;
	        this.lib = lib;
	    }
	    /**
	     * Synchronize the device
	     */
	    sync() {
	        return __awaiter(this, void 0, void 0, function* () {
	            if (this.deviceType == DeviceStrToEnum.webgpu) {
	                support.assert(this.lib.webGPUContext !== undefined);
	                yield this.lib.webGPUContext.sync();
	            }
	        });
	    }
	    toString() {
	        return (DeviceEnumToStr[this.deviceType] + "(" + this.deviceId.toString() + ")");
	    }
	}
	exports.DLDevice = DLDevice;
	const DLDataTypeCodeToStr = {
	    0: "int",
	    1: "uint",
	    2: "float",
	    3: "handle",
	};
	/**
	 * Runtime data type of NDArray.
	 */
	class DLDataType {
	    constructor(code, bits, lanes) {
	        this.code = code;
	        this.bits = bits;
	        this.lanes = lanes;
	    }
	    toString() {
	        const ret = DLDataTypeCodeToStr[this.code] + this.bits.toString();
	        if (this.lanes != 1) {
	            return ret + "x" + this.lanes.toString();
	        }
	        else {
	            return ret;
	        }
	    }
	    numStorageBytes() {
	        return (this.bits * this.lanes + 7) >> 3;
	    }
	}
	exports.DLDataType = DLDataType;
	/**
	 * n-dimnesional array.
	 */
	class NDArray {
	    constructor(handle, isView, lib, ctx) {
	        this.handle = handle;
	        this.isView = isView;
	        this.lib = lib;
	        this.ctx = ctx;
	        if (this.isView) {
	            this.dltensor = handle;
	        }
	        else {
	            this.dltensor = this.getDLTensorFromArrayHandle(this.handle);
	        }
	        // constant offsets.
	        const arrayOffsetData = 0;
	        const arrayOffsetContext = arrayOffsetData + this.lib.sizeofPtr();
	        const arrayOffsetDevType = arrayOffsetContext;
	        const arrayOffsetDevId = arrayOffsetContext + 4 /* I32 */;
	        const arrayOffsetNdim = arrayOffsetContext + 8 /* DLDevice */;
	        const arrayOffsetDtype = arrayOffsetNdim + 4 /* I32 */;
	        const arrayOffsetDtypeCode = arrayOffsetDtype;
	        const arrayOffsetDtypeBits = arrayOffsetDtype + 1 /* U8 */;
	        const arrayOffsetDtypeLanes = arrayOffsetDtypeBits + 1 /* U8 */;
	        const arrayOffsetShape = arrayOffsetDtype + 4 /* DLDataType */;
	        const arrayOffsetStrides = arrayOffsetShape + this.lib.sizeofPtr();
	        const arrayOffsetByteOffset = arrayOffsetStrides + this.lib.sizeofPtr();
	        // dataPtr
	        this.dataPtr = lib.memory.loadPointer(this.dltensor);
	        // ndim
	        this.ndim = lib.memory.loadI32(this.dltensor + arrayOffsetNdim);
	        // shape
	        const cshapePtr = lib.memory.loadPointer(this.dltensor + arrayOffsetShape);
	        this.shape = [];
	        for (let i = 0; i < this.ndim; ++i) {
	            this.shape.push(lib.memory.loadI64(cshapePtr + i * 8 /* I64 */));
	        }
	        // dtype
	        const code = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeCode);
	        const bits = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeBits);
	        const lanes = lib.memory.loadU16(this.dltensor + arrayOffsetDtypeLanes);
	        this.dlDataType = new DLDataType(code, bits, lanes);
	        this.dtype = this.dlDataType.toString();
	        // device
	        const deviceType = lib.memory.loadI32(this.dltensor + arrayOffsetDevType);
	        const deviceId = lib.memory.loadI32(this.dltensor + arrayOffsetDevId);
	        this.device = new DLDevice(deviceType, deviceId, lib);
	        // byte_offset
	        this.byteOffset = lib.memory.loadI64(this.dltensor + arrayOffsetByteOffset);
	    }
	    /**
	     * Create a view of the array.
	     * @param shape The shape of the view.
	     * @returns The new sliced ndarray.
	     */
	    view(shape) {
	        const shapeArray = shape.map((value) => new Scalar(value, "int"));
	        return this.ctx.ndarrayCreateView(this, this.ctx.makeShapeTuple(...shapeArray));
	    }
	    /**
	     * Get handle of ndarray, check it is not null.
	     *
	     * @param requireNotNull require handle is not null.
	     * @returns The handle.
	     */
	    getHandle(requireNotNull = true) {
	        if (requireNotNull && this.handle == 0) {
	            throw Error("NDArray has already been disposed");
	        }
	        return this.handle;
	    }
	    /**
	     * Get dataPtr of NDarray
	     *
	     * @returns The handle.
	     */
	    getDataPtr() {
	        if (this.handle == 0) {
	            throw Error("NDArray has already been disposed");
	        }
	        return this.dataPtr;
	    }
	    dispose() {
	        if (this.handle != 0 && !this.isView) {
	            this.lib.checkCall(this.lib.exports.TVMArrayFree(this.handle));
	            this.handle = 0;
	        }
	    }
	    /**
	     * Copy data from another NDArray or javascript array.
	     * The number of elements must match.
	     *
	     * @param data The source data array.
	     * @returns this
	     */
	    copyFrom(data) {
	        if (data instanceof NDArray) {
	            this.lib.checkCall(this.lib.exports.TVMArrayCopyFromTo(data.getHandle(), this.getHandle(), 0));
	            return this;
	        }
	        else {
	            const size = this.shape.reduce((a, b) => {
	                return a * b;
	            }, 1);
	            if (data.length != size) {
	                throw new Error("data size and shape mismatch data.length" +
	                    data.length +
	                    " vs " +
	                    size);
	            }
	            let buffer;
	            if (this.dtype == "float32") {
	                buffer = Float32Array.from(data).buffer;
	            }
	            else if (this.dtype == "float64") {
	                buffer = Float64Array.from(data).buffer;
	            }
	            else if (this.dtype == "int32") {
	                buffer = Int32Array.from(data).buffer;
	            }
	            else if (this.dtype == "int8") {
	                buffer = Int8Array.from(data).buffer;
	            }
	            else if (this.dtype == "uint8") {
	                buffer = Uint8Array.from(data).buffer;
	            }
	            else {
	                throw new Error("Unsupported data type " + this.dtype);
	            }
	            return this.copyFromRawBytes(new Uint8Array(buffer));
	        }
	    }
	    /**
	     * Copy data from raw bytes.
	     * @param data Uint8Array of bytes.
	     * @returns this
	     */
	    copyFromRawBytes(data) {
	        var _a;
	        // short cut for gpu copy
	        if (this.device.deviceType == DeviceStrToEnum.webgpu) {
	            (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.copyRawBytesToBuffer(data, this.getDataPtr(), 0, data.length);
	            return this;
	        }
	        // CPU copy
	        const size = this.shape.reduce((a, b) => {
	            return a * b;
	        }, 1);
	        const nbytes = this.dlDataType.numStorageBytes() * size;
	        if (nbytes != data.length) {
	            throw new Error("Expect the data's length equals nbytes=" + nbytes);
	        }
	        const stack = this.lib.getOrAllocCallStack();
	        const tempOffset = stack.allocRawBytes(nbytes);
	        const tempPtr = stack.ptrFromOffset(tempOffset);
	        this.lib.memory.storeRawBytes(tempPtr, data);
	        this.lib.checkCall(this.lib.exports.TVMArrayCopyFromBytes(this.getHandle(), tempPtr, nbytes));
	        this.lib.recycleCallStack(stack);
	        return this;
	    }
	    /**
	     * Return a copied Uint8Array of the raw bytes in the NDArray.
	     * @returns The result array.
	     */
	    toRawBytes() {
	        if (this.device.deviceType != DeviceStrToEnum.cpu) {
	            throw new Error("Can only sync copy CPU array, use cpu_arr.copyfrom(gpu_arr) then sync instead.");
	        }
	        const size = this.shape.reduce((a, b) => {
	            return a * b;
	        }, 1);
	        const nbytes = this.dlDataType.numStorageBytes() * size;
	        const stack = this.lib.getOrAllocCallStack();
	        const tempOffset = stack.allocRawBytes(nbytes);
	        const tempPtr = stack.ptrFromOffset(tempOffset);
	        this.lib.checkCall(this.lib.exports.TVMArrayCopyToBytes(this.getHandle(), tempPtr, nbytes));
	        const ret = this.lib.memory.loadRawBytes(tempPtr, nbytes);
	        this.lib.recycleCallStack(stack);
	        return ret;
	    }
	    /**
	     * Return a TypedArray copy of the NDArray, the specific type depends on
	     * the dtype of the NDArray.
	     * @returns The result array.
	     */
	    toArray() {
	        const stype = this.dtype;
	        if (stype == "float32") {
	            return new Float32Array(this.toRawBytes().buffer);
	        }
	        else if (stype == "float64") {
	            return new Float64Array(this.toRawBytes().buffer);
	        }
	        else if (stype == "int32") {
	            return new Int32Array(this.toRawBytes().buffer);
	        }
	        else if (stype == "int8") {
	            return new Int8Array(this.toRawBytes().buffer);
	        }
	        else if (stype == "uint8") {
	            return new Uint8Array(this.toRawBytes().buffer);
	        }
	        else {
	            throw new Error("Unsupported data type " + this.dtype);
	        }
	    }
	    getDLTensorFromArrayHandle(handle) {
	        // Note: this depends on the NDArray C ABI.
	        // keep this function in case of ABI change.
	        return handle;
	    }
	}
	exports.NDArray = NDArray;
	/**
	 * Runtime Module.
	 */
	class Module {
	    constructor(handle, lib, makePackedFunc) {
	        this.handle = handle;
	        this.lib = lib;
	        this.makePackedFunc = makePackedFunc;
	    }
	    dispose() {
	        if (this.handle != 0) {
	            this.lib.checkCall(this.lib.exports.TVMModFree(this.handle));
	            this.handle = 0;
	        }
	    }
	    /**
	     * Get handle of module, check it is not null.
	     *
	     * @param requireNotNull require handle is not null.
	     * @returns The handle.
	     */
	    getHandle(requireNotNull = true) {
	        if (requireNotNull && this.handle == 0) {
	            throw Error("Module has already been disposed");
	        }
	        return this.handle;
	    }
	    /**
	     * Get a function in the module.
	     * @param name The name of the function.
	     * @param queryImports Whether to also query imports
	     * @returns The result function.
	     */
	    getFunction(name, queryImports = true) {
	        if (this.handle == 0) {
	            throw Error("Module has already been disposed");
	        }
	        const stack = this.lib.getOrAllocCallStack();
	        const nameOffset = stack.allocRawBytes(name.length + 1);
	        stack.storeRawBytes(nameOffset, support.StringToUint8Array(name));
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        stack.commitToWasmMemory(outOffset);
	        this.lib.checkCall(this.lib.exports.TVMModGetFunction(this.getHandle(), stack.ptrFromOffset(nameOffset), queryImports ? 1 : 0, outPtr));
	        const handle = this.lib.memory.loadPointer(outPtr);
	        this.lib.recycleCallStack(stack);
	        if (handle == 0) {
	            throw Error("Cannot find function " + name);
	        }
	        const ret = this.makePackedFunc(handle);
	        return ret;
	    }
	    /**
	     * Import another module into the current runtime module.
	     * @param mod The module to be imported.
	     */
	    importModule(mod) {
	        this.lib.checkCall(this.lib.exports.TVMModImport(this.getHandle(), mod.getHandle()));
	    }
	}
	exports.Module = Module;
	/**
	 * Generic object base
	 */
	class TVMObject {
	    constructor(handle, lib, ctx) {
	        this.handle = handle;
	        this.lib = lib;
	        this.ctx = ctx;
	    }
	    dispose() {
	        if (this.handle != 0) {
	            this.lib.checkCall(this.lib.exports.TVMObjectFree(this.handle));
	            this.handle = 0;
	        }
	    }
	    /**
	     * Get handle of module, check it is not null.
	     *
	     * @param requireNotNull require handle is not null.
	     * @returns The handle.
	     */
	    getHandle(requireNotNull = true) {
	        if (requireNotNull && this.handle == 0) {
	            throw Error("Module has already been disposed");
	        }
	        return this.handle;
	    }
	    /** get the type index of the object */
	    typeIndex() {
	        if (this.handle == 0) {
	            throw Error("The current Object has already been disposed");
	        }
	        const stack = this.lib.getOrAllocCallStack();
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        this.lib.checkCall(this.lib.exports.TVMObjectGetTypeIndex(this.getHandle(), outPtr));
	        const result = this.lib.memory.loadU32(outPtr);
	        this.lib.recycleCallStack(stack);
	        return result;
	    }
	    /** get the type key of the object */
	    typeKey() {
	        const type_index = this.typeIndex();
	        const stack = this.lib.getOrAllocCallStack();
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        this.lib.checkCall(this.lib.exports.TVMObjectTypeIndex2Key(type_index, outPtr));
	        const result = this.lib.memory.loadCString(this.lib.memory.loadPointer(outPtr));
	        this.lib.recycleCallStack(stack);
	        return result;
	    }
	}
	exports.TVMObject = TVMObject;
	/** Runtime array object. */
	class TVMArray extends TVMObject {
	    constructor(handle, lib, ctx) {
	        super(handle, lib, ctx);
	    }
	    /**
	     * @returns the size of the array.
	     */
	    size() {
	        return this.ctx.arrayGetSize(this);
	    }
	    /**
	     * Get index-th element of the array
	     * @param index the array index.
	     * @returns The element.
	     */
	    get(index) {
	        return this.ctx.arrayGetItem(this, new Scalar(index, "int32"));
	    }
	}
	exports.TVMArray = TVMArray;
	/**
	 *  VirtualMachine Executor.
	 *
	 *  This is a thin wrapper of the underlying TVM module.
	 *  you can also directly call set_input, run, and get_output
	 *  of underlying module functions
	 */
	class VirtualMachine {
	    /**
	     * Constructor
	     * @param mod The underlying module, need to be detached.
	     * @param device The main device ro run VM on.
	     */
	    constructor(mod, device) {
	        this.mod = mod;
	        this.mod.getFunction("vm_initialization")(new Scalar(device.deviceType, "int"), new Scalar(device.deviceId, "int"), new Scalar(2 /* POOLED_ALLOCATOR */, "int"), 
	        // explicitly specify host device type
	        new Scalar(DeviceStrToEnum.cpu, "int"), new Scalar(0, "int"), new Scalar(2 /* POOLED_ALLOCATOR */, "int"));
	    }
	    dispose() {
	        this.mod.dispose();
	    }
	    /**
	     * Get a function in the VM module.
	     * @param name The name of the function.
	     * @returns The result function.
	     */
	    getFunction(name) {
	        return this.mod.getFunction(name);
	    }
	    /**
	     * Get the internal module.
	     */
	    getInternalModule() {
	        return this.mod;
	    }
	}
	exports.VirtualMachine = VirtualMachine;
	/**
	 * TVM runtime instance.
	 *
	 * All objects(NDArray, Module, PackedFunc) returned by TVM runtim function call
	 * and PackedFunc instance are tracked through a scope mechanism that will get
	 * auto-released when we call EndScope.
	 *
	 * This is necessarily to be able to release the underlying WASM and WebGPU memory that
	 * are not tracked through JS native garbage collection mechanism.
	 *
	 * This does mean that we have to get familar with the following functions:
	 * - {@link beginScope}
	 * - {@link endScope}
	 * - {@link withNewScope}
	 * - {@link attachToCurrentScope}
	 * - {@link detachFromCurrentScope}
	 */
	class Instance {
	    /**
	     * Constructor
	     *
	     * importObject can also be a {@link LibraryProvider} object,
	     * a WASI object, or an object containing wasmLibraryProvider field.
	     *
	     * @param wasmModule The input module or instance.
	     * @param importObject The imports to initialize the wasmInstance if it is not provided.
	     * @param wasmInstance Additional wasm instance argument for deferred construction.
	     * @param env Directly specified environment module.
	     *
	     * @see Please use the async version {@link instantiate} when targeting browsers.
	     */
	    constructor(wasmModule, importObject = {}, wasmInstance, env) {
	        this.cacheMetadata = {};
	        this.initProgressCallback = [];
	        if (wasmInstance instanceof WebAssembly.Instance) {
	            support.assert(env instanceof environment.Environment, "env must be provided when passing in instance");
	        }
	        else {
	            support.assert(env === undefined);
	            env = new environment.Environment(importObject);
	            wasmInstance = new WebAssembly.Instance(wasmModule, env.imports);
	        }
	        env.start(wasmInstance);
	        this.env = env;
	        this.lib = new FFILibrary(wasmInstance, env.imports);
	        this.memory = this.lib.memory;
	        this.exports = this.lib.exports;
	        this.objFactory = new Map();
	        this.ctx = new RuntimeContext((name) => {
	            const autoAttachToScope = false;
	            // runtime context function do not auto-release.
	            return this.getGlobalFuncInternal(name, autoAttachToScope);
	        });
	        this.registerEnvGlobalPackedFuncs();
	        this.registerObjectFactoryFuncs();
	    }
	    /**
	     * Benchmark stable execution of the run function.
	     *
	     * @params run The run function
	     * @params dev The device to sync during each run.
	     * @number The number of times to compute the average.
	     * @repeat The number of times to repeat the run.
	     */
	    benchmark(run, dev, number = 10, repeat = 1) {
	        return __awaiter(this, void 0, void 0, function* () {
	            // Skip first run as it can involve GPU warmup and module loading time.
	            const perf = compact.getPerformance();
	            const results = [];
	            // run with new scope
	            this.withNewScope(run);
	            yield dev.sync();
	            for (let k = 0; k < repeat; ++k) {
	                const tstart = perf.now();
	                for (let i = 0; i < number; ++i) {
	                    this.withNewScope(run);
	                }
	                yield dev.sync();
	                const tend = perf.now();
	                results.push((tend - tstart) / number);
	            }
	            return results;
	        });
	    }
	    dispose() {
	        // order matters
	        // ctx release goes back into lib.
	        this.ctx.dispose();
	        this.lib.dispose();
	    }
	    /**
	     * Obtain the runtime information in readable format.
	     */
	    runtimeStatsText() {
	        if (this.lib.webGPUContext !== undefined) {
	            return this.lib.webGPUContext.runtimeStatsText();
	        }
	        else {
	            return "";
	        }
	    }
	    /**
	     * Begin a new scope for tracking object disposal.
	     */
	    beginScope() {
	        this.ctx.beginScope();
	    }
	    /**
	     * End a scope and release all created TVM objects
	     * under the current scope.
	     *
	     * Exception: one can call {@link moveToParentScope} to move
	     * a value to parent scope.
	     */
	    endScope() {
	        this.ctx.endScope();
	    }
	    /**
	     * Perform action under a new scope.
	     *
	     * @param action The action function.
	     * @returns The result value.
	     *
	     * @note For action to return a valid value,
	     *       we will need to call {@link moveToParentScope}
	     *       for the objects that are created in the scope.
	     */
	    withNewScope(action) {
	        this.beginScope();
	        const val = action();
	        this.endScope();
	        return val;
	    }
	    /**
	     * Attach a detached obj to the auto-release pool of the current scope.
	     *
	     * @param obj The input obj.
	     * @note Normally user do not need to call this function explicitly, as
	     *       all library call return values are explicitly attached to
	     *       the current scope. You only need to do so when you call
	     *       {@link detachFromCurrentScope} to create a detached object.
	     */
	    attachToCurrentScope(obj) {
	        return this.ctx.attachToCurrentScope(obj);
	    }
	    /**
	     * Move obj's attachment to the parent scope.
	     *
	     * This function is useful to make sure objects are still
	     * alive when exit the current scope.
	     *
	     * @param obj The object to be moved.
	     * @returns The input obj.
	     */
	    moveToParentScope(obj) {
	        return this.ctx.moveToParentScope(obj);
	    }
	    /**
	     * Detach the object from the current scope
	     * so it won't be released via auto-release during endscope.
	     *
	     * User needs to either explicitly call obj.dispose(), or
	     * {@link attachToCurrentScope} to re-attach to the current scope.
	     *
	     * This function can be used to return values to the parent scope.
	     * @param obj The object.
	     */
	    detachFromCurrentScope(obj) {
	        return this.ctx.detachFromCurrentScope(obj);
	    }
	    /**
	     * Get system-wide library module in the wasm.
	     * System lib is a global module that contains self register functions in startup.
	     * @returns The system library module.
	     */
	    systemLib() {
	        return this.ctx.getSysLib();
	    }
	    /**
	     * List all the global function names registered in the runtime.
	     * @returns The name list.
	     */
	    listGlobalFuncNames() {
	        const stack = this.lib.getOrAllocCallStack();
	        const outSizeOffset = stack.allocPtrArray(2);
	        const outSizePtr = stack.ptrFromOffset(outSizeOffset);
	        const outArrayPtr = stack.ptrFromOffset(outSizeOffset + this.lib.sizeofPtr());
	        this.lib.checkCall(this.exports.TVMFuncListGlobalNames(outSizePtr, outArrayPtr));
	        const size = this.memory.loadI32(outSizePtr);
	        const array = this.memory.loadPointer(outArrayPtr);
	        const names = [];
	        for (let i = 0; i < size; ++i) {
	            names.push(this.memory.loadCString(this.memory.loadPointer(array + this.lib.sizeofPtr() * i)));
	        }
	        this.lib.recycleCallStack(stack);
	        return names;
	    }
	    /**
	     * Register function to be global function in tvm runtime.
	     * @param name The name of the function.
	     * @param f function to be registered.
	     * @param override Whether overwrite function in existing registry.
	     */
	    registerFunc(name, func, override = false) {
	        this.withNewScope(() => {
	            const autoAttachToScope = true;
	            // packed func can be released once it is registered
	            const packedFunc = this.toPackedFuncInternal(func, autoAttachToScope);
	            const ioverride = override ? 1 : 0;
	            const stack = this.lib.getOrAllocCallStack();
	            const nameOffset = stack.allocRawBytes(name.length + 1);
	            stack.storeRawBytes(nameOffset, support.StringToUint8Array(name));
	            stack.commitToWasmMemory();
	            this.lib.checkCall(this.lib.exports.TVMFuncRegisterGlobal(stack.ptrFromOffset(nameOffset), packedFunc._tvmPackedCell.getHandle(), ioverride));
	            this.lib.recycleCallStack(stack);
	        });
	    }
	    /**
	     * Get global PackedFunc from the runtime.
	     * @param name The name of the function.
	     * @param autoAttachToScope Whether to track it via autoDispose
	     * @returns The result function.
	     */
	    getGlobalFunc(name) {
	        return this.getGlobalFuncInternal(name, true);
	    }
	    getGlobalFuncInternal(name, autoAttachToScope = true) {
	        const stack = this.lib.getOrAllocCallStack();
	        const nameOffset = stack.allocRawBytes(name.length + 1);
	        stack.storeRawBytes(nameOffset, support.StringToUint8Array(name));
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        stack.commitToWasmMemory(outOffset);
	        this.lib.checkCall(this.exports.TVMFuncGetGlobal(stack.ptrFromOffset(nameOffset), outPtr));
	        const handle = this.memory.loadPointer(outPtr);
	        this.lib.recycleCallStack(stack);
	        if (handle == 0) {
	            throw Error("Cannot find global function " + name);
	        }
	        const ret = this.makePackedFunc(handle);
	        if (autoAttachToScope)
	            this.ctx.attachToCurrentScope(ret);
	        return ret;
	    }
	    /**
	     * Check if func is PackedFunc.
	     *
	     * @param func The input.
	     * @returns The check result.
	     */
	    isPackedFunc(func) {
	        // eslint-disable-next-line no-prototype-builtins
	        return typeof func == "function" && func.hasOwnProperty("_tvmPackedCell");
	    }
	    /**
	     * Convert func to PackedFunc
	     *
	     * @param func Input function.
	     * @returns The converted function.
	     */
	    toPackedFunc(func) {
	        return this.toPackedFuncInternal(func, true);
	    }
	    toPackedFuncInternal(func, autoAttachToScope) {
	        if (this.isPackedFunc(func))
	            return func;
	        const ret = this.createPackedFuncFromCFunc(this.wrapJSFuncAsPackedCFunc(func));
	        if (autoAttachToScope)
	            return this.ctx.attachToCurrentScope(ret);
	        return ret;
	    }
	    /**
	    * Setup a virtual machine module with given device.
	    *
	    * @param dev DLDevice the device.
	    * @returns The created virtual machime.
	    */
	    createVirtualMachine(dev) {
	        const mod = this.ctx.detachFromCurrentScope(this.systemLib().getFunction("vm_load_executable")());
	        return this.ctx.attachToCurrentScope(new VirtualMachine(mod, dev));
	    }
	    //-----------------------------------------------
	    // Native NDArray Cache Support
	    //-----------------------------------------------
	    /**
	     * Register a call back for fetch progress.
	    *
	     * @param cb the fetch progress callback.
	     */
	    registerInitProgressCallback(cb) {
	        this.initProgressCallback.push(cb);
	    }
	    /**
	     * Get parameters in the form of prefix_i
	     *
	     * @param prefix The parameter prefix.
	     * @param numParams  Number of parameters.
	     * @returns
	     */
	    getParamsFromCache(prefix, numParams) {
	        return this.ctx.paramModuleFromCache(prefix, new Scalar(numParams, "int32")).getFunction("get_params")();
	    }
	    /**
	     * Get NDArray from cache.
	     * @param name  The name of array.
	     * @returns  The result.
	     */
	    ndarrayCacheGet(name) {
	        return this.ctx.arrayCacheGet(name);
	    }
	    /**
	     * Get NDArray from cache.
	     * @param name  The name of array.
	     * @returns  The result.
	     */
	    ndarrayCacheRemove(name) {
	        return this.ctx.arrayCacheRemove(name);
	    }
	    /**
	     * Update the ndarray cache.
	     * @param name The name of the array.
	     * @param arr The content.
	     */
	    ndarrayCacheUpdate(name, arr, override = false) {
	        this.ctx.arrayCacheUpdate(name, arr, this.scalar(override ? 1 : 0, "int32"));
	    }
	    /**
	     * Update the ndarray cache.
	     * @param name The name of the array.
	     * @param arr The content.
	     */
	    ndarrayCacheClear() {
	        this.ctx.arrayCacheClear();
	    }
	    /**
	     * Fetch NDArray cache from url.
	     *
	     * @param ndarrayCacheUrl The cache url.
	     * @param device The device to be fetched to.
	     * @returns The meta data
	     */
	    fetchNDArrayCache(ndarrayCacheUrl, device) {
	        return __awaiter(this, void 0, void 0, function* () {
	            const jsonUrl = new URL("ndarray-cache.json", ndarrayCacheUrl).href;
	            var list;
	            try {
	                list = yield (yield fetch(jsonUrl)).json();
	            }
	            catch (err) {
	                this.env.logger("Cannot fetch " + jsonUrl);
	            }
	            yield this.fetchNDArrayCacheInternal(ndarrayCacheUrl, list["records"], device);
	            this.cacheMetadata = Object.assign(Object.assign({}, this.cacheMetadata), list["metadata"]);
	        });
	    }
	    /**
	     * Fetch list of NDArray into the NDArrayCache.
	     *
	     * @param ndarrayCacheUrl The cache url.
	     * @param list The list of array data.
	     * @param device The device to store the data to.
	     */
	    fetchNDArrayCacheInternal(ndarrayCacheUrl, list, device) {
	        return __awaiter(this, void 0, void 0, function* () {
	            const perf = compact.getPerformance();
	            let tstart = perf.now();
	            let totalBytes = 0;
	            for (let i = 0; i < list.length; ++i) {
	                totalBytes += list[i].nbytes;
	            }
	            let fetchedBytes = 0;
	            let timeElapsed = 0;
	            const reportCallback = (iter) => {
	                // report
	                for (let j = 0; j < this.initProgressCallback.length; ++j) {
	                    let text = "Fetching param cache[" + iter + "/" + list.length + "]: ";
	                    text += Math.ceil(fetchedBytes / (1024 * 1024)).toString() + "MB fetched. ";
	                    text += Math.floor(fetchedBytes * 100 / totalBytes).toString() + "% completed, ";
	                    text += timeElapsed + " secs elapsed.";
	                    text += " It can take a while when we first visit this page to populate the cache.";
	                    text += " Later refreshes will become faster.";
	                    this.initProgressCallback[j]({
	                        progress: fetchedBytes / totalBytes,
	                        timeElapsed: timeElapsed,
	                        text: text
	                    });
	                }
	            };
	            for (let j = 0; j < this.initProgressCallback.length; ++j) {
	                this.initProgressCallback[j]({
	                    progress: fetchedBytes / totalBytes,
	                    timeElapsed: 0,
	                    text: "Start to fetch params",
	                });
	            }
	            const cache = yield caches.open("tvmjs");
	            for (let i = 0; i < list.length; ++i) {
	                reportCallback(i);
	                fetchedBytes += list[i].nbytes;
	                const dataUrl = new URL(list[i].dataPath, ndarrayCacheUrl).href;
	                const request = new Request(dataUrl);
	                let buffer;
	                try {
	                    // use native cache
	                    let result = yield cache.match(request);
	                    if (result === undefined) {
	                        yield cache.add(request);
	                        result = yield cache.match(request);
	                    }
	                    if (result == undefined) {
	                        this.env.logger("Error: Cannot cache " + dataUrl + ", reloading will be slow");
	                        result = yield fetch(request);
	                    }
	                    buffer = yield result.arrayBuffer();
	                }
	                catch (err) {
	                    this.env.logger("Error: Cannot fetch " + dataUrl + " err= " + err);
	                    throw err;
	                }
	                const shardRecords = list[i].records;
	                for (let j = 0; j < shardRecords.length; ++j) {
	                    const rec = shardRecords[j];
	                    const cpu_arr = this.withNewScope(() => {
	                        return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, this.cpu()));
	                    });
	                    const recSource = buffer.slice(rec.byteOffset, rec.byteOffset + rec.nbytes);
	                    // first sync copy to cpu.
	                    this.ctx.arrayDecodeStorage(cpu_arr, new Uint8Array(recSource), rec.format);
	                    // then async stream into GPU if needed
	                    if (device.deviceType == DeviceStrToEnum.cpu) {
	                        this.ndarrayCacheUpdate(rec.name, cpu_arr, false);
	                        cpu_arr.dispose();
	                    }
	                    else {
	                        // allocate a gpu arr and async copy to it.
	                        const gpu_arr = this.withNewScope(() => {
	                            return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, device));
	                        });
	                        gpu_arr.copyFrom(cpu_arr);
	                        yield device.sync();
	                        this.ndarrayCacheUpdate(rec.name, gpu_arr, false);
	                        cpu_arr.dispose();
	                        gpu_arr.dispose();
	                    }
	                }
	                timeElapsed = Math.ceil((perf.now() - tstart) / 1000);
	            }
	            reportCallback(list.length);
	        });
	    }
	    /**
	     * Convert dtype to {@link DLDataType}
	     *
	     * @param dtype The input dtype string or DLDataType.
	     * @returns The converted result.
	     */
	    toDLDataType(dtype) {
	        if (dtype instanceof DLDataType)
	            return dtype;
	        if (typeof dtype == "string") {
	            let pattern = dtype;
	            let code, bits = 32, lanes = 1;
	            if (pattern.substring(0, 5) == "float") {
	                pattern = pattern.substring(5, pattern.length);
	                code = 2 /* Float */;
	            }
	            else if (pattern.substring(0, 3) == "int") {
	                pattern = pattern.substring(3, pattern.length);
	                code = 0 /* Int */;
	            }
	            else if (pattern.substring(0, 4) == "uint") {
	                pattern = pattern.substring(4, pattern.length);
	                code = 1 /* UInt */;
	            }
	            else if (pattern.substring(0, 6) == "handle") {
	                pattern = pattern.substring(5, pattern.length);
	                code = 3 /* OpaqueHandle */;
	                bits = 64;
	            }
	            else {
	                throw new Error("Unknown dtype " + dtype);
	            }
	            const arr = pattern.split("x");
	            if (arr.length >= 1) {
	                const parsed = parseInt(arr[0]);
	                if (parsed + "" == arr[0]) {
	                    bits = parsed;
	                }
	            }
	            if (arr.length >= 2) {
	                lanes = parseInt(arr[1]);
	            }
	            return new DLDataType(code, bits, lanes);
	        }
	        else {
	            throw new Error("Unknown dtype " + dtype);
	        }
	    }
	    /**
	     * Create a new {@link Scalar} that can be passed to a PackedFunc.
	     * @param value The number value.
	     * @param dtype The dtype string.
	     * @returns The created scalar.
	     */
	    scalar(value, dtype) {
	        return new Scalar(value, dtype);
	    }
	    /**
	     * Create a new {@link DLDevice}
	     * @param deviceType The device type.
	     * @param deviceId The device index.
	     * @returns The created device.
	     */
	    device(deviceType, deviceId = 0) {
	        return new DLDevice(deviceType, deviceId, this.lib);
	    }
	    /**
	     * Create a new cpu {@link DLDevice}
	     * @param deviceId The device index.
	     */
	    cpu(deviceId = 0) {
	        return this.device("cpu", deviceId);
	    }
	    /**
	     * Create a new webgpu {@link DLDevice}
	     * @param deviceId The device index.
	     */
	    webgpu(deviceId = 0) {
	        return this.device("webgpu", deviceId);
	    }
	    /**
	     * Create an empty {@link NDArray} with given shape and dtype.
	     *
	     * @param shape The shape of the array.
	     * @param dtype The data type of the array.
	     * @param dev The device of the ndarray.
	     * @returns The created ndarray.
	     */
	    empty(shape, dtype = "float32", dev = this.device("cpu", 0)) {
	        dtype = this.toDLDataType(dtype);
	        shape = typeof shape == "number" ? [shape] : shape;
	        const stack = this.lib.getOrAllocCallStack();
	        const shapeOffset = stack.allocRawBytes(shape.length * 8 /* I64 */);
	        for (let i = 0; i < shape.length; ++i) {
	            stack.storeI64(shapeOffset + i * 8 /* I64 */, shape[i]);
	        }
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        stack.commitToWasmMemory(outOffset);
	        this.lib.checkCall(this.exports.TVMArrayAlloc(stack.ptrFromOffset(shapeOffset), shape.length, dtype.code, dtype.bits, dtype.lanes, dev.deviceType, dev.deviceId, outPtr));
	        const ret = this.ctx.attachToCurrentScope(new NDArray(this.memory.loadPointer(outPtr), false, this.lib, this.ctx));
	        this.lib.recycleCallStack(stack);
	        return ret;
	    }
	    /**
	     * Create am uniform {@link NDArray} with given shape.
	     *
	     * @param shape The shape of the array.
	     * @param low The low value.
	     * @param high The high value.
	     * @param dev The device of the ndarray.
	     * @returns The created ndarray.
	     */
	    uniform(shape, low, high, dev) {
	        const ret = this.empty(shape, "float32", dev);
	        const size = shape.reduce((a, b) => {
	            return a * b;
	        }, 1);
	        const scale = high - low;
	        const input = new Float32Array(size);
	        for (let i = 0; i < input.length; ++i) {
	            input[i] = low + Math.random() * scale;
	        }
	        return ret.copyFrom(input);
	    }
	    /**
	     * Sample index via top-p sampling.
	     *
	     * @param logits The input logits before normalization.
	     * @param temperature  The temperature factor, will take argmax if temperature = 0.0
	     * @param top_p The top_p
	     * @returns The sampled index.
	     */
	    sampleTopPFromLogits(logits, temperature, top_p) {
	        return this.ctx.sampleTopPFromLogits(logits, temperature, top_p, Math.random());
	    }
	    /**
	     * Bind canvas to the current WebGPU context
	     * @param canvas The canvas.
	     */
	    bindCanvas(canvas) {
	        var _a;
	        (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.bindCanvas(canvas);
	    }
	    /**
	     * Show image in canvas.
	     *
	     * @param dataRGBA Image array in height x width uint32 NDArray RGBA format on GPU.
	     */
	    showImage(dataRGBA) {
	        var _a;
	        if (dataRGBA.shape.length != 2) {
	            throw Error("Require a height x width uint32 NDArray in RGBA" +
	                "get shape=" + dataRGBA.shape.toString() + " instead.");
	        }
	        if (dataRGBA.device.deviceType != DeviceStrToEnum.webgpu) {
	            throw new Error("Can only run showImage on WebGPU array, " +
	                "get " + DeviceEnumToStr[dataRGBA.device.deviceType] + " instead.");
	        }
	        if (dataRGBA.dtype != "uint32") {
	            throw Error("Require a height x width uint32 NDArray in RGBA, " +
	                "get " + dataRGBA.dtype + " instead.");
	        }
	        (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.drawImageFromBuffer(dataRGBA.getDataPtr(), dataRGBA.shape[0], dataRGBA.shape[1]);
	    }
	    /**
	     * Clear canvas
	     */
	    clearCanvas() {
	        var _a;
	        (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.clearCanvas();
	    }
	    /**
	     * Create an tuple {@link TVMArray} input array.
	     *
	     * The input array can be passed to tvm runtime function
	     * and needs to b explicitly disposed.
	     *
	     * @param inputs The input array
	     * @returns The result array.
	     */
	    makeTVMArray(inputs) {
	        return this.ctx.arrayMake(...inputs);
	    }
	    /**
	     * Create a shape tuple to pass to runtime.
	     * @param shape The shape .
	     * @returns The created shape tuple.
	     */
	    makeShapeTuple(shape) {
	        const shapeArray = shape.map((value) => new Scalar(value, "int"));
	        return this.ctx.makeShapeTuple(...shapeArray);
	    }
	    /**
	     * Get type index from type key.
	     * @param typeKey The type key.
	     * @returns The corresponding type index.
	     */
	    typeKey2Index(typeKey) {
	        const stack = this.lib.getOrAllocCallStack();
	        const typeKeyOffset = stack.allocRawBytes(typeKey.length + 1);
	        stack.storeRawBytes(typeKeyOffset, support.StringToUint8Array(typeKey));
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        stack.commitToWasmMemory(outOffset);
	        this.lib.checkCall(this.lib.exports.TVMObjectTypeKey2Index(stack.ptrFromOffset(typeKeyOffset), outPtr));
	        const typeIndex = this.memory.loadU32(outPtr);
	        this.lib.recycleCallStack(stack);
	        return typeIndex;
	    }
	    /**
	     * Register an object constructor.
	     * @param typeKey The name of the function.
	     * @param func function to be registered.
	     * @param override Whether overwrite function in existing registry.
	     */
	    registerObjectConstructor(typeKey, func, override = false) {
	        const typeIndex = this.typeKey2Index(typeKey);
	        if (this.objFactory.has(typeIndex)) {
	            if (!override) {
	                throw new Error("Type " + typeKey + " already registered");
	            }
	        }
	        this.objFactory.set(typeIndex, func);
	    }
	    /**
	     * Register an asyncfunction to be global function in the server.
	     * @param name The name of the function.
	     * @param func function to be registered.
	     * @param override Whether overwrite function in existing registry.
	     *
	     * @note The async function will only be used for serving remote calls in the rpc.
	     */
	    registerAsyncServerFunc(name, func, override = false) {
	        const asyncVariant = (...args) => {
	            const fargs = args.slice(0, args.length - 1);
	            // need to keep it alive until callback is fulfilled.
	            const callback = this.detachFromCurrentScope(args[args.length - 1]);
	            const promise = func(...fargs);
	            promise.then((rv) => {
	                callback(this.scalar(4 /* kReturn */, "int32"), rv);
	                callback.dispose();
	            });
	        };
	        this.registerFunc("__async." + name, asyncVariant, override);
	    }
	    /**
	     * Asynchrously load webgpu pipelines when possible.
	     * @param mod The input module.
	     */
	    asyncLoadWebGPUPiplines(mod) {
	        return __awaiter(this, void 0, void 0, function* () {
	            if (this.lib.webGPUContext == undefined)
	                throw Error("WebGPU not initialied");
	            const webgpuContext = this.lib.webGPUContext;
	            this.beginScope();
	            const fmap_str = mod.getFunction("webgpu.get_fmap", true)();
	            let fmap = JSON.parse(fmap_str);
	            fmap.length;
	            const fGetShader = this.detachFromCurrentScope(mod.getFunction("webgpu.get_shader"));
	            const fUpdatePrebuild = this.detachFromCurrentScope(mod.getFunction("webgpu.update_prebuild"));
	            this.endScope();
	            const perf = compact.getPerformance();
	            const tstart = perf.now();
	            let tlastReport = tstart;
	            let finishCounter = 0;
	            const fmapEntries = Object.entries(fmap);
	            let allEvents = Promise.resolve();
	            for (const [key, finfo] of fmapEntries) {
	                const code = fGetShader(key);
	                support.assert(key == finfo.name);
	                const event = webgpuContext.createShaderAsync(finfo, code).then((func) => {
	                    this.beginScope();
	                    fUpdatePrebuild(key, func);
	                    this.endScope();
	                }).then(() => {
	                    finishCounter += 1;
	                    const tend = perf.now();
	                    // skip report if gap is smaller than 1000
	                    if ((tend - tlastReport) < 1000 && finishCounter != fmapEntries.length) {
	                        return;
	                    }
	                    tlastReport = tend;
	                    const timeElapsed = Math.ceil((perf.now() - tstart) / 1000);
	                    // report
	                    for (let j = 0; j < this.initProgressCallback.length; ++j) {
	                        const progress = finishCounter / fmapEntries.length;
	                        let text = "Loading GPU shader modules[" + finishCounter + "/" + fmapEntries.length + "]: ";
	                        text += Math.floor(progress * 100).toString() + "% completed, ";
	                        text += timeElapsed + " secs elapsed.";
	                        this.initProgressCallback[j]({
	                            progress: progress,
	                            timeElapsed: timeElapsed,
	                            text: text
	                        });
	                    }
	                });
	                allEvents = Promise.all([allEvents, event]).then(() => { });
	            }
	            yield allEvents;
	            support.assert(finishCounter == fmapEntries.length);
	        });
	    }
	    /**
	     * Initialize webgpu in the runtime.
	     * @param device The given GPU device.
	     */
	    initWebGPU(device) {
	        const webGPUContext = new webgpu.WebGPUContext(this.memory, device);
	        this.registerFunc("wasm.WebGPUDeviceAPI", (name) => {
	            return webGPUContext.getDeviceAPI(name);
	        });
	        this.registerFunc("wasm.WebGPUCreateShader", (info, code) => {
	            const finfo = JSON.parse(info);
	            return webGPUContext.createShader(finfo, code);
	        });
	        this.registerAsyncServerFunc("wasm.WebGPUWaitForTasks", () => __awaiter(this, void 0, void 0, function* () {
	            yield webGPUContext.sync();
	        }));
	        this.lib.webGPUContext = webGPUContext;
	    }
	    /** Register all object factory */
	    registerObjectFactoryFuncs() {
	        this.registerObjectConstructor("Array", (handle, lib, ctx) => {
	            return new TVMArray(handle, lib, ctx);
	        });
	    }
	    /** Register global packed functions needed by the backend to the env. */
	    registerEnvGlobalPackedFuncs() {
	        // Register the timer function to enable the time_evaluator.
	        const perf = compact.getPerformance();
	        // Helper function to time the finvoke
	        const timeExecution = (finvoke, dev, nstep, repeat, minRepeatMs, limitZeroTimeIterations, cooldownIntervalMs, repeatsToCooldown) => __awaiter(this, void 0, void 0, function* () {
	            // detach and explicit dispose when tasks is fullfilled
	            // the promise will immediately return and we need to makesure
	            // finvoke do not get recycled.
	            this.ctx.detachFromCurrentScope(finvoke);
	            finvoke(this.scalar(1, "int32"));
	            yield dev.sync();
	            const result = [];
	            let setupNumber = nstep;
	            for (let i = 0; i < repeat; ++i) {
	                let durationMs = 0.0;
	                let absoluteZeroTimes = 0;
	                do {
	                    if (durationMs > 0.0) {
	                        let golden_ratio = 1.618;
	                        setupNumber = Math.floor(Math.max(minRepeatMs / (durationMs / setupNumber) + 1, setupNumber * golden_ratio));
	                    }
	                    const tstart = perf.now();
	                    finvoke(this.scalar(setupNumber, "int32"));
	                    yield dev.sync();
	                    const tend = perf.now();
	                    durationMs = tend - tstart;
	                    if (durationMs == 0) {
	                        absoluteZeroTimes++;
	                    }
	                } while (durationMs < minRepeatMs && absoluteZeroTimes < limitZeroTimeIterations);
	                const speed = durationMs / setupNumber / 1000;
	                result.push(speed);
	                if (cooldownIntervalMs > 0.0 && (i % repeatsToCooldown) == 0) {
	                    yield new Promise(r => setTimeout(r, cooldownIntervalMs));
	                }
	            }
	            const ret = new Float64Array(result.length);
	            ret.set(result);
	            // dispose finvoke
	            finvoke.dispose();
	            return new Uint8Array(ret.buffer);
	        });
	        const addOne = (x) => __awaiter(this, void 0, void 0, function* () {
	            yield new Promise(resolve => setTimeout(resolve, 100));
	            return x + 1;
	        });
	        this.registerAsyncServerFunc("wasm.TimeExecution", timeExecution);
	        this.registerAsyncServerFunc("testing.asyncAddOne", addOne);
	    }
	    createPackedFuncFromCFunc(func) {
	        let findex = this.env.packedCFuncTable.length;
	        if (this.env.packedCFuncTableFreeId.length != 0) {
	            findex = this.env.packedCFuncTableFreeId.pop();
	        }
	        else {
	            this.env.packedCFuncTable.push(undefined);
	        }
	        this.env.packedCFuncTable[findex] = func;
	        const stack = this.lib.getOrAllocCallStack();
	        const outOffset = stack.allocPtrArray(1);
	        const outPtr = stack.ptrFromOffset(outOffset);
	        this.lib.checkCall(this.exports
	            .TVMWasmFuncCreateFromCFunc(findex, outPtr));
	        const ret = this.makePackedFunc(this.memory.loadPointer(outPtr));
	        this.lib.recycleCallStack(stack);
	        return ret;
	    }
	    /**
	     * Set packed function arguments into the location indicated by argsValue and argsCode.
	     * Allocate new temporary space from the stack if necessary.
	     *
	     * @parma stack The call stack
	     * @param args  The input arguments.
	     * @param argsValue The offset of argsValue.
	     * @param argsCode The offset of argsCode.
	     */
	    setPackedArguments(stack, args, argsValue, argsCode) {
	        for (let i = 0; i < args.length; ++i) {
	            let val = args[i];
	            const tp = typeof val;
	            const valueOffset = argsValue + i * 8 /* TVMValue */;
	            const codeOffset = argsCode + i * 4 /* I32 */;
	            if (val instanceof NDArray) {
	                if (!val.isView) {
	                    stack.storePtr(valueOffset, val.getHandle());
	                    stack.storeI32(codeOffset, 13 /* TVMNDArrayHandle */);
	                }
	                else {
	                    stack.storePtr(valueOffset, val.getHandle());
	                    stack.storeI32(codeOffset, 7 /* TVMDLTensorHandle */);
	                }
	            }
	            else if (val instanceof Scalar) {
	                if (val.dtype.startsWith("int") || val.dtype.startsWith("uint")) {
	                    stack.storeI64(valueOffset, val.value);
	                    stack.storeI32(codeOffset, 0 /* Int */);
	                }
	                else if (val.dtype.startsWith("float")) {
	                    stack.storeF64(valueOffset, val.value);
	                    stack.storeI32(codeOffset, 2 /* Float */);
	                }
	                else {
	                    support.assert(val.dtype == "handle", "Expect handle");
	                    stack.storePtr(valueOffset, val.value);
	                    stack.storeI32(codeOffset, 3 /* TVMOpaqueHandle */);
	                }
	            }
	            else if (val instanceof DLDevice) {
	                stack.storeI32(valueOffset, val.deviceType);
	                stack.storeI32(valueOffset + 4 /* I32 */, val.deviceType);
	                stack.storeI32(codeOffset, 6 /* DLDevice */);
	            }
	            else if (tp == "number") {
	                stack.storeF64(valueOffset, val);
	                stack.storeI32(codeOffset, 2 /* Float */);
	                // eslint-disable-next-line no-prototype-builtins
	            }
	            else if (tp == "function" && val.hasOwnProperty("_tvmPackedCell")) {
	                stack.storePtr(valueOffset, val._tvmPackedCell.getHandle());
	                stack.storeI32(codeOffset, 10 /* TVMPackedFuncHandle */);
	            }
	            else if (val === null || val == undefined) {
	                stack.storePtr(valueOffset, 0);
	                stack.storeI32(codeOffset, 4 /* Null */);
	            }
	            else if (tp == "string") {
	                stack.allocThenSetArgString(valueOffset, val);
	                stack.storeI32(codeOffset, 11 /* TVMStr */);
	            }
	            else if (val instanceof Uint8Array) {
	                stack.allocThenSetArgBytes(valueOffset, val);
	                stack.storeI32(codeOffset, 12 /* TVMBytes */);
	            }
	            else if (val instanceof Function) {
	                val = this.toPackedFuncInternal(val, false);
	                stack.tempArgs.push(val);
	                stack.storePtr(valueOffset, val._tvmPackedCell.getHandle());
	                stack.storeI32(codeOffset, 10 /* TVMPackedFuncHandle */);
	            }
	            else if (val instanceof Module) {
	                stack.storePtr(valueOffset, val.getHandle());
	                stack.storeI32(codeOffset, 9 /* TVMModuleHandle */);
	            }
	            else if (val instanceof TVMObject) {
	                stack.storePtr(valueOffset, val.getHandle());
	                stack.storeI32(codeOffset, 8 /* TVMObjectHandle */);
	            }
	            else {
	                throw new Error("Unsupported argument type " + tp);
	            }
	        }
	    }
	    wrapJSFuncAsPackedCFunc(func) {
	        const lib = this.lib;
	        return (argValues, argCodes, nargs, ret, 
	        // eslint-disable-next-line @typescript-eslint/no-unused-vars
	        _handle) => {
	            const jsArgs = [];
	            // use scope to track js values.
	            this.ctx.beginScope();
	            for (let i = 0; i < nargs; ++i) {
	                const valuePtr = argValues + i * 8 /* TVMValue */;
	                const codePtr = argCodes + i * 4 /* I32 */;
	                let tcode = lib.memory.loadI32(codePtr);
	                if (tcode == 8 /* TVMObjectHandle */ ||
	                    tcode == 14 /* TVMObjectRValueRefArg */ ||
	                    tcode == 10 /* TVMPackedFuncHandle */ ||
	                    tcode == 13 /* TVMNDArrayHandle */ ||
	                    tcode == 9 /* TVMModuleHandle */) {
	                    lib.checkCall(lib.exports.TVMCbArgToReturn(valuePtr, codePtr));
	                }
	                tcode = lib.memory.loadI32(codePtr);
	                jsArgs.push(this.retValueToJS(valuePtr, tcode, true));
	            }
	            const rv = func(...jsArgs);
	            // recycle all js object value in function unless we want to retain them.
	            this.ctx.endScope();
	            if (rv !== undefined && rv !== null) {
	                const stack = lib.getOrAllocCallStack();
	                const valueOffset = stack.allocRawBytes(8 /* TVMValue */);
	                const codeOffset = stack.allocRawBytes(4 /* I32 */);
	                this.setPackedArguments(stack, [rv], valueOffset, codeOffset);
	                const valuePtr = stack.ptrFromOffset(valueOffset);
	                const codePtr = stack.ptrFromOffset(codeOffset);
	                stack.commitToWasmMemory();
	                lib.checkCall(lib.exports.TVMCFuncSetReturn(ret, valuePtr, codePtr, 1));
	                lib.recycleCallStack(stack);
	            }
	            return 0;
	        };
	    }
	    makePackedFunc(handle) {
	        const cell = new PackedFuncCell(handle, this.lib);
	        const packedFunc = (...args) => {
	            const stack = this.lib.getOrAllocCallStack();
	            const valueOffset = stack.allocRawBytes(8 /* TVMValue */ * args.length);
	            const tcodeOffset = stack.allocRawBytes(4 /* I32 */ * args.length);
	            this.setPackedArguments(stack, args, valueOffset, tcodeOffset);
	            const rvalueOffset = stack.allocRawBytes(8 /* TVMValue */);
	            const rcodeOffset = stack.allocRawBytes(4 /* I32 */);
	            const rvaluePtr = stack.ptrFromOffset(rvalueOffset);
	            const rcodePtr = stack.ptrFromOffset(rcodeOffset);
	            // commit to wasm memory, till rvalueOffset (the return value don't need to be committed)
	            stack.commitToWasmMemory(rvalueOffset);
	            this.lib.checkCall(this.exports.TVMFuncCall(cell.getHandle(), stack.ptrFromOffset(valueOffset), stack.ptrFromOffset(tcodeOffset), args.length, rvaluePtr, rcodePtr));
	            const ret = this.retValueToJS(rvaluePtr, this.memory.loadI32(rcodePtr), false);
	            this.lib.recycleCallStack(stack);
	            return ret;
	        };
	        // Attach attributes to the function type.
	        // This is because javascript do not allow us to overload call.
	        const ret = packedFunc;
	        ret.dispose = () => {
	            cell.dispose();
	        };
	        ret._tvmPackedCell = cell;
	        return ret;
	    }
	    /**
	     * Creaye return value of the packed func. The value us auto-tracked for dispose.
	     * @param rvaluePtr The location of rvalue
	     * @param tcode     The type code.
	     * @param callbackArg Whether it is being used in callbackArg.
	     * @returns The JS value.
	     */
	    retValueToJS(rvaluePtr, tcode, callbackArg) {
	        switch (tcode) {
	            case 0 /* Int */:
	            case 1 /* UInt */:
	                return this.memory.loadI64(rvaluePtr);
	            case 2 /* Float */:
	                return this.memory.loadF64(rvaluePtr);
	            case 3 /* TVMOpaqueHandle */: {
	                return this.memory.loadPointer(rvaluePtr);
	            }
	            case 13 /* TVMNDArrayHandle */: {
	                return this.ctx.attachToCurrentScope(new NDArray(this.memory.loadPointer(rvaluePtr), false, this.lib, this.ctx));
	            }
	            case 7 /* TVMDLTensorHandle */: {
	                support.assert(callbackArg);
	                // no need to attach as we are only looking at view
	                return new NDArray(this.memory.loadPointer(rvaluePtr), true, this.lib, this.ctx);
	            }
	            case 10 /* TVMPackedFuncHandle */: {
	                return this.ctx.attachToCurrentScope(this.makePackedFunc(this.memory.loadPointer(rvaluePtr)));
	            }
	            case 9 /* TVMModuleHandle */: {
	                return this.ctx.attachToCurrentScope(new Module(this.memory.loadPointer(rvaluePtr), this.lib, (ptr) => {
	                    return this.ctx.attachToCurrentScope(this.makePackedFunc(ptr));
	                }));
	            }
	            case 8 /* TVMObjectHandle */: {
	                const obj = new TVMObject(this.memory.loadPointer(rvaluePtr), this.lib, this.ctx);
	                const func = this.objFactory.get(obj.typeIndex());
	                if (func != undefined) {
	                    return this.ctx.attachToCurrentScope(func(obj.getHandle(), this.lib, this.ctx));
	                }
	                else {
	                    return this.ctx.attachToCurrentScope(obj);
	                }
	            }
	            case 4 /* Null */: return undefined;
	            case 6 /* DLDevice */: {
	                const deviceType = this.memory.loadI32(rvaluePtr);
	                const deviceId = this.memory.loadI32(rvaluePtr + 4 /* I32 */);
	                return this.device(deviceType, deviceId);
	            }
	            case 11 /* TVMStr */: {
	                const ret = this.memory.loadCString(this.memory.loadPointer(rvaluePtr));
	                return ret;
	            }
	            case 12 /* TVMBytes */: {
	                return this.memory.loadTVMBytes(this.memory.loadPointer(rvaluePtr));
	            }
	            default:
	                throw new Error("Unsupported return type code=" + tcode);
	        }
	    }
	}
	exports.Instance = Instance;
	/**
	 * Asynchrously instantiate a new {@link Instance}.
	 *
	 * importObject can also be a {@link LibraryProvider} object,
	 * a WASI object, or an object containing wasmLibraryProvider field.
	 * We can take benefit of syslib implementations from the Emscripten
	 * by passing its generated js Module as the imports.
	 *
	 * @param bufferSource The source to be compiled.
	 * @param importObject The import objects.
	 * @param logger The system logger.
	 */
	function instantiate(bufferSource, importObject = {}, logger = console.log) {
	    const env = new environment.Environment(importObject, logger);
	    return WebAssembly.instantiate(bufferSource, env.imports).then((result) => {
	        return new Instance(result.module, {}, result.instance, env);
	    });
	}
	exports.instantiate = instantiate;

	});

	unwrapExports(runtime);
	runtime.instantiate;
	runtime.Instance;
	runtime.VirtualMachine;
	runtime.TVMArray;
	runtime.TVMObject;
	runtime.Module;
	runtime.NDArray;
	runtime.DLDataType;
	runtime.DLDevice;
	runtime.Scalar;

	var rpc_server = createCommonjsModule(function (module, exports) {
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
	var __awaiter = (commonjsGlobal && commonjsGlobal.__awaiter) || function (thisArg, _arguments, P, generator) {
	    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
	    return new (P || (P = Promise))(function (resolve, reject) {
	        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
	        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
	        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
	        step((generator = generator.apply(thisArg, _arguments || [])).next());
	    });
	};
	Object.defineProperty(exports, "__esModule", { value: true });
	exports.RPCServer = void 0;




	var RPCServerState;
	(function (RPCServerState) {
	    RPCServerState[RPCServerState["InitHeader"] = 0] = "InitHeader";
	    RPCServerState[RPCServerState["InitHeaderKey"] = 1] = "InitHeaderKey";
	    RPCServerState[RPCServerState["InitServer"] = 2] = "InitServer";
	    RPCServerState[RPCServerState["WaitForCallback"] = 3] = "WaitForCallback";
	    RPCServerState[RPCServerState["ReceivePacketHeader"] = 4] = "ReceivePacketHeader";
	    RPCServerState[RPCServerState["ReceivePacketBody"] = 5] = "ReceivePacketBody";
	})(RPCServerState || (RPCServerState = {}));
	/** RPC magic header */
	const RPC_MAGIC = 0xff271;
	/**
	 * An utility class to read from binary bytes.
	 */
	class ByteStreamReader {
	    constructor(bytes) {
	        this.offset = 0;
	        this.bytes = bytes;
	    }
	    readU32() {
	        const i = this.offset;
	        const b = this.bytes;
	        const val = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
	        this.offset += 4;
	        return val;
	    }
	    readU64() {
	        const val = this.readU32();
	        this.offset += 4;
	        return val;
	    }
	    readByteArray() {
	        const len = this.readU64();
	        support.assert(this.offset + len <= this.bytes.byteLength);
	        const ret = new Uint8Array(len);
	        ret.set(this.bytes.slice(this.offset, this.offset + len));
	        this.offset += len;
	        return ret;
	    }
	}
	/**
	 * A websocket based RPC
	 */
	class RPCServer {
	    constructor(url, key, getImports, logger = console.log, ndarrayCacheUrl = "", ndarrayCacheDevice = "cpu", initProgressCallback = undefined, asyncOnServerLoad = undefined) {
	        this.state = RPCServerState.InitHeader;
	        this.pendingSend = Promise.resolve();
	        this.inst = undefined;
	        this.globalObjects = [];
	        this.currPacketLength = 0;
	        this.remoteKeyLength = 0;
	        this.pendingBytes = 0;
	        this.buffredBytes = 0;
	        this.messageQueue = [];
	        this.url = url;
	        this.key = key;
	        this.name = "WebSocketRPCServer[" + this.key + "]: ";
	        this.getImports = getImports;
	        this.logger = logger;
	        this.ndarrayCacheUrl = ndarrayCacheUrl;
	        this.ndarrayCacheDevice = ndarrayCacheDevice;
	        this.initProgressCallback = initProgressCallback;
	        this.asyncOnServerLoad = asyncOnServerLoad;
	        this.checkLittleEndian();
	        this.socket = compact.createWebSocket(url);
	        this.socket.binaryType = "arraybuffer";
	        this.socket.addEventListener("open", (event) => {
	            return this.onOpen(event);
	        });
	        this.socket.addEventListener("message", (event) => {
	            return this.onMessage(event);
	        });
	        this.socket.addEventListener("close", (event) => {
	            return this.onClose(event);
	        });
	    }
	    // eslint-disable-next-line @typescript-eslint/no-unused-vars
	    onClose(_event) {
	        if (this.inst !== undefined) {
	            this.globalObjects.forEach(obj => {
	                obj.dispose();
	            });
	            this.log(this.inst.runtimeStatsText());
	            this.inst.dispose();
	        }
	        if (this.state == RPCServerState.ReceivePacketHeader) {
	            this.log("Closing the server in clean state");
	            this.log("Automatic reconnecting..");
	            new RPCServer(this.url, this.key, this.getImports, this.logger, this.ndarrayCacheUrl, this.ndarrayCacheDevice, this.initProgressCallback, this.asyncOnServerLoad);
	        }
	        else {
	            this.log("Closing the server, final state=" + this.state);
	        }
	    }
	    // eslint-disable-next-line @typescript-eslint/no-unused-vars
	    onOpen(_event) {
	        // Send the headers
	        let bkey = support.StringToUint8Array("server:" + this.key);
	        bkey = bkey.slice(0, bkey.length - 1);
	        const intbuf = new Int32Array(1);
	        intbuf[0] = RPC_MAGIC;
	        this.socket.send(intbuf);
	        intbuf[0] = bkey.length;
	        this.socket.send(intbuf);
	        this.socket.send(bkey);
	        this.log("connected...");
	        // request bytes: magic + keylen
	        this.requestBytes(4 /* I32 */ + 4 /* I32 */);
	        this.state = RPCServerState.InitHeader;
	    }
	    /** Handler for raw message. */
	    onMessage(event) {
	        const buffer = event.data;
	        this.buffredBytes += buffer.byteLength;
	        this.messageQueue.push(new Uint8Array(buffer));
	        this.processEvents();
	    }
	    /** Process ready events. */
	    processEvents() {
	        while (this.buffredBytes >= this.pendingBytes && this.pendingBytes != 0) {
	            this.onDataReady();
	        }
	    }
	    /** State machine to handle each request */
	    onDataReady() {
	        switch (this.state) {
	            case RPCServerState.InitHeader: {
	                this.handleInitHeader();
	                break;
	            }
	            case RPCServerState.InitHeaderKey: {
	                this.handleInitHeaderKey();
	                break;
	            }
	            case RPCServerState.ReceivePacketHeader: {
	                this.currPacketHeader = this.readFromBuffer(8 /* I64 */);
	                const reader = new ByteStreamReader(this.currPacketHeader);
	                this.currPacketLength = reader.readU64();
	                support.assert(this.pendingBytes == 0);
	                this.requestBytes(this.currPacketLength);
	                this.state = RPCServerState.ReceivePacketBody;
	                break;
	            }
	            case RPCServerState.ReceivePacketBody: {
	                const body = this.readFromBuffer(this.currPacketLength);
	                support.assert(this.pendingBytes == 0);
	                support.assert(this.currPacketHeader !== undefined);
	                this.onPacketReady(this.currPacketHeader, body);
	                break;
	            }
	            case RPCServerState.WaitForCallback: {
	                support.assert(this.pendingBytes == 0);
	                break;
	            }
	            default: {
	                throw new Error("Cannot handle state " + this.state);
	            }
	        }
	    }
	    onPacketReady(header, body) {
	        if (this.inst === undefined) {
	            // initialize server.
	            const reader = new ByteStreamReader(body);
	            // eslint-disable-next-line @typescript-eslint/no-unused-vars
	            reader.readU32();
	            // eslint-disable-next-line @typescript-eslint/no-unused-vars
	            support.Uint8ArrayToString(reader.readByteArray());
	            const nargs = reader.readU32();
	            const tcodes = [];
	            const args = [];
	            for (let i = 0; i < nargs; ++i) {
	                tcodes.push(reader.readU32());
	            }
	            for (let i = 0; i < nargs; ++i) {
	                const tcode = tcodes[i];
	                if (tcode == 11 /* TVMStr */) {
	                    const str = support.Uint8ArrayToString(reader.readByteArray());
	                    args.push(str);
	                }
	                else if (tcode == 12 /* TVMBytes */) {
	                    args.push(reader.readByteArray());
	                }
	                else {
	                    throw new Error("cannot support type code " + tcode);
	                }
	            }
	            this.onInitServer(args, header, body);
	        }
	        else {
	            support.assert(this.serverRecvData !== undefined);
	            this.serverRecvData(header, body);
	            this.requestBytes(8 /* I64 */);
	            this.state = RPCServerState.ReceivePacketHeader;
	        }
	    }
	    /** Event handler during server initialization. */
	    onInitServer(args, header, body) {
	        // start the server
	        support.assert(args[0] == "rpc.WasmSession");
	        support.assert(this.pendingBytes == 0);
	        const asyncInitServer = () => __awaiter(this, void 0, void 0, function* () {
	            support.assert(args[1] instanceof Uint8Array);
	            const inst = yield runtime.instantiate(args[1].buffer, this.getImports(), this.logger);
	            try {
	                const output = yield webgpu.detectGPUDevice();
	                if (output !== undefined) {
	                    const label = "WebGPU: " + output.adapterInfo.description;
	                    this.log("Initialize GPU device: " + label);
	                    inst.initWebGPU(output.device);
	                }
	                else {
	                    this.log("Cannot find WebGPU device in the env");
	                }
	            }
	            catch (err) {
	                this.log("Cannnot initialize WebGPU, " + err.toString());
	            }
	            this.inst = inst;
	            // begin scope to allow handling of objects
	            this.inst.beginScope();
	            if (this.initProgressCallback !== undefined) {
	                this.inst.registerInitProgressCallback(this.initProgressCallback);
	            }
	            if (this.ndarrayCacheUrl.length != 0) {
	                if (this.ndarrayCacheDevice == "cpu") {
	                    yield this.inst.fetchNDArrayCache(this.ndarrayCacheUrl, this.inst.cpu());
	                }
	                else {
	                    support.assert(this.ndarrayCacheDevice == "webgpu");
	                    yield this.inst.fetchNDArrayCache(this.ndarrayCacheUrl, this.inst.webgpu());
	                }
	            }
	            support.assert(this.inst !== undefined);
	            if (this.asyncOnServerLoad !== undefined) {
	                yield this.asyncOnServerLoad(this.inst);
	            }
	            const fcreate = this.inst.getGlobalFunc("rpc.CreateEventDrivenServer");
	            const messageHandler = fcreate((cbytes) => {
	                support.assert(this.inst !== undefined);
	                if (this.socket.readyState == 1) {
	                    // WebSocket will automatically close the socket
	                    // if we burst send data that exceeds its internal buffer
	                    // wait a bit before we send next one.
	                    const sendDataWithCongestionControl = () => __awaiter(this, void 0, void 0, function* () {
	                        const packetSize = 4 << 10;
	                        const maxBufferAmount = 4 * packetSize;
	                        const waitTimeMs = 20;
	                        for (let offset = 0; offset < cbytes.length; offset += packetSize) {
	                            const end = Math.min(offset + packetSize, cbytes.length);
	                            while (this.socket.bufferedAmount >= maxBufferAmount) {
	                                yield new Promise((r) => setTimeout(r, waitTimeMs));
	                            }
	                            this.socket.send(cbytes.slice(offset, end));
	                        }
	                    });
	                    // Chain up the pending send so that the async send is always in-order.
	                    this.pendingSend = this.pendingSend.then(sendDataWithCongestionControl);
	                    // Directly return since the data are "sent" from the caller's pov.
	                    return this.inst.scalar(cbytes.length, "int32");
	                }
	                else {
	                    return this.inst.scalar(0, "int32");
	                }
	            }, this.name, this.key);
	            // message handler should persist across RPC runs
	            this.globalObjects.push(this.inst.detachFromCurrentScope(messageHandler));
	            const writeFlag = this.inst.scalar(3, "int32");
	            this.serverRecvData = (header, body) => {
	                if (messageHandler(header, writeFlag) == 0) {
	                    this.socket.close();
	                }
	                if (messageHandler(body, writeFlag) == 0) {
	                    this.socket.close();
	                }
	            };
	            // Forward the same init sequence to the wasm RPC.
	            // The RPC will look for "rpc.wasmSession"
	            // and we will redirect it to the correct local session.
	            // register the callback to redirect the session to local.
	            const flocal = this.inst.getGlobalFunc("wasm.LocalSession");
	            const localSession = flocal();
	            support.assert(localSession instanceof runtime.Module);
	            // eslint-disable-next-line @typescript-eslint/no-unused-vars
	            this.inst.registerFunc("rpc.WasmSession", 
	            // eslint-disable-next-line @typescript-eslint/no-unused-vars
	            (_args) => {
	                return localSession;
	            });
	            messageHandler(header, writeFlag);
	            messageHandler(body, writeFlag);
	            this.log("Finish initializing the Wasm Server..");
	            this.requestBytes(8 /* I64 */);
	            this.state = RPCServerState.ReceivePacketHeader;
	            // call process events in case there are bufferred data.
	            this.processEvents();
	            // recycle all values.
	            this.inst.endScope();
	        });
	        this.state = RPCServerState.WaitForCallback;
	        asyncInitServer();
	    }
	    log(msg) {
	        this.logger(this.name + msg);
	    }
	    handleInitHeader() {
	        const reader = new ByteStreamReader(this.readFromBuffer(4 /* I32 */ * 2));
	        const magic = reader.readU32();
	        if (magic == RPC_MAGIC + 1) {
	            throw new Error("key: " + this.key + " has already been used in proxy");
	        }
	        else if (magic == RPC_MAGIC + 2) {
	            throw new Error("RPCProxy do not have matching client key " + this.key);
	        }
	        support.assert(magic == RPC_MAGIC, this.url + " is not an RPC Proxy");
	        this.remoteKeyLength = reader.readU32();
	        support.assert(this.pendingBytes == 0);
	        this.requestBytes(this.remoteKeyLength);
	        this.state = RPCServerState.InitHeaderKey;
	    }
	    handleInitHeaderKey() {
	        // eslint-disable-next-line @typescript-eslint/no-unused-vars
	        support.Uint8ArrayToString(this.readFromBuffer(this.remoteKeyLength));
	        support.assert(this.pendingBytes == 0);
	        this.requestBytes(8 /* I64 */);
	        this.state = RPCServerState.ReceivePacketHeader;
	    }
	    checkLittleEndian() {
	        const a = new ArrayBuffer(4);
	        const b = new Uint8Array(a);
	        const c = new Uint32Array(a);
	        b[0] = 0x11;
	        b[1] = 0x22;
	        b[2] = 0x33;
	        b[3] = 0x44;
	        support.assert(c[0] === 0x44332211, "RPCServer little endian to work");
	    }
	    requestBytes(nbytes) {
	        this.pendingBytes += nbytes;
	    }
	    readFromBuffer(nbytes) {
	        const ret = new Uint8Array(nbytes);
	        let ptr = 0;
	        while (ptr < nbytes) {
	            support.assert(this.messageQueue.length != 0);
	            const nleft = nbytes - ptr;
	            if (this.messageQueue[0].byteLength <= nleft) {
	                const buffer = this.messageQueue.shift();
	                ret.set(buffer, ptr);
	                ptr += buffer.byteLength;
	            }
	            else {
	                const buffer = this.messageQueue[0];
	                ret.set(buffer.slice(0, nleft), ptr);
	                this.messageQueue[0] = buffer.slice(nleft, buffer.byteLength);
	                ptr += nleft;
	            }
	        }
	        this.buffredBytes -= nbytes;
	        this.pendingBytes -= nbytes;
	        return ret;
	    }
	}
	exports.RPCServer = RPCServer;

	});

	unwrapExports(rpc_server);
	rpc_server.RPCServer;

	var dist = createCommonjsModule(function (module, exports) {
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
	Object.defineProperty(exports, "__esModule", { value: true });

	Object.defineProperty(exports, "Scalar", { enumerable: true, get: function () { return runtime.Scalar; } });
	Object.defineProperty(exports, "DLDevice", { enumerable: true, get: function () { return runtime.DLDevice; } });
	Object.defineProperty(exports, "DLDataType", { enumerable: true, get: function () { return runtime.DLDataType; } });
	Object.defineProperty(exports, "Module", { enumerable: true, get: function () { return runtime.Module; } });
	Object.defineProperty(exports, "NDArray", { enumerable: true, get: function () { return runtime.NDArray; } });
	Object.defineProperty(exports, "TVMArray", { enumerable: true, get: function () { return runtime.TVMArray; } });
	Object.defineProperty(exports, "Instance", { enumerable: true, get: function () { return runtime.Instance; } });
	Object.defineProperty(exports, "instantiate", { enumerable: true, get: function () { return runtime.instantiate; } });

	Object.defineProperty(exports, "RPCServer", { enumerable: true, get: function () { return rpc_server.RPCServer; } });

	Object.defineProperty(exports, "wasmPath", { enumerable: true, get: function () { return support.wasmPath; } });

	Object.defineProperty(exports, "detectGPUDevice", { enumerable: true, get: function () { return webgpu.detectGPUDevice; } });
	var support_2 = support;
	Object.defineProperty(exports, "assert", { enumerable: true, get: function () { return support_2.assert; } });

	});

	var index = unwrapExports(dist);
	var dist_1 = dist.Scalar;
	var dist_2 = dist.DLDevice;
	var dist_3 = dist.DLDataType;
	var dist_4 = dist.Module;
	var dist_5 = dist.NDArray;
	var dist_6 = dist.TVMArray;
	var dist_7 = dist.Instance;
	var dist_8 = dist.instantiate;
	var dist_9 = dist.RPCServer;
	var dist_10 = dist.wasmPath;
	var dist_11 = dist.detectGPUDevice;
	var dist_12 = dist.assert;

	exports.DLDataType = dist_3;
	exports.DLDevice = dist_2;
	exports.Instance = dist_7;
	exports.Module = dist_4;
	exports.NDArray = dist_5;
	exports.RPCServer = dist_9;
	exports.Scalar = dist_1;
	exports.TVMArray = dist_6;
	exports.assert = dist_12;
	exports["default"] = index;
	exports.detectGPUDevice = dist_11;
	exports.instantiate = dist_8;
	exports.wasmPath = dist_10;

	Object.defineProperty(exports, '__esModule', { value: true });

}));
