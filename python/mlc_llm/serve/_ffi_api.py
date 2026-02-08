"""FFI APIs for mlc_llm.serve"""

import tvm_ffi

# Exports functions registered via TVM_FFI_REGISTER_GLOBAL with the "mlc.serve" prefix.
# e.g. TVM_FFI_REGISTER_GLOBAL("mlc.serve.TextData")
tvm_ffi.init_ffi_api("mlc.serve", __name__)  # pylint: disable=protected-access
