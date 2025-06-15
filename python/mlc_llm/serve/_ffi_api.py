"""FFI APIs for mlc_llm.serve"""

import tvm.ffi

# Exports functions registered via TVM_FFI_REGISTER_GLOBAL with the "mlc.serve" prefix.
# e.g. TVM_FFI_REGISTER_GLOBAL("mlc.serve.TextData")
tvm.ffi._init_api("mlc.serve", __name__)  # pylint: disable=protected-access
