"""FFI APIs for mlc_llm"""

import tvm_ffi

# Exports functions registered via TVM_FFI_REGISTER_GLOBAL with the "mlc" prefix.
# e.g. TVM_FFI_REGISTER_GLOBAL("mlc.Tokenizer")
tvm_ffi.init_ffi_api("mlc.tokenizers", __name__)  # pylint: disable=protected-access
