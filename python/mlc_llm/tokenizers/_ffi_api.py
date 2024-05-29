"""FFI APIs for mlc_llm"""

import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "mlc" prefix.
# e.g. TVM_REGISTER_GLOBAL("mlc.Tokenizer")
tvm._ffi._init_api("mlc.tokenizers", __name__)  # pylint: disable=protected-access
