"""FFI APIs for mlc_llm grammar"""

import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "mlc.grammar" prefix.
tvm._ffi._init_api("mlc.grammar", __name__)  # pylint: disable=protected-access
