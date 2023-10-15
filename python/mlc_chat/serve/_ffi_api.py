"""FFI APIs for mlc_chat.serve"""
import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "mlc.serve" prefix.
# e.g. TVM_REGISTER_GLOBAL("mlc.serve.TextData")
tvm._ffi._init_api("mlc.serve", __name__)  # pylint: disable=protected-access
