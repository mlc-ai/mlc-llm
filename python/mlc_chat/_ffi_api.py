"""FFI APIs for mlc_chat"""
import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "mlc" prefix.
# e.g. TVM_REGISTER_GLOBAL("mlc.Tokenizer")
tvm._ffi._init_api("mlc", __name__)  # pylint: disable=protected-access
