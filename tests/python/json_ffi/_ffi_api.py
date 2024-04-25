"""FFI APIs for mlc.json_ffi"""
import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "mlc.json_ffi" prefix.
# e.g. TVM_REGISTER_GLOBAL("mlc.serve.TextData")
tvm._ffi._init_api("mlc.json_ffi", __name__)  # pylint: disable=protected-access
