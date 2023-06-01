package ai.mlc.mlcchat

import android.os.Looper
import org.apache.tvm.Device
import org.apache.tvm.Function
import org.apache.tvm.Module

class ChatModule {
    private var reload_func: Function? = null
    private var unload_func: Function? = null
    private var prefill_func_: Function? = null
    private var decode_func_: Function? = null
    private var get_message_: Function? = null
    private var stopped_func_: Function? = null
    private var reset_chat_func_: Function? = null
    private var runtime_stats_text_func_: Function? = null
    private var llm_chat_: Module;

    init {
        val fcreate = Function.getFunction("mlc.llm_chat_create")
        assert(fcreate != null)
        llm_chat_ = fcreate.pushArg(Device.opencl().deviceType).pushArg(0).invoke().asModule()

        reload_func = llm_chat_.getFunction("reload")
        unload_func = llm_chat_.getFunction("unload")
        prefill_func_ = llm_chat_.getFunction("prefill")
        decode_func_ = llm_chat_.getFunction("decode")
        get_message_ = llm_chat_.getFunction("get_message")
        stopped_func_ = llm_chat_.getFunction("stopped")
        reset_chat_func_ = llm_chat_.getFunction("reset_chat")
        runtime_stats_text_func_ = llm_chat_.getFunction("runtime_stats_text")

        assert(reload_func != null)
        assert(unload_func != null)
        assert(prefill_func_ != null)
        assert(decode_func_ != null)
        assert(get_message_ != null)
        assert(stopped_func_ != null)
        assert(reset_chat_func_ != null)
        assert(runtime_stats_text_func_ != null)
    }

    fun unload() {
        assert(!Looper.getMainLooper().isCurrentThread)
        unload_func!!.invoke()
    }

    fun reload(modelLib: String, modelPath: String) {
        assert(!Looper.getMainLooper().isCurrentThread)
        var lib_prefix = modelLib
        System.err.println("lib_prefix: $lib_prefix")
        lib_prefix = lib_prefix.replace('-', '_')
        lib_prefix += "_"
        System.err.println("lib_prefix: $lib_prefix")
        var system_lib_func = Function.getFunction("runtime.SystemLib")
        assert(system_lib_func != null)
        System.err.println("system_lib_func: $system_lib_func")
        system_lib_func = system_lib_func.pushArg(lib_prefix)
        val lib = system_lib_func!!.invoke().asModule()
        assert(lib != null)
        System.err.println("lib: $lib")
        System.err.println("modelPath: $modelPath")
        System.err.println("reload_func: $reload_func")
        reload_func = reload_func!!.pushArg(lib).pushArg(modelPath)
        reload_func!!.invoke()
    }

    fun resetChat() {
        assert(!Looper.getMainLooper().isCurrentThread)
        reset_chat_func_!!.invoke();
    }

    fun prefill(input: String) {
        assert(!Looper.getMainLooper().isCurrentThread)
        prefill_func_!!.pushArg(input).invoke();
    }

    fun getMessage(): String {
        assert(!Looper.getMainLooper().isCurrentThread)
        return get_message_!!.invoke().asString()
    }

    fun runtimeStatsText(): String {
        assert(!Looper.getMainLooper().isCurrentThread)
        return runtime_stats_text_func_!!.invoke().asString()
    }

    fun evaluate() {
        llm_chat_.getFunction("evaluate").invoke()
    }

    fun stopped(): Boolean {
        assert(!Looper.getMainLooper().isCurrentThread)
        return stopped_func_!!.invoke().asLong() != 0L
    }

    fun decode() {
        assert(!Looper.getMainLooper().isCurrentThread)
        decode_func_!!.invoke()
    }
}