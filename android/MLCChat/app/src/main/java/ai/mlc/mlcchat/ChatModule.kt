package ai.mlc.mlcchat

import android.os.Looper
import android.util.Log
import org.apache.tvm.Device
import org.apache.tvm.Function
import org.apache.tvm.Module

class ChatModule {
    private var reloadFunc: Function
    private var unloadFunc: Function
    private var prefillFunc: Function
    private var decodeFunc: Function
    private var getMessage: Function
    private var stoppedFunc: Function
    private var resetChatFunc: Function
    private var runtimeStatsTextFunc: Function
    private var llmChat: Module

    init {
        val fcreate = Function.getFunction("mlc.llm_chat_create")
        require(fcreate != null)
        llmChat = fcreate.pushArg(Device.opencl().deviceType).pushArg(0).invoke().asModule()

        reloadFunc = llmChat.getFunction("reload")
        unloadFunc = llmChat.getFunction("unload")
        prefillFunc = llmChat.getFunction("prefill")
        decodeFunc = llmChat.getFunction("decode")
        getMessage = llmChat.getFunction("get_message")
        stoppedFunc = llmChat.getFunction("stopped")
        resetChatFunc = llmChat.getFunction("reset_chat")
        runtimeStatsTextFunc = llmChat.getFunction("runtime_stats_text")
    }

    fun unload() {
        require(!Looper.getMainLooper().isCurrentThread)
        unloadFunc.invoke()
    }

    fun reload(modelLib: String, modelPath: String) {
        require(!Looper.getMainLooper().isCurrentThread)
        var libPrefix = modelLib
        Log.i(TAG, "lib_prefix: $libPrefix")
        libPrefix = libPrefix.replace('-', '_')
        libPrefix += "_"
        Log.i(TAG, "lib_prefix: $libPrefix")
        var systemLibFunc = Function.getFunction("runtime.SystemLib")
        require(systemLibFunc != null)
        Log.i(TAG, "system_lib_func: $systemLibFunc")
        systemLibFunc = systemLibFunc.pushArg(libPrefix)
        val lib = systemLibFunc.invoke().asModule()
        require(lib != null)
        Log.i(TAG, "lib: $lib")
        Log.i(TAG, "modelPath: $modelPath")
        Log.i(TAG, "reload_func: $reloadFunc")
        reloadFunc = reloadFunc.pushArg(lib).pushArg(modelPath)
        reloadFunc.invoke()
    }

    fun resetChat() {
        require(!Looper.getMainLooper().isCurrentThread)
        resetChatFunc.invoke();
    }

    fun prefill(input: String) {
        require(!Looper.getMainLooper().isCurrentThread)
        prefillFunc.pushArg(input).invoke();
    }

    fun getMessage(): String {
        require(!Looper.getMainLooper().isCurrentThread)
        return getMessage.invoke().asString()
    }

    fun runtimeStatsText(): String {
        require(!Looper.getMainLooper().isCurrentThread)
        return runtimeStatsTextFunc.invoke().asString()
    }

    fun evaluate() {
        llmChat.getFunction("evaluate").invoke()
    }

    fun stopped(): Boolean {
        require(!Looper.getMainLooper().isCurrentThread)
        return stoppedFunc.invoke().asLong() != 0L
    }

    fun decode() {
        require(!Looper.getMainLooper().isCurrentThread)
        decodeFunc.invoke()
    }

    companion object {
        private const val TAG = "ChatModule"
    }
}