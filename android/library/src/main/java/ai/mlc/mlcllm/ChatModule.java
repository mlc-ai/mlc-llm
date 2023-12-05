package ai.mlc.mlcllm;

import org.apache.tvm.Device;
import org.apache.tvm.Function;
import org.apache.tvm.Module;

public class ChatModule {
    private Function reloadFunc;
    private Function unloadFunc;
    private Function prefillFunc;
    private Function decodeFunc;
    private Function getMessage;
    private Function stoppedFunc;
    private Function resetChatFunc;
    private Function runtimeStatsTextFunc;
    private Module llmChat;

    public ChatModule() {
        Function createFunc = Function.getFunction("mlc.llm_chat_create");
        assert createFunc != null;
        llmChat = createFunc.pushArg(Device.opencl().deviceType).pushArg(0).invoke().asModule();
        reloadFunc = llmChat.getFunction("reload");
        unloadFunc = llmChat.getFunction("unload");
        prefillFunc = llmChat.getFunction("prefill");
        decodeFunc = llmChat.getFunction("decode");
        getMessage = llmChat.getFunction("get_message");
        stoppedFunc = llmChat.getFunction("stopped");
        resetChatFunc = llmChat.getFunction("reset_chat");
        runtimeStatsTextFunc = llmChat.getFunction("runtime_stats_text");
    }

    public void unload() {
        unloadFunc.invoke();
    }

    public void reload(String modelLib, String modelPath) {
        String libPrefix = modelLib.replace('-', '_') + "_";
        Function systemLibFunc = Function.getFunction("runtime.SystemLib");
        assert systemLibFunc != null;
        systemLibFunc = systemLibFunc.pushArg(libPrefix);
        Module lib = systemLibFunc.invoke().asModule();
        reloadFunc = reloadFunc.pushArg(lib).pushArg(modelPath);
        reloadFunc.invoke();
    }

    public void resetChat() {
        resetChatFunc.invoke();
    }

    public void prefill(String input) {
        prefillFunc.pushArg(input).invoke();
    }

    public String getMessage() {
        return getMessage.invoke().asString();
    }

    public String runtimeStatsText() {
        return runtimeStatsTextFunc.invoke().asString();
    }

    public void evaluate() {
        llmChat.getFunction("evaluate").invoke();
    }

    public boolean stopped() {
        return stoppedFunc.invoke().asLong() != 0L;
    }

    public void decode() {
        decodeFunc.invoke();
    }
}