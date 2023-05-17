package ai.mlc.mlcchat;

import android.content.Context;
import android.os.Environment;

import org.apache.tvm.Device;
import org.apache.tvm.Function;
import org.apache.tvm.Module;

public class LLMChat {
    private Function encode_func_;
    private Function decode_func_;
    private Function get_message_;
    private Function stopped_func_;
    private Function reset_chat_func_;
    private Function runtime_stats_text_func_;
    private Module llm_chat_;
    private final Context context;

    public LLMChat(Context c) {
        this.context = c;
    }

    public void Init() {
        Function systemlib_func = Function.getFunction("runtime.SystemLib");
        assert systemlib_func != null;
        Module lib = systemlib_func.invoke().asModule();
        assert lib != null;
        Function fcreate = Function.getFunction("mlc.llm_chat_create");
        assert fcreate != null;
        String dist_path = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();
        String tokenizer_path = dist_path + "/vicuna-v1-7b/tokenizer.model";
        String param_path = dist_path + "/vicuna-v1-7b/params";
        System.err.println("[INFO] " + tokenizer_path);
        System.err.println("[INFO] " + param_path);
        System.err.println("[INFO] Before LLM Chat create");
        llm_chat_ = fcreate.pushArg(lib).pushArg(tokenizer_path).pushArg(param_path).pushArg(Device.opencl().deviceType).pushArg(0).invoke().asModule();
        System.err.println("[INFO] LLM Chat created!");
        encode_func_ = llm_chat_.getFunction("encode");
        decode_func_ = llm_chat_.getFunction("decode");
        get_message_ = llm_chat_.getFunction("get_message");

        stopped_func_ = llm_chat_.getFunction("stopped");
        reset_chat_func_ = llm_chat_.getFunction("reset_chat");

        runtime_stats_text_func_ = llm_chat_.getFunction("runtime_stats_text");

        assert encode_func_ != null;
        assert decode_func_ != null;
        assert stopped_func_ != null;
        assert runtime_stats_text_func_ != null;

        String model = "vicuna";
        String conv_template = "vicuna_v1.1";
        double temperature = 0.7;
        double top_p = 0.95;
        int stream_interval = 1;
        int max_window_size = 512 + 256;
        int mean_gen_len = 128;
        double shift_fill_factor = 0.2;
        llm_chat_.getFunction("init_chat").pushArg(model).pushArg(conv_template).pushArg(temperature).pushArg(top_p).pushArg(stream_interval).pushArg(max_window_size).pushArg(mean_gen_len).pushArg(shift_fill_factor).invoke();

        systemlib_func.release();
        lib.release();
        fcreate.release();

        System.err.println("[INFO] Init done");
    }
    public void Evaluate() {
        llm_chat_.getFunction("evaluate").invoke();
    }

    public String GetMessage() {
        return get_message_.invoke().asString();
    }

    public void Encode(String prompt) {
        encode_func_.pushArg(prompt).invoke();
    }

    public boolean Stopped() {
        return stopped_func_.invoke().asLong() != 0;
    }

    public void Decode() {
        decode_func_.invoke();
    }

    public String RuntimeStatsText() {
        return runtime_stats_text_func_.invoke().asString();
    }

    public void ResetChat() {
        reset_chat_func_.invoke();
    }
}
