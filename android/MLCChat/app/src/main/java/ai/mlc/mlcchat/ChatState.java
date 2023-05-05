package ai.mlc.mlcchat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.content.Context;
import android.os.Handler;

public class ChatState {
    private boolean resetRequested;
    private boolean stopRequested;
    private final ExecutorService executorService;
    private final Handler handler;

    private final LLMChat backend;

    public ChatState(Handler h, Context c) {
        resetRequested = false;
        stopRequested = false;
        executorService = Executors.newSingleThreadExecutor();
        handler = h;
        backend = new LLMChat(c);
        executorService.execute(() -> {
            backend.Init();
            Utils.sendInitDone(h);
        });
    }

    public void stop() {
        if (isStopRequested()) return;
        requestStop();
        executorService.execute(this::finishStop);
    }

    public void reset() {
        stop();
        if (isResetRequested()) return;
        requestReset();
        executorService.execute(() -> {
            backend.ResetChat();
            finishReset();
            Utils.sendReset(handler);
        });
    }

    public synchronized void requestReset() {
        assert !resetRequested;
        resetRequested = true;
    }

    public synchronized void finishReset() {
        assert resetRequested;
        resetRequested = false;
    }

    public synchronized boolean isResetRequested() {
        return resetRequested;
    }

    public synchronized void requestStop() {
        assert !stopRequested;
        stopRequested = true;
    }

    public synchronized void finishStop() {
        assert stopRequested;
        stopRequested = false;
    }

    public synchronized boolean isStopRequested() {
        return stopRequested;
    }

    void generate(String text) {
        assert !isStopRequested() && !isResetRequested();
        executorService.execute(() -> Generate(text, handler));
    }


    void Dummy(String text, Handler handler) {
        String result = "";
        for (int i = 0; i < 8; ++i) {
            try {
                Thread.sleep(1000);
                if (isStopRequested()) {
                    break;
                }
                result = result + " aba";
                Utils.sendUpdateMessage(result, handler);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        Utils.sendEnd("encode: 100.0 tok/s, decode: 100.0 tok/s", handler);
    }

    void Generate(String prompt, Handler handler) {
        // System.err.println("Start generating");
        backend.Encode(prompt);
        // System.err.println("Encoding " + prompt);
        while (!backend.Stopped()) {
            backend.Decode();
            // System.err.println("[INFO] " + backend.GetMessage());
            Utils.sendUpdateMessage(backend.GetMessage(), handler);
            if (isStopRequested()) {
                break;
            }
        }
        String stats = backend.RuntimeStatsText();
        Utils.sendEnd(stats, handler);
    }

    public void terminate() {
        Utils.terminate(executorService);
    }

}
