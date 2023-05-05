package ai.mlc.mlcchat;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public class Utils {
    public static void sendUpdateMessage(String text, Handler handler) {
        Bundle bundle = new Bundle();
        bundle.putString(Config.MSG_KEY, text);
        Message message = new Message();
        message.setData(bundle);
        message.what = Config.UPD_MSG;
        handler.sendMessage(message);
    }

    public static void sendAppendMessage(String text, Handler handler) {
        Bundle bundle = new Bundle();
        bundle.putString(Config.MSG_KEY, text);
        Message message = new Message();
        message.setData(bundle);
        message.what = Config.APP_MSG;
        handler.sendMessage(message);
    }

    public static void sendEnd(String stats, Handler handler) {
        Bundle bundle = new Bundle();
        bundle.putString(Config.STATS_KEY, stats);
        Message message = new Message();
        message.setData(bundle);
        message.what = Config.END;
        handler.sendMessage(message);
    }

    public static void sendReset(Handler handler) {
        Message message = new Message();
        message.what = Config.RESET;
        handler.sendMessage(message);
    }

    public static void sendInitDone(Handler handler) {
        Message message = new Message();
        message.what = Config.INIT_DONE;
        handler.sendMessage(message);
    }

    public static void sendParamsDone(Handler handler) {
        Message message = new Message();
        message.what = Config.PARAMS_DONE;
        handler.sendMessage(message);
    }

    public static void terminate(ExecutorService executorService) {
        executorService.shutdown(); // Disable new tasks from being submitted
        try {
            // Wait a while for existing tasks to terminate
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow(); // Cancel currently executing tasks
                // Wait a while for tasks to respond to being cancelled
                if (!executorService.awaitTermination(60, TimeUnit.SECONDS))
                    System.err.println("Pool did not terminate");
            }
        } catch (InterruptedException ie) {
            // (Re-)Cancel if current thread also interrupted
            executorService.shutdownNow();
            // Preserve interrupt status
            Thread.currentThread().interrupt();
        }
    }
}
