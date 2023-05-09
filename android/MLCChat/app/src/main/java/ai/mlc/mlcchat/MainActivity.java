package ai.mlc.mlcchat;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentHashMap;

public class MainActivity extends AppCompatActivity {
    private MessageAdapter messageAdapter;
    private Button sendButton;
    private Button resetButton;
    private EditText editText;
    private ListView listView;
    private TextView speedText;
    private Handler handler;
    private ChatState chatState;
    private Context context;

    private Downloader downloader;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sendButton = findViewById(R.id.send);
        resetButton = findViewById(R.id.reset);
        editText = findViewById(R.id.input);
        listView = findViewById(R.id.messages);
        speedText = findViewById(R.id.speed);

        context = getApplicationContext();

        messageAdapter = new MessageAdapter(this);
        listView.setAdapter(messageAdapter);
        handler = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(@NonNull Message msg) {
                super.handleMessage(msg);
                Bundle bundle = msg.getData();
                switch (msg.what) {
                    case Config.INIT_DONE:
                        assert messageAdapter.isBotLast();
                        updateMessage(new MessageData(MessageRole.BOT, "[System] Ready to chat"));
                        resumeSend();
                        resumeReset();
                        break;
                    case Config.UPD_MSG:
                        assert messageAdapter.isBotLast();
                        updateMessage(new MessageData(MessageRole.BOT, bundle.getString(Config.MSG_KEY)));
                        break;
                    case Config.APP_MSG:
                        appendMessage(new MessageData(MessageRole.BOT, bundle.getString(Config.MSG_KEY)));
                        break;
                    case Config.END:
                        speedText.setText(bundle.getString(Config.STATS_KEY));
                        resumeSend();
                        break;
                    case Config.PARAMS_DONE:
                        init();
                        break;
                    case Config.RESET:
                        reset();
                        break;
                }
            }
        };
        // System starting: freeze send and reset
        freezeSend();
        freezeReset();
        // Prepare tokenizer and model params
        downloader = new Downloader(handler, context);
        prepareParams();
    }

    private void init() {
        // Follow up after param is confirmed
        // Chat state will initialize in another thread
        appendMessage(new MessageData(MessageRole.BOT, "[System] Initializing..."));
        chatState = new ChatState(handler, context);

        sendButton.setOnClickListener(v -> {
            String message = freezeSend();
            appendMessage(new MessageData(MessageRole.USER, message));
            appendMessage(new MessageData(MessageRole.BOT, ""));
            chatState.generate(message);
        });
        resetButton.setOnClickListener(v -> {
            freezeSend();
            freezeReset();
            chatState.reset();
        });
    }

    private void prepareParams() {
        appendMessage(new MessageData(MessageRole.BOT, "[System] Preparing Parameters..."));
        String model_name = "vicuna-v1-7b";
        String base_url = "https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int4/resolve/main";
        downloader.addFile(base_url + "/tokenizer.model", model_name + "/tokenizer.model");
        downloader.addFile(base_url + "/float16/ndarray-cache.json", model_name + "/params/ndarray-cache.json");
        for (int i = 0; i <= 131; ++i) {
            String param_name = "params_shard_" + i + ".bin";
            downloader.addFile(base_url + "/float16/" + param_name, model_name + "/params/" + param_name);
        }
        downloader.download();
    }


    private String freezeSend() {
        String text = editText.getText().toString();
        editText.getText().clear();
        editText.setEnabled(false);
        sendButton.setEnabled(false);
        sendButton.setAlpha((float) 0.1);
        return text;
    }

    private void resumeSend() {
        sendButton.setEnabled(true);
        sendButton.setAlpha((float) 1.0);
        editText.setEnabled(true);
    }

    private void freezeReset() {
        resetButton.setEnabled(false);
    }

    private void resumeReset() {
        resetButton.setEnabled(true);
    }

    @Override
    protected void onRestart() {
        super.onRestart();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        chatState.terminate();
        downloader.terminate();
    }

    private void appendMessage(MessageData messageData) {
        messageAdapter.appendMessage(messageData);
    }

    private void updateMessage(MessageData messageData) {
        messageAdapter.updateMessage(messageData);
    }

    private void reset() {
        resumeSend();
        editText.getText().clear();
        messageAdapter.reset();
        resumeReset();
    }

}