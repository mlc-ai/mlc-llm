package ai.mlc.mlcchat;

import android.content.Context;
import android.os.Environment;
import android.os.Handler;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Downloader {
    private ExecutorService executorService;
    private Handler handler;
    private boolean isDownloading;

    private Context context;

    private HashMap<String, String> files;

    public Downloader(Handler h, Context c) {
        handler = h;
        context = c;
        executorService = Executors.newSingleThreadExecutor();
        isDownloading = false;
        files = new HashMap<>();
    }

    public synchronized void startDownload() {
        assert !isDownloading;
        isDownloading = true;
    }

    public synchronized void endDownload() {
        assert isDownloading;
        isDownloading = false;
    }

    public synchronized boolean inDownload() {
        return isDownloading;
    }

    public void addFile(String url, String path) {
        assert !inDownload();
        File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), path);
        if (!file.exists()) {
            files.put(url, path);
        }
    }

    public static String reportProgress(int done, int all) {
        return String.format("[System] downloading [%d / %d]", done, all);
    }

    public void download() {
        startDownload();
        executorService.execute(() -> {
            HashMap<String, String> filesToDownload = new HashMap<>(files);
            files.clear();
            int n = filesToDownload.size();
            int i = 0;
            if (n > 0) {
                Utils.sendAppendMessage(reportProgress(i, n), handler);
            }
            for (Map.Entry<String, String> entry : filesToDownload.entrySet()) {
                try {
                    File temp = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "temp");
                    File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), entry.getValue());
                    if (!file.exists()) {
                        URL url = new URL(entry.getKey());
                        InputStream inputStream = url.openStream();
                        DataInputStream dataInputStream =
                                new DataInputStream(inputStream);
                        byte[] buffer = new byte[1024];
                        int length;

                        FileOutputStream fileOutputStream = new FileOutputStream(temp);
                        while ((length = dataInputStream.read(buffer)) > 0) {
                            fileOutputStream.write(buffer, 0, length);
                        }

                        Files.createDirectories(file.toPath().getParent());
                        Files.move(temp.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE);
                    }
                    ++i;
                    Utils.sendUpdateMessage(reportProgress(i, n), handler);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            endDownload();
            Utils.sendParamsDone(handler);
        });
    }

    public void terminate() {
        Utils.terminate(executorService);
    }
}
