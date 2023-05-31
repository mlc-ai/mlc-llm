package ai.mlc.mlcchat

import android.os.Looper

class ChatModule {
    var count = 0
    var text = ""
    val maxCount = 10

    init {

    }

    fun unload() {
        assert(!Looper.getMainLooper().isCurrentThread)
        Thread.sleep(1000)
    }

    fun reload(modelLib: String, modelPath: String) {
        assert(!Looper.getMainLooper().isCurrentThread)
        Thread.sleep(1000)
    }

    fun resetChat() {
        assert(!Looper.getMainLooper().isCurrentThread)
        count = 0
        Thread.sleep(1000)
    }

    fun prefill(input: String) {
        assert(!Looper.getMainLooper().isCurrentThread)
        count = 0
        text = input
        Thread.sleep(1000)
    }

    fun getMessage(): String {
        assert(!Looper.getMainLooper().isCurrentThread)
        return text.repeat(count)
    }

    fun runtimeStatsText(): String {
        assert(!Looper.getMainLooper().isCurrentThread)
        return "prefill: 12.3 tok/s, decode: 45.6 tok/s"
    }

    fun evaluate() {
        assert(!Looper.getMainLooper().isCurrentThread)

    }

    fun stopped(): Boolean {
        assert(!Looper.getMainLooper().isCurrentThread)
        return count == maxCount
    }

    fun decode() {
        assert(!Looper.getMainLooper().isCurrentThread)
        Thread.sleep(1000)
        ++count
    }
}