package ai.mlc.mlcchat

import android.app.Application
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.toMutableStateList
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.nio.channels.Channels
import java.util.UUID
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class AppViewModel(application: Application) : AndroidViewModel(application) {
    val modelList = emptyList<ModelState>().toMutableStateList()
    val chatState = ChatState()
    val modelSampleList = emptyList<ModelRecord>().toMutableStateList()
    private var appConfig = AppConfig(
        emptyList(),
        emptyList<ModelRecord>().toMutableList(),
        emptyList<ModelRecord>().toMutableList()
    )
    private val application = getApplication<Application>()
    private val appDirFile = application.getExternalFilesDir("")
    private val gson = Gson()
    private val localIdSet = emptySet<String>().toMutableSet()

    companion object {
        private const val TAG = "AppViewModel"

        const val AppConfigFilename = "app-config.json"
        const val ModelConfigFilename = "mlc-chat-config.json"
        const val ParamsConfigFilename = "ndarray-cache.json"
    }

    init {
        loadAppConfig()
    }

    fun requestAddModel(url: String, localId: String?) {
        if (localId != null && localIdSet.contains(localId)) {
            Toast.makeText(
                application,
                "localId: $localId has been occupied",
                Toast.LENGTH_SHORT
            ).show()
        } else {
            downloadModelConfig(url, localId, false)
        }
    }


    private fun loadAppConfig() {
        val appConfigFile = File(appDirFile, AppConfigFilename)
        val jsonString: String = if (!appConfigFile.exists()) {
            application.assets.open(AppConfigFilename).bufferedReader().use { it.readText() }
        } else {
            appConfigFile.readText()
        }
        appConfig = gson.fromJson(jsonString, AppConfig::class.java)
        modelList.clear()
        localIdSet.clear()
        modelSampleList.clear()
        for (modelRecord in appConfig.modelList) {
            val modelDirFile = File(appDirFile, modelRecord.localId)
            val modelConfigFile = File(modelDirFile, ModelConfigFilename)
            if (modelConfigFile.exists()) {
                val modelConfigString = modelConfigFile.readText()
                val modelConfig = gson.fromJson(modelConfigString, ModelConfig::class.java)
                addModelConfig(modelConfig, modelRecord.modelUrl)
            } else {
                downloadModelConfig(modelRecord.modelUrl, modelRecord.localId, true)
            }
        }
        modelSampleList += appConfig.modelSamples
    }

    private fun updateAppConfig(modelUrl: String, localId: String) {
        appConfig.modelList.add(ModelRecord(modelUrl, localId))
        val jsonString = gson.toJson(appConfig)
        val appConfigFile = File(appDirFile, AppConfigFilename)
        appConfigFile.writeText(jsonString)
    }

    private fun addModelConfig(modelConfig: ModelConfig, modelUrl: String) {
        require(!localIdSet.contains(modelConfig.localId))
        localIdSet.add(modelConfig.localId)
        modelList.add(
            ModelState(
                modelConfig,
                modelUrl,
                File(appDirFile, modelConfig.localId)
            )
        )
    }

    private fun downloadModelConfig(modelUrl: String, localId: String?, isBuiltin: Boolean) {
        thread(start = true) {
            try {
                val url = URL("${modelUrl}${ModelConfigFilename}")
                val tempId = UUID.randomUUID().toString()
                val tempFile = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                    tempId
                )
                url.openStream().use {
                    Channels.newChannel(it).use { src ->
                        FileOutputStream(tempFile).use { fileOutputStream ->
                            fileOutputStream.channel.transferFrom(src, 0, Long.MAX_VALUE)
                        }
                    }
                }
                require(tempFile.exists())
                viewModelScope.launch {
                    val modelConfigString = tempFile.readText()
                    val modelConfig = gson.fromJson(modelConfigString, ModelConfig::class.java)
                    if (localId != null) {
                        require(modelConfig.localId == localId)
                    }
                    if (localIdSet.contains(modelConfig.localId)) {
                        tempFile.delete()
                        Toast.makeText(
                            application,
                            "${modelConfig.localId} has been used, please consider another local ID",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    val modelDirFile = File(appDirFile, modelConfig.localId)
                    val modelConfigFile = File(modelDirFile, ModelConfigFilename)
                    tempFile.copyTo(modelConfigFile, overwrite = true)
                    tempFile.delete()
                    require(modelConfigFile.exists())
                    addModelConfig(modelConfig, modelUrl)
                    if (!isBuiltin) {
                        updateAppConfig(modelUrl, modelConfig.localId)
                    }
                }
            } catch (e: Exception) {
                viewModelScope.launch {
                    Toast.makeText(
                        application,
                        "Add model failed: ${e.localizedMessage}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

        }
    }

    inner class ModelState(
        val modelConfig: ModelConfig,
        private val modelUrl: String,
        private val modelDirFile: File
    ) {
        var modelInitState = mutableStateOf(ModelInitState.Initializing)
        private var paramsConfig = ParamsConfig(emptyList())
        val progress = mutableStateOf(0)
        val total = mutableStateOf(1)
        val id: UUID = UUID.randomUUID()
        private val remainingTasks = emptySet<DownloadTask>().toMutableSet()
        private val downloadingTasks = emptySet<DownloadTask>().toMutableSet()
        private val maxDownloadTasks = 3
        private val gson = Gson()


        init {
            switchToInitializing()
        }

        private fun switchToInitializing() {
            val paramsConfigFile = File(modelDirFile, ParamsConfigFilename)
            if (paramsConfigFile.exists()) {
                loadParamsConfig()
                switchToIndexing()
            } else {
                downloadParamsConfig()
            }
        }

        private fun loadParamsConfig() {
            val paramsConfigFile = File(modelDirFile, ParamsConfigFilename)
            require(paramsConfigFile.exists())
            val jsonString = paramsConfigFile.readText()
            paramsConfig = gson.fromJson(jsonString, ParamsConfig::class.java)
        }

        private fun downloadParamsConfig() {
            thread(start = true) {
                val url = URL("${modelUrl}${ParamsConfigFilename}")
                val tempId = UUID.randomUUID().toString()
                val tempFile = File(modelDirFile, tempId)
                url.openStream().use {
                    Channels.newChannel(it).use { src ->
                        FileOutputStream(tempFile).use { fileOutputStream ->
                            fileOutputStream.channel.transferFrom(src, 0, Long.MAX_VALUE)
                        }
                    }
                }
                require(tempFile.exists())
                val paramsConfigFile = File(modelDirFile, ParamsConfigFilename)
                tempFile.renameTo(paramsConfigFile)
                require(paramsConfigFile.exists())
                viewModelScope.launch {
                    loadParamsConfig()
                    switchToIndexing()
                }
            }
        }

        fun handleStart() {
            switchToDownloading()
        }

        fun handlePause() {
            switchToPausing()
        }

        private fun switchToIndexing() {
            modelInitState.value = ModelInitState.Indexing
            progress.value = 0
            total.value = modelConfig.tokenizerFiles.size + paramsConfig.paramsRecords.size
            for (tokenizerFilename in modelConfig.tokenizerFiles) {
                val file = File(modelDirFile, tokenizerFilename)
                if (file.exists()) {
                    ++progress.value
                } else {
                    remainingTasks.add(DownloadTask(URL("${modelUrl}${tokenizerFilename}"), file))
                }
            }
            for (paramsRecord in paramsConfig.paramsRecords) {
                val file = File(modelDirFile, paramsRecord.dataPath)
                if (file.exists()) {
                    ++progress.value
                } else {
                    remainingTasks.add(
                        DownloadTask(
                            URL("${modelUrl}${paramsRecord.dataPath}"),
                            file
                        )
                    )
                }
            }
            if (progress.value < total.value) {
                switchToPaused()
            } else {
                switchToFinished()
            }
        }

        private fun switchToDownloading() {
            modelInitState.value = ModelInitState.Downloading
            for (downloadTask in remainingTasks) {
                if (downloadingTasks.size < maxDownloadTasks) {
                    handleNewDownload(downloadTask)
                } else {
                    return
                }
            }
        }

        private fun handleNewDownload(downloadTask: DownloadTask) {
            require(modelInitState.value == ModelInitState.Downloading)
            require(!downloadingTasks.contains(downloadTask))
            downloadingTasks.add(downloadTask)
            thread(start = true) {
                val tempId = UUID.randomUUID().toString()
                val tempFile = File(modelDirFile, tempId)
                downloadTask.url.openStream().use {
                    Channels.newChannel(it).use { src ->
                        FileOutputStream(tempFile).use { fileOutputStream ->
                            fileOutputStream.channel.transferFrom(src, 0, Long.MAX_VALUE)
                        }
                    }
                }
                require(tempFile.exists())
                tempFile.renameTo(downloadTask.file)
                require(downloadTask.file.exists())
                viewModelScope.launch {
                    handleFinishDownload(downloadTask)
                }
            }
        }

        private fun handleNextDownload() {
            require(modelInitState.value == ModelInitState.Downloading)
            for (downloadTask in remainingTasks) {
                if (!downloadingTasks.contains(downloadTask)) {
                    handleNewDownload(downloadTask)
                    break
                }
            }
        }

        private fun handleFinishDownload(downloadTask: DownloadTask) {
            remainingTasks.remove(downloadTask)
            downloadingTasks.remove(downloadTask)
            ++progress.value
            require(modelInitState.value == ModelInitState.Downloading || modelInitState.value == ModelInitState.Pausing)
            if (modelInitState.value == ModelInitState.Downloading) {
                if (remainingTasks.isEmpty()) {
                    if (downloadingTasks.isEmpty()) {
                        switchToFinished()
                    }
                } else {
                    handleNextDownload()
                }
            } else if (modelInitState.value == ModelInitState.Pausing) {
                if (downloadingTasks.isEmpty()) {
                    switchToPaused()
                }
            }
        }

        private fun switchToPausing() {
            modelInitState.value = ModelInitState.Pausing
        }

        private fun switchToPaused() {
            modelInitState.value = ModelInitState.Paused
        }


        private fun switchToFinished() {
            modelInitState.value = ModelInitState.Finished
        }

        fun startChat() {
            chatState.requestReloadChat(
                modelConfig.localId,
                modelConfig.modelLib,
                modelDirFile.absolutePath
            )
        }

    }

    inner class ChatState {
        val messages = emptyList<MessageData>().toMutableStateList()
        val report = mutableStateOf("")
        val modelName = mutableStateOf("")
        private var modelChatState = mutableStateOf(ModelChatState.Ready)
            @Synchronized get
            @Synchronized set
        private val backend = ChatModule()
        private var modelLib = ""
        private var modelPath = ""
        private val executorService = Executors.newSingleThreadExecutor()

        private fun mainResetChat() {
            messages.clear()
            report.value = ""
        }


        private fun switchToResetting() {
            modelChatState.value = ModelChatState.Resetting
        }

        private fun switchToGenerting() {
            modelChatState.value = ModelChatState.Generating
        }

        private fun switchToReloading() {
            modelChatState.value = ModelChatState.Reloading
        }

        private fun switchToReady() {
            modelChatState.value = ModelChatState.Ready
        }

        fun requestResetChat() {
            require(interuptable())
            if (modelChatState.value == ModelChatState.Ready) {
                switchToResetting()
                mainResetChat()
                switchToReady()
            } else if (modelChatState.value == ModelChatState.Generating) {
                switchToResetting()
                executorService.submit {
                    backend.resetChat()
                    viewModelScope.launch {
                        mainResetChat()
                        switchToReady()
                    }
                }
            } else {
                require(false)
            }
        }

        fun requestReloadChat(modelName: String, modelLib: String, modelPath: String) {
            require(interuptable())
            if (modelChatState.value == ModelChatState.Ready) {
                switchToReloading()
                mainReloadChat(modelName, modelLib, modelPath)
            } else if (modelChatState.value == ModelChatState.Generating) {
                switchToReloading()
                executorService.submit {
                    viewModelScope.launch {
                        mainReloadChat(modelName, modelLib, modelPath)
                    }
                }
            }
        }

        private fun mainReloadChat(modelName: String, modelLib: String, modelPath: String) {
            mainResetChat()
            if (this.modelName.value == modelName && this.modelLib == modelLib && this.modelPath == modelPath) {
                switchToReady()
                return
            }
            this.modelName.value = modelName
            this.modelLib = modelLib
            this.modelPath = modelPath
            executorService.submit {
                viewModelScope.launch {
                    Toast.makeText(application, "Initialize...", Toast.LENGTH_SHORT).show()
                }
                Log.i(TAG, "Unloading model")
                backend.unload()
                Log.i(TAG, "Loading model $modelLib $modelPath")
                backend.reload(modelLib, modelPath)
                Log.i(TAG, "Model loaded")
                viewModelScope.launch {
                    Toast.makeText(application, "Ready to chat", Toast.LENGTH_SHORT).show()
                    switchToReady()
                }
            }
        }

        fun requestGenerate(prompt: String) {
            require(chatable())
            switchToGenerting()
            executorService.submit {
                appendMessage(MessageRole.User, prompt)
                appendMessage(MessageRole.Bot, "")
                backend.prefill(prompt)
                while (!backend.stopped()) {
                    backend.decode()
                    val newText = backend.getMessage()
                    viewModelScope.launch { updateMessage(MessageRole.Bot, newText) }
                    if (modelChatState.value != ModelChatState.Generating) return@submit
                }
                val runtimeStats = backend.runtimeStatsText()
                viewModelScope.launch {
                    report.value = runtimeStats
                    if (modelChatState.value == ModelChatState.Generating) switchToReady()
                }
            }
        }

        private fun appendMessage(role: MessageRole, text: String) {
            messages.add(MessageData(role, text))
        }


        private fun updateMessage(role: MessageRole, text: String) {
            messages[messages.size - 1] = MessageData(role, text)
        }

        fun chatable(): Boolean {
            return modelChatState.value == ModelChatState.Ready
        }

        fun interuptable(): Boolean {
            return modelChatState.value == ModelChatState.Ready || modelChatState.value == ModelChatState.Generating
        }
    }
}

enum class ModelInitState {
    Initializing,
    Indexing,
    Paused,
    Downloading,
    Pausing,
    Finished
}

enum class ModelChatState {
    Generating,
    Resetting,
    Reloading,
    Ready
}

enum class MessageRole {
    Bot,
    User
}

data class DownloadTask(val url: URL, val file: File)

data class MessageData(val role: MessageRole, val text: String, val id: UUID = UUID.randomUUID())

data class AppConfig(
    @SerializedName("model_libs") val modelLibs: List<String>,
    @SerializedName("model_list") val modelList: MutableList<ModelRecord>,
    @SerializedName("add_model_samples") val modelSamples: MutableList<ModelRecord>
)

data class ModelRecord(
    @SerializedName("model_url") val modelUrl: String,
    @SerializedName("local_id") val localId: String
)

data class ModelConfig(
    @SerializedName("model_lib") val modelLib: String,
    @SerializedName("local_id") val localId: String,
    @SerializedName("tokenizer_files") val tokenizerFiles: List<String>
)

data class ParamsRecord(
    @SerializedName("dataPath") val dataPath: String
)

data class ParamsConfig(
    @SerializedName("records") val paramsRecords: List<ParamsRecord>
)