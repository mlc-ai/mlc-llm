package ai.mlc.mlcengineexample

import ai.mlc.mlcengineexample.ui.theme.MLCEngineExampleTheme
import ai.mlc.mlcllm.MLCEngine
import ai.mlc.mlcllm.OpenAIProtocol.*
import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.launch
import java.io.File


class MainActivity : ComponentActivity() {
    @SuppressLint("CoroutineCreationDuringComposition")
    @ExperimentalMaterial3Api
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val modelName = "phi-2-q4f16_1-MLC"
        var modelPath = File(application.getExternalFilesDir(""), modelName).toString()
        Log.i("MLC", "model path: $modelPath")
        // need to be changed to the custom system lib prefix used while compiling the model
        val modelLib = "phi_msft_q4f16_1_4aec0e0a2bf3cf16e8dc33c012538136"
        Log.i("MLC", "engine loaded")

        setContent {
            val responseText = remember { mutableStateOf("") }
            val coroutineScope = rememberCoroutineScope()
            val engine = MLCEngine()
            engine.reload(modelPath, modelLib)
            val messages=listOf(
                ChatCompletionMessage(
                    role=ChatCompletionRole.user,
                    content="What is the meaning of life?"
                )
            )
            val response: ReceiveChannel<ChatCompletionStreamResponse> = engine.chatCompletion(
                messages=listOf(
                    ChatCompletionMessage(
                        role=ChatCompletionRole.user,
                        content="What is the meaning of life?"
                    )
                ),
                model=modelPath,
            )
            coroutineScope.launch {
                for (it in response) {
                    responseText.value += it.choices[0].delta.content?.asText()
                }
            }
            Surface(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                MLCEngineExampleTheme {
                    Text(text = responseText.value)
                }
            }
        }
    }
}
