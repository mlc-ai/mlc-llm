package ai.mlc.mlcengineexample

import ai.mlc.mlcengineexample.ui.theme.MLCEngineExampleTheme
import ai.mlc.mlcllm.MLCEngine
import ai.mlc.mlcllm.OpenAIProtocol
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
        val modelLib = "phi_msft_q4f16_1_686d8979c6ebf05d142d9081f1b87162"
        Log.i("MLC", "engine loaded")

        setContent {
            val responseText = remember { mutableStateOf("") }
            val coroutineScope = rememberCoroutineScope()
            val engine = MLCEngine()
            engine.unload()
            engine.reload(modelPath, modelLib)
            coroutineScope.launch {
                var channel = engine.chat.completions.create(
                    messages = listOf(
                        ChatCompletionMessage(
                            role = OpenAIProtocol.ChatCompletionRole.user,
                            content = "What is the meaning of life?"
                        )
                    ),
                    stream_options = OpenAIProtocol.StreamOptions(include_usage = true)
                )


                for (response in channel) {
                    val finalusage = response.usage
                    if (finalusage != null) {
                        responseText.value += "\n" + (finalusage.extra?.asTextLabel() ?: "")
                    } else {
                        if (response.choices.size > 0) {
                            responseText.value += response.choices[0].delta.content?.asText()
                                .orEmpty()
                        }
                    }

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
