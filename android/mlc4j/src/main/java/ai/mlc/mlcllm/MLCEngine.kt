package ai.mlc.mlcllm

import ai.mlc.mlcllm.JSONFFIEngine
import ai.mlc.mlcllm.OpenAIProtocol.*
import kotlinx.coroutines.GlobalScope
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.launch
import java.lang.Exception
import java.util.UUID

class MLCEngine () {
    private val jsonFFIEngine = JSONFFIEngine()
    private val channelMap = mutableMapOf<String, Channel<ChatCompletionStreamResponse>>()

    init {
        jsonFFIEngine.initBackgroundEngine(this::streamCallback)
        GlobalScope.launch {
            jsonFFIEngine.runBackgroundLoop()
        }
        GlobalScope.launch {
            jsonFFIEngine.runBackgroundStreamBackLoop()
        }
    }

    private fun streamCallback(result: String?) {
        val responses = mutableListOf<ChatCompletionStreamResponse>()
        val json = Json { ignoreUnknownKeys = true }
        try {
            val msg = json.decodeFromString<ChatCompletionStreamResponse>(result!!)
            responses.add(msg)
        } catch (lastError: Exception) {
            println("Kotlin json parsing error: error=$lastError, jsonsrc=$result")
        }

        // dispatch to right request ID
        for (res in responses) {
            val channel = channelMap[res.id]
            if (channel != null) {
                GlobalScope.launch {
                    channel.send(res)
                    // detect finished from result
                    var finished = false
                    for (choice in res.choices) {
                        if (choice.finish_reason != "" && choice.finish_reason != null) {
                            finished = true
                        }
                    }
                    if (finished) {
                        channel.close()
                        channelMap.remove(res.id)
                    }
                }

            }
        }
    }

    private fun deinit() {
        jsonFFIEngine.exitBackgroundLoop()
    }

    fun reload(modelPath: String, modelLib: String) {
        val engineConfigJSONStr = """
            {
                "model": "$modelPath",
                "model_lib": "system://$modelLib",
                "mode": "interactive"
            }
        """.trimIndent()
        jsonFFIEngine.reload(engineConfigJSONStr)
    }

    private fun unload() {
        jsonFFIEngine.unload()
    }

    fun chatCompletion(
        messages: List<ChatCompletionMessage>,
        model: String? = null,
        frequency_penalty: Float? = null,
        presence_penalty: Float? = null,
        logprobs: Boolean = false,
        top_logprobs: Int = 0,
        logit_bias: Map<Int, Float>? = null,
        max_tokens: Int? = null,
        n: Int = 1,
        seed: Int? = null,
        stop: List<String>? = null,
        stream: Boolean = false,
        temperature: Float? = null,
        top_p: Float? = null,
        tools: List<ChatTool>? = null,
        user: String? = null,
        response_format: ResponseFormat? = null
    ): ReceiveChannel<ChatCompletionStreamResponse> {
        val request = ChatCompletionRequest(
            messages = messages,
            model = model,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            logprobs = logprobs,
            top_logprobs = top_logprobs,
            logit_bias = logit_bias,
            max_tokens = max_tokens,
            n = n,
            seed = seed,
            stop = stop,
            stream = stream,
            temperature = temperature,
            top_p = top_p,
            tools = tools,
            user = user,
            response_format = response_format
        )
        return chatCompletion(request)
    }

    private fun chatCompletion(request: ChatCompletionRequest): ReceiveChannel<ChatCompletionStreamResponse> {
        val channel = Channel<ChatCompletionStreamResponse>()
        val jsonRequest = Json.encodeToString(request)
        val requestId = UUID.randomUUID().toString()

        // Store the channel in the map for further callbacks
        channelMap[requestId] = channel

        jsonFFIEngine.chatCompletion(jsonRequest, requestId)

        return channel
    }
}
