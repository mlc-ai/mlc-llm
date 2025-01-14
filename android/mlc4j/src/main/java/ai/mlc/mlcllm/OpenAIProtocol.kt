package ai.mlc.mlcllm

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive
import java.util.*

// Data classes for v1/chat/completions
// API reference: https://platform.openai.com/docs/api-reference/chat/create

class OpenAIProtocol {
    @Serializable
    data class TopLogProbs(
        val token: String,
        val logprob: Float,
        val bytes: List<Int>? = null
    )

    @Serializable
    data class LogProbsContent(
        val token: String,
        val logprob: Float,
        var bytes: List<Int>? = null,
        var top_logprobs: List<TopLogProbs> = listOf()
    )

    @Serializable
    data class LogProbs(
        var content: List<LogProbsContent> = listOf()
    )

    @Serializable
    data class ChatFunction(
        val name: String,
        var description: String? = null,
        val parameters: Map<String, String>
    )

    @Serializable
    data class ChatTool(
        val type: String = "function",
        val function: ChatFunction
    )

    @Serializable
    data class ChatFunctionCall(
        val name: String,
        // NOTE: arguments should be dict str to any codable
        // for now only allow string output due to typing issues
        var arguments: Map<String, String>? = null
    )

    @Serializable
    data class ChatToolCall(
        val id: String = UUID.randomUUID().toString(),
        val type: String = "function",
        val function: ChatFunctionCall
    )

    @Serializable
    enum class ChatCompletionRole {
        system,
        user,
        assistant,
        tool
    }

    @Serializable(with = ChatCompletionMessageContentSerializer::class)
    data class ChatCompletionMessageContent(
        val text: String? = null,
        val parts: List<Map<String, String>>? = null
    ) {
        constructor(text: String) : this(text, null)
        constructor(parts: List<Map<String, String>>) : this(null, parts)

        fun isText(): Boolean {
            return text != null
        }

        fun isParts(): Boolean {
            return parts != null
        }

        fun asText(): String {
            return text ?: (parts?.filter { it["type"] == "text" }?.joinToString("") { it["text"] ?: "" } ?: "")
        }
    }

    object ChatCompletionMessageContentSerializer : KSerializer<ChatCompletionMessageContent> {
        override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ChatCompletionMessageContent") {
            element("text", String.serializer().descriptor)
            element("parts", ListSerializer(MapSerializer(String.serializer(), String.serializer())).descriptor)
        }

        override fun serialize(encoder: Encoder, value: ChatCompletionMessageContent) {
            if (value.isText()) {
                encoder.encodeString(value.text!!)
            } else {
                encoder.encodeSerializableValue(ListSerializer(MapSerializer(String.serializer(), String.serializer())), value.parts ?: listOf())
            }
        }

        override fun deserialize(decoder: Decoder): ChatCompletionMessageContent {
            return when (val element = decoder.decodeSerializableValue(JsonElement.serializer())) {
                is JsonArray -> {
                    val parts = element.map { (it as JsonObject).map { entry -> entry.key to entry.value.jsonPrimitive.content }.toMap() }
                    ChatCompletionMessageContent(parts)
                }
                is JsonPrimitive -> {
                    ChatCompletionMessageContent(element.content)
                }
                else -> throw IllegalStateException("Unexpected JsonElement type")
            }
        }
    }

    @Serializable
    data class ChatCompletionMessage(
        val role: ChatCompletionRole,
        var content: ChatCompletionMessageContent? = null,
        var name: String? = null,
        var tool_calls: List<ChatToolCall>? = null,
        var tool_call_id: String? = null
    ) {
        constructor(
            role: ChatCompletionRole,
            content: String,
            name: String? = null,
            tool_calls: List<ChatToolCall>? = null,
            tool_call_id: String? = null
        ) : this(role, ChatCompletionMessageContent(content), name, tool_calls, tool_call_id)
    }

    @Serializable
    data class CompletionUsageExtra(
        val prefill_tokens_per_s: Float? = null,
        val decode_tokens_per_s: Float? = null,
        val num_prefill_tokens: Int? = null
    ) {
        fun asTextLabel(): String {
            var outputText = ""
            if (prefill_tokens_per_s != null) {
                outputText += "prefill: ${String.format("%.1f", prefill_tokens_per_s)} tok/s"
            }
            if (decode_tokens_per_s != null) {
                if (outputText.isNotEmpty()) {
                    outputText += ", "
                }
                outputText += "decode: ${String.format("%.1f", decode_tokens_per_s)} tok/s"
            }
            return outputText
        }
    }

    @Serializable
    data class CompletionUsage(
        val prompt_tokens: Int,
        val completion_tokens: Int,
        val total_tokens: Int,
        val extra: CompletionUsageExtra? = null
    )

    @Serializable
    data class StreamOptions(
        val include_usage: Boolean = false
    )

    @Serializable
    data class ChatCompletionStreamResponseChoice(
        var finish_reason: String? = null,
        val index: Int,
        val delta: ChatCompletionMessage,
        var lobprobs: LogProbs? = null
    )

    @Serializable
    data class ChatCompletionStreamResponse(
        val id: String,
        var choices: List<ChatCompletionStreamResponseChoice> = listOf(),
        var created: Int? = null,
        var model: String? = null,
        val system_fingerprint: String,
        var `object`: String? = null,
        val usage: CompletionUsage? = null
    )

    @Serializable
    data class ChatCompletionRequest(
        val messages: List<ChatCompletionMessage>,
        val model: String? = null,
        val frequency_penalty: Float? = null,
        val presence_penalty: Float? = null,
        val logprobs: Boolean = false,
        val top_logprobs: Int = 0,
        val logit_bias: Map<Int, Float>? = null,
        val max_tokens: Int? = null,
        val n: Int = 1,
        val seed: Int? = null,
        val stop: List<String>? = null,
        val stream: Boolean = true,
        val stream_options: StreamOptions? = null,
        val temperature: Float? = null,
        val top_p: Float? = null,
        val tools: List<ChatTool>? = null,
        val user: String? = null,
        val response_format: ResponseFormat? = null
    )

    @Serializable
    data class ResponseFormat(
        val type: String,
        val schema: String? = null
    )
}
