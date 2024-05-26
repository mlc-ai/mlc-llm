// Protocol definition of OpenAI API
import Foundation

// Protocols for v1/chat/completions
// API reference: https://platform.openai.com/docs/api-reference/chat/create

public struct TopLogProbs : Codable {
    public var token: String
    public var logprob: Float
    public var bytes: Optional<[Int]>
}

public struct LogProbsContent : Codable {
    public var token: String
    public var logprob: Float
    public var bytes: Optional<[Int]> = nil
    public var top_logprobs: [TopLogProbs] = []
}

public struct LogProbs : Codable {
    public var content: [LogProbsContent] = []
}

public struct ChatFunction : Codable {
    public var name: String
    public var description: Optional<String> = nil
    public var parameters: [String: String]

    public init(
        name: String,
        description: Optional<String> = nil,
        parameters: [String : String]
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct ChatTool : Codable {
    public var type: String = "function"
    public let function: ChatFunction

    public init(type: String, function: ChatFunction) {
        self.type = type
        self.function = function
    }
}

public struct ChatFunctionCall : Codable {
    public var name: String
    // NOTE: arguments shold be dict str to any codable
    // for now only allow string output due to typing issues
    public var arguments: Optional<[String: String]> = nil

    public init(name: String, arguments: Optional<[String : String]> = nil) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ChatToolCall : Codable {
    public var id: String = UUID().uuidString
    public var type: String = "function"
    public var function: ChatFunctionCall

    public init(
        id: String = UUID().uuidString,
        type: String = "function",
        function: ChatFunctionCall
    ) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public enum ChatCompletionRole: String, Codable {
    case system = "system"
    case user = "user"
    case assistant = "assistant"
    case tool = "tool"
}

public enum ChatCompletionMessageContent: Codable {
    case text(String)
    case parts([[String: String]])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            let parts = try container.decode([[String: String]].self)
            self = .parts(parts)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    public func asText() -> String {
        switch (self) {
        case .text(let text): return text
        case .parts(let parts):
            var res = ""
            for item in parts {
                if item["type"]! == "text" {
                    res += item["text"]!
                }
            }
            return res
        }
    }
}

public struct ChatCompletionMessage: Codable {
    public var role: ChatCompletionRole
    public var content: Optional<ChatCompletionMessageContent> = nil
    public var name: Optional<String> = nil
    public var tool_calls: Optional<[ChatToolCall]> = nil
    public var tool_call_id: Optional<String> = nil

    // more complicated content construction
    public init(
        role: ChatCompletionRole,
        content: Optional<[[String : String]]> = nil,
        name: Optional<String> = nil,
        tool_calls: Optional<[ChatToolCall]> = nil,
        tool_call_id: Optional<String> = nil
    ) {
        self.role = role
        if let cvalue = content {
            self.content = .parts(cvalue)
        } else {
            self.content = nil
        }
        self.name = name
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
    }

    // convenient method to construct content from string
    public init(
        role: ChatCompletionRole,
        content: String,
        name: Optional<String> = nil,
        tool_calls: Optional<[ChatToolCall]> = nil,
        tool_call_id: Optional<String> = nil
    ) {
        self.role = role
        self.content = .text(content)
        self.name = name
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
    }
}

public struct ChatCompletionStreamResponseChoice: Codable {
    public var finish_reason: Optional<String> = nil
    public var index: Int
    public var delta: ChatCompletionMessage
    public var lobprobs: Optional<LogProbs> = nil
}

public struct CompletionUsageExtra: Codable {
    public var prefill_tokens_per_s: Optional<Float> = nil
    public var decode_tokens_per_s: Optional<Float> = nil
    public var num_prefill_tokens: Optional<Int> = nil

    public func asTextLabel() -> String {
        var outputText = ""
        if let prefill_tokens_per_s = self.prefill_tokens_per_s {
            outputText += "prefill: "
            outputText += String(format: "%.1f", prefill_tokens_per_s)
            outputText += " tok/s"
        }
        if let decode_tokens_per_s = self.decode_tokens_per_s {
            if !outputText.isEmpty {
                outputText += ", "
            }
            outputText += "decode: "
            outputText += String(format: "%.1f", decode_tokens_per_s)
            outputText += " tok/s"
        }
        return outputText
    }
}

public struct CompletionUsage: Codable {
    public var prompt_tokens: Int
    public var completion_tokens: Int
    public var total_tokens: Int
    public var extra: Optional<CompletionUsageExtra>
}

public struct ChatCompletionStreamResponse: Codable {
    public var id : String
    public var choices: [ChatCompletionStreamResponseChoice] = []
    public var created: Optional<Int> = nil
    public var model: Optional<String> = nil
    public var system_fingerprint: String
    public var object: Optional<String> = nil
    public var usage: Optional<CompletionUsage> = nil
}

public struct ResponseFormat: Codable {
    public var type: String
    public var schema: Optional<String> = nil

    public init(type: String, schema: Optional<String> = nil) {
        self.type = type
        self.schema = schema
    }
}

public struct StreamOptions: Codable {
    public var include_usage: Bool = false

    public init(include_usage: Bool) {
        self.include_usage = include_usage
    }
}

public struct ChatCompletionRequest: Codable {
    public var messages: [ChatCompletionMessage]
    public var model: Optional<String> = nil
    public var frequency_penalty: Optional<Float> = nil
    public var presence_penalty: Optional<Float> = nil
    public var logprobs: Bool = false
    public var top_logprobs: Int = 0
    public var logit_bias: Optional<[Int: Float]> = nil
    public var max_tokens: Optional<Int> = nil
    public var n: Int = 1
    public var seed: Optional<Int> = nil
    public var stop: Optional<[String]> = nil
    public var stream: Bool = true
    public var stream_options: Optional<StreamOptions> = nil
    public var temperature: Optional<Float> = nil
    public var top_p: Optional<Float> = nil
    public var tools: Optional<[ChatTool]> = nil
    public var user: Optional<String> = nil
    public var response_format: Optional<ResponseFormat> = nil

    public init(
        messages: [ChatCompletionMessage],
        model: Optional<String> = nil,
        frequency_penalty: Optional<Float> = nil,
        presence_penalty: Optional<Float> = nil,
        logprobs: Bool = false,
        top_logprobs: Int = 0,
        logit_bias: Optional<[Int : Float]> = nil,
        max_tokens: Optional<Int> = nil,
        n: Int = 1,
        seed: Optional<Int> = nil,
        stop: Optional<[String]> = nil,
        stream: Bool = true,
        stream_options: Optional<StreamOptions> = nil,
        temperature: Optional<Float> = nil,
        top_p: Optional<Float> = nil,
        tools: Optional<[ChatTool]> = nil,
        user: Optional<String> = nil,
        response_format: Optional<ResponseFormat> = nil
    ) {
        self.messages = messages
        self.model = model
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.n = n
        self.seed = seed
        self.stop = stop
        self.stream = stream
        self.stream_options = stream_options
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.user = user
        self.response_format = response_format
    }
}
