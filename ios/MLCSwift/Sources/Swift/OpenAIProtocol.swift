// Protocol definition of OpenAI API
import Foundation

// Protocols for v1/chat/completions
// API reference: https://platform.openai.com/docs/api-reference/chat/create

public struct TopLogProbs : Codable {
    public let token: String
    public let logprob: Float
    public let bytes: Optional<[Int]>
}

public struct LogProbsContent : Codable {
    public let token: String
    public let logprob: Float
    public var bytes: Optional<[Int]> = nil
    public var top_logprobs: [TopLogProbs] = []
}

public struct LogProbs : Codable {
    public var content: [LogProbsContent] = []
}

public struct ChatFunction : Codable {
    public let name: String
    public var description: Optional<String> = nil
    public let parameters: [String: String]
}

public struct ChatTool : Codable {
    public var type: String = "function"
    public let function: ChatFunction
}

public struct ChatFunctionCall : Codable {
    public let name: String
    // NOTE: arguments shold be dict str to any codable
    // for now only allow string output due to typing issues
    public var arguments: Optional<[String: String]> = nil
}

public struct ChatToolCall : Codable {
    public var id: String = UUID().uuidString
    public var type: String = "function"
    public let function: ChatFunctionCall
}

public struct ChatCompletionMessage : Codable {
    public let role: String
    public var content: Optional<[[String: String]]> = nil
    public var name: Optional<String> = nil
    public var tool_calls: Optional<[ChatToolCall]> = nil
    public var tool_call_id: Optional<String> = nil
}

public struct ChatCompletionStreamResponseChoice: Codable {
    public var finish_reason: Optional<String> = nil
    public let index: Int
    public let delta: ChatCompletionMessage
    public var lobprobs: Optional<LogProbs> = nil
}

public struct ChatCompletionStreamResponse: Codable {
    public let id : String
    public var choices: [ChatCompletionStreamResponseChoice] = []
    public var created: Optional<Int> = nil
    public var model: Optional<String> = nil
    public let system_fingerprint: String
    public var object: Optional<String> = nil
}
