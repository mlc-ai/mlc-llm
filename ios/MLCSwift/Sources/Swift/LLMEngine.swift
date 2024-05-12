import Foundation
import LLMChatObjC
import os

class BackgroundWorker : Thread {
    private var task: ()->Void;

    public init(task: @escaping () -> Void) {
        self.task = task
    }

    public override func main()  {
        self.task();
    }
}

@available(iOS 14.0.0, *)
public actor MLCEngine {
    private let jsonFFIEngine = JSONFFIEngine()
    private var threads = Array<Thread>();
    private var continuationMap = Dictionary<String, AsyncStream<ChatCompletionStreamResponse>.Continuation>()
    private let logger = Logger()


    public init() {
        jsonFFIEngine.initBackgroundEngine { (result : String?) -> Void in
            self.streamCallback(result: result)
        }
        // startup background threads with
        let backgroundWorker = BackgroundWorker {
            Thread.setThreadPriority(1)
            self.jsonFFIEngine.runBackgroundLoop()
        }
        let backgroundStreamBackWorker = BackgroundWorker {
            self.jsonFFIEngine.runBackgroundStreamBackLoop()
        }
        // set background worker to be high QoS so it gets higher p for gpu
        backgroundWorker.qualityOfService = QualityOfService.userInteractive
        threads.append(backgroundWorker)
        threads.append(backgroundStreamBackWorker)
        backgroundWorker.start()
        backgroundStreamBackWorker.start()
    }

    deinit {
        jsonFFIEngine.exitBackgroundLoop()
    }

    public func reload(modelPath: String, modelLib: String) {
        let engineConfig = """
        {
            "model": "\(modelPath)",
            "model_lib": "system://\(modelLib)",
            "mode": "interactive"
        }
        """
        jsonFFIEngine.reload(engineConfig)
    }

    public func unload() {
        jsonFFIEngine.unload()
    }

    // offer a direct convenient method to pass in messages
    public func chatCompletion(
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
        stream: Bool = false,
        temperature: Optional<Float> = nil,
        top_p: Optional<Float> = nil,
        tools: Optional<[ChatTool]> = nil,
        user: Optional<String> = nil,
        response_format: Optional<ResponseFormat> = nil
    ) -> AsyncStream<ChatCompletionStreamResponse> {
        let request = ChatCompletionRequest(
            messages: messages,
            model: model,
            frequency_penalty: frequency_penalty,
            presence_penalty: presence_penalty,
            logprobs: logprobs,
            top_logprobs: top_logprobs,
            logit_bias: logit_bias,
            max_tokens: max_tokens,
            n: n,
            seed: seed,
            stop: stop,
            stream: stream,
            temperature: temperature,
            top_p: top_p,
            tools: tools,
            user: user,
            response_format: response_format
        )
        return self.chatCompletion(request: request)
    }

    // completion function
    public func chatCompletion(
        request: ChatCompletionRequest
    ) -> AsyncStream<ChatCompletionStreamResponse> {
        let encoder = JSONEncoder()
        let data = try! encoder.encode(request)
        let jsonRequest = String(data: data, encoding: .utf8)!

        // generate a UUID for the request
        let requestID = UUID().uuidString
        let stream = AsyncStream(ChatCompletionStreamResponse.self) { continuation in
            continuation.onTermination = { termination in
                if termination == .cancelled {
                    self.jsonFFIEngine.abort(requestID);
                }
            }
            // store continuation map for further callbacks
            self.continuationMap[requestID] = continuation
            // start invoking engine for completion
            self.jsonFFIEngine.chatCompletion(jsonRequest, requestID: requestID)
        }
        return stream
    }

    private func streamCallback(result: String?) {
        var responses: [ChatCompletionStreamResponse] = []

        let decoder = JSONDecoder()
        do {
            let msg = try decoder.decode(ChatCompletionStreamResponse.self, from: result!.data(using: .utf8)!)
            responses.append(msg)
        } catch let lastError {
            logger.error("Swift json parsing error: error=\(lastError), jsonsrc=\(result!)")
         }

        // dispatch to right request ID
        for res in responses {
            if let continuation = self.continuationMap[res.id] {
                continuation.yield(res)
                // detect finished from result
                var finished = false
                for choice in res.choices {
                    if choice.finish_reason != "" && choice.finish_reason != nil {
                        finished = true;
                    }
                }
                if finished {
                    continuation.finish()
                    self.continuationMap.removeValue(forKey: res.id)
                }
            }
        }
    }
}
