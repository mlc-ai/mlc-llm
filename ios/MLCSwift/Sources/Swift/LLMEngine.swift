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

    public struct ChatCompletionRequest {
        let model: String
        let messages: [Message]

        struct Message: Codable {
            let role: String        // "user"
            let content: [Content]
            
            struct Content: Codable {
                let type: String    // "text"
                let text: String
            }
        }

        var dictionary: [String: Any] {
            return [
                "model": model,
                "messages": messages.map { message in
                    return [
                        "role": message.role,
                        "content": message.content.map { content in
                            return ["type": content.type, "text": content.text]
                        }
                    ]
                }
            ]
        }
    }

    // TODO(mlc-team) turn into a structured interface
    public func chatCompletion(request: ChatCompletionRequest) -> AsyncStream<ChatCompletionStreamResponse> {
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
            
            // convert to json
            var jsonRequest: String?
            do {
                let jsonData = try JSONSerialization.data(withJSONObject: request.dictionary, options: [])
                jsonRequest = String(data: jsonData, encoding: .utf8)
            } catch {
                print("Error when converting ChatCompletionRequest to JSON: \(error)")
            }
            
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
