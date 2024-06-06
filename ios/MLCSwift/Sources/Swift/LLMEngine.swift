import Foundation
import MLCEngineObjC
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
public class MLCEngine {
    struct RequestState {
        let request: ChatCompletionRequest
        let continuation: AsyncStream<ChatCompletionStreamResponse>.Continuation

        init(
            request: ChatCompletionRequest,
            continuation: AsyncStream<ChatCompletionStreamResponse>.Continuation
        ) {
            self.request = request
            self.continuation = continuation
        }
    }

    // internal engine state
    // that maintains logger and continuations
    // we decouple it from MLCEngine
    // and explicitly pass in jsonFFIEngine
    // so there is no cyclic dependency
    // when we capture things
    actor EngineState {
        public let logger = Logger()
        private var requestStateMap = Dictionary<String, RequestState>()

        // completion function
        func chatCompletion(
            jsonFFIEngine: JSONFFIEngine,
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
                        jsonFFIEngine.abort(requestID);
                    }
                }
                // store continuation map for further callbacks
                self.requestStateMap[requestID] = RequestState(
                    request: request, continuation: continuation
                )
                // start invoking engine for completion
                jsonFFIEngine.chatCompletion(jsonRequest, requestID: requestID)
            }
            return stream
        }

        func streamCallback(result: String?) {
            var responses: [ChatCompletionStreamResponse] = []

            let decoder = JSONDecoder()
            do {
                responses = try decoder.decode([ChatCompletionStreamResponse].self, from: result!.data(using: .utf8)!)
            } catch let lastError {
                logger.error("Swift json parsing error: error=\(lastError), jsonsrc=\(result!)")
             }

            // dispatch to right request ID
            for res in responses {
                if let requestState = self.requestStateMap[res.id] {
                    // final chunk always come with usage
                    if let finalUsage = res.usage {
                        if let include_usage = requestState.request.stream_options?.include_usage {
                            if include_usage {
                                requestState.continuation.yield(res)
                            }
                        }
                        requestState.continuation.finish()
                        self.requestStateMap.removeValue(forKey: res.id)
                    } else {
                        requestState.continuation.yield(res)
                    }
                }
            }
            // Todo(mlc-team): check the last error in engine and report if there's any
        }
    }

    public class Completions {
        private let jsonFFIEngine: JSONFFIEngine
        private let state: EngineState

        init(jsonFFIEngine: JSONFFIEngine, state: EngineState) {
            self.jsonFFIEngine = jsonFFIEngine
            self.state = state
        }

        private func create(
            request: ChatCompletionRequest
        ) async -> AsyncStream<ChatCompletionStreamResponse> {
            return await state.chatCompletion(jsonFFIEngine: jsonFFIEngine, request: request)
        }

        // offer a direct convenient method to pass in messages
        public func create(
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
        ) async -> AsyncStream<ChatCompletionStreamResponse> {
            if !stream {
                state.logger.error("Only stream=true is supported in MLCSwift")
            }
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
                stream_options: stream_options,
                temperature: temperature,
                top_p: top_p,
                tools: tools,
                user: user,
                response_format: response_format
            )
            return await self.create(request: request)
        }
    }

    public class Chat {
        public let completions: Completions

        init(jsonFFIEngine: JSONFFIEngine, state: EngineState) {
            self.completions = Completions(
                jsonFFIEngine: jsonFFIEngine,
                state: state
            )
        }
    }

    private let state : EngineState;
    private let jsonFFIEngine: JSONFFIEngine;
    public let chat : Chat;
    private var threads = Array<Thread>();

    public init() {
        let state_ = EngineState();
        let jsonFFIEngine_ = JSONFFIEngine();

        self.chat = Chat(jsonFFIEngine: jsonFFIEngine_, state: state_)
        self.jsonFFIEngine = jsonFFIEngine_
        self.state = state_

        // note: closure do not capture self
        jsonFFIEngine_.initBackgroundEngine {
            [state_](result : String?) -> Void in
            state_.streamCallback(result: result)
        }
        let backgroundWorker = BackgroundWorker { [jsonFFIEngine_] in
            Thread.setThreadPriority(1)
            jsonFFIEngine_.runBackgroundLoop()
        }
        let backgroundStreamBackWorker = BackgroundWorker {
            [jsonFFIEngine_] in
            jsonFFIEngine_.runBackgroundStreamBackLoop()
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

    // The following functions do not have to be async for now
    // But to be safe and consistent with chat.completions.create
    // and for future API changes we keep them as async calls
    public func reload(modelPath: String, modelLib: String) async {
        let engineConfig = """
        {
            "model": "\(modelPath)",
            "model_lib": "system://\(modelLib)",
            "mode": "interactive"
        }
        """
        jsonFFIEngine.reload(engineConfig)
    }

    public func reset() async {
        jsonFFIEngine.reset()
    }

    public func unload() async {
        jsonFFIEngine.unload()
    }
}
