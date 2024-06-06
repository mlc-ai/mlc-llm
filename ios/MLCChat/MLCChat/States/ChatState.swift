//
//  ChatState.swift
//  LLMChat
//

import Foundation
import MLCSwift

enum MessageRole {
    case user
    case assistant
}

extension MessageRole {
    var isUser: Bool { self == .user }
}

struct MessageData: Hashable {
    let id = UUID()
    var role: MessageRole
    var message: String
}

final class ChatState: ObservableObject {
    fileprivate enum ModelChatState {
        case generating
        case resetting
        case reloading
        case terminating
        case ready
        case failed
        case pendingImageUpload
        case processingImage
    }

    @Published var displayMessages = [MessageData]()
    @Published var infoText = ""
    @Published var displayName = ""
    // this is a legacy UI option for upload image
    // TODO(mlc-team) support new UI for image processing
    @Published var legacyUseImage = false

    private let modelChatStateLock = NSLock()
    private var modelChatState: ModelChatState = .ready

    // the new mlc engine
    private let engine = MLCEngine()
    // history messages
    private var historyMessages = [ChatCompletionMessage]()

    // streaming text that get updated
    private var streamingText = ""

    private var modelLib = ""
    private var modelPath = ""
    var modelID = ""

    init() {
    }

    var isInterruptible: Bool {
        return getModelChatState() == .ready
        || getModelChatState() == .generating
        || getModelChatState() == .failed
        || getModelChatState() == .pendingImageUpload
    }

    var isChattable: Bool {
        return getModelChatState() == .ready
    }

    var isUploadable: Bool {
        return getModelChatState() == .pendingImageUpload
    }

    var isResettable: Bool {
        return getModelChatState() == .ready
        || getModelChatState() == .generating
    }

    func requestResetChat() {
        assert(isResettable)
        interruptChat(prologue: {
            switchToResetting()
        }, epilogue: { [weak self] in
            self?.mainResetChat()
        })
    }

    // reset the chat if we switch to background
    // during generation to avoid permission issue
    func requestSwitchToBackground() {
        if (getModelChatState() == .generating) {
            self.requestResetChat()
        }
    }


    func requestTerminateChat(callback: @escaping () -> Void) {
        assert(isInterruptible)
        interruptChat(prologue: {
            switchToTerminating()
        }, epilogue: { [weak self] in
            self?.mainTerminateChat(callback: callback)
        })
    }

    func requestReloadChat(modelID: String, modelLib: String, modelPath: String, estimatedVRAMReq: Int, displayName: String) {
        if (isCurrentModel(modelID: modelID)) {
            return
        }
        assert(isInterruptible)
        interruptChat(prologue: {
            switchToReloading()
        }, epilogue: { [weak self] in
            self?.mainReloadChat(modelID: modelID,
                                 modelLib: modelLib,
                                 modelPath: modelPath,
                                 estimatedVRAMReq: estimatedVRAMReq,
                                 displayName: displayName)
        })
    }


    func requestGenerate(prompt: String) {
        assert(isChattable)
        switchToGenerating()
        appendMessage(role: .user, message: prompt)
        appendMessage(role: .assistant, message: "")

        Task {
            self.historyMessages.append(
                ChatCompletionMessage(role: .user, content: prompt)
            )
            var finishReasonLength = false
            var finalUsageTextLabel = ""

            for await res in await engine.chat.completions.create(
                messages: self.historyMessages,
                stream_options: StreamOptions(include_usage: true)
            ) {
                for choice in res.choices {
                    if let content = choice.delta.content {
                        self.streamingText += content.asText()
                    }
                    if let finish_reason = choice.finish_reason {
                        if finish_reason == "length" {
                            finishReasonLength = true
                        }
                    }
                }
                if let finalUsage = res.usage {
                    finalUsageTextLabel = finalUsage.extra?.asTextLabel() ?? ""
                }
                if getModelChatState() != .generating {
                    break
                }

                var updateText = self.streamingText
                if finishReasonLength {
                    updateText += " [output truncated due to context length limit...]"
                }

                let newText = updateText
                DispatchQueue.main.async {
                    self.updateMessage(role: .assistant, message: newText)
                }
            }

            // record history messages
            if !self.streamingText.isEmpty {
                self.historyMessages.append(
                    ChatCompletionMessage(role: .assistant, content: self.streamingText)
                )
                // stream text can be cleared
                self.streamingText = ""
            } else {
                self.historyMessages.removeLast()
            }

            // if we exceed history
            // we can try to reduce the history and see if it can fit
            if (finishReasonLength) {
                let windowSize = self.historyMessages.count
                assert(windowSize % 2 == 0)
                let removeEnd = ((windowSize + 3) / 4) * 2
                self.historyMessages.removeSubrange(0..<removeEnd)
            }

            if getModelChatState() == .generating {
                let runtimStats = finalUsageTextLabel

                DispatchQueue.main.async {
                    self.infoText = runtimStats
                    self.switchToReady()

                }
            }
        }
    }

    func isCurrentModel(modelID: String) -> Bool {
        return self.modelID == modelID
    }
}

private extension ChatState {
    func getModelChatState() -> ModelChatState {
        modelChatStateLock.lock()
        defer { modelChatStateLock.unlock() }
        return modelChatState
    }

    func setModelChatState(_ newModelChatState: ModelChatState) {
        modelChatStateLock.lock()
        modelChatState = newModelChatState
        modelChatStateLock.unlock()
    }

    func appendMessage(role: MessageRole, message: String) {
        displayMessages.append(MessageData(role: role, message: message))
    }

    func updateMessage(role: MessageRole, message: String) {
        displayMessages[displayMessages.count - 1] = MessageData(role: role, message: message)
    }

    func clearHistory() {
        displayMessages.removeAll()
        infoText = ""
        historyMessages.removeAll()
        streamingText = ""
    }

    func switchToResetting() {
        setModelChatState(.resetting)
    }

    func switchToGenerating() {
        setModelChatState(.generating)
    }

    func switchToReloading() {
        setModelChatState(.reloading)
    }

    func switchToReady() {
        setModelChatState(.ready)
    }

    func switchToTerminating() {
        setModelChatState(.terminating)
    }

    func switchToFailed() {
        setModelChatState(.failed)
    }

    func switchToPendingImageUpload() {
        setModelChatState(.pendingImageUpload)
    }

    func switchToProcessingImage() {
        setModelChatState(.processingImage)
    }

    func interruptChat(prologue: () -> Void, epilogue: @escaping () -> Void) {
        assert(isInterruptible)
        if getModelChatState() == .ready
            || getModelChatState() == .failed
            || getModelChatState() == .pendingImageUpload {
            prologue()
            epilogue()
        } else if getModelChatState() == .generating {
            prologue()
            DispatchQueue.main.async {
                epilogue()
            }
        } else {
            assert(false)
        }
    }

    func mainResetChat() {
        Task {
            await engine.reset()
            self.historyMessages = []
            self.streamingText = ""

            DispatchQueue.main.async {
                self.clearHistory()
                self.switchToReady()
            }
        }
    }

    func mainTerminateChat(callback: @escaping () -> Void) {
        Task {
            await engine.unload()
            DispatchQueue.main.async {
                self.clearHistory()
                self.modelID = ""
                self.modelLib = ""
                self.modelPath = ""
                self.displayName = ""
                self.legacyUseImage = false
                self.switchToReady()
                callback()
            }
        }
    }

    func mainReloadChat(modelID: String, modelLib: String, modelPath: String, estimatedVRAMReq: Int, displayName: String) {
        clearHistory()
        self.modelID = modelID
        self.modelLib = modelLib
        self.modelPath = modelPath
        self.displayName = displayName

        Task {
            DispatchQueue.main.async {
                self.appendMessage(role: .assistant, message: "[System] Initalize...")
            }

            await engine.unload()
            let vRAM = os_proc_available_memory()
            if (vRAM < estimatedVRAMReq) {
                let requiredMemory = String (
                    format: "%.1fMB", Double(estimatedVRAMReq) / Double(1 << 20)
                )
                let errorMessage = (
                    "Sorry, the system cannot provide \(requiredMemory) VRAM as requested to the app, " +
                    "so we cannot initialize this model on this device."
                )
                DispatchQueue.main.sync {
                    self.displayMessages.append(MessageData(role: MessageRole.assistant, message: errorMessage))
                    self.switchToFailed()
                }
                return
            }
            await engine.reload(
                modelPath: modelPath, modelLib: modelLib
            )

            // run a simple prompt with empty content to warm up system prompt
            // helps to start things before user start typing
            for await _ in await engine.chat.completions.create(
                messages: [ChatCompletionMessage(role: .user, content: "")],
                max_tokens: 1
            ) {}

            // TODO(mlc-team) run a system message prefill
            DispatchQueue.main.async {
                self.updateMessage(role: .assistant, message: "[System] Ready to chat")
                self.switchToReady()
            }

        }
    }
}
