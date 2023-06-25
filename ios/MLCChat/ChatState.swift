//
//  ChatState.swift
//  LLMChat
//
import Foundation
import MLCSwift

enum MessageRole {
    case user
    case bot
}

struct MessageData: Hashable, Identifiable {
    let id = UUID()
    var role: MessageRole
    var message: String
}

enum ModelChatState {
    case Generating
    case Resetting
    case Reloading
    case Terminating
    case Ready
    case Failed
}

class ChatState : ObservableObject {
    @Published var messages = [MessageData]()
    @Published var infoText = ""
    @Published var displayName = ""
    @Published var modelChatState = ModelChatState.Ready
    
    private let modelChatStateLock = NSLock()
    
    private var threadWorker = ThreadWorker()
    private var backend = ChatModule()
    private var modelLib = ""
    private var modelPath = ""
    var localId = ""
    
    init() {
        threadWorker.qualityOfService = QualityOfService.userInteractive
        threadWorker.start()
    }
    
    func getModelChatState() -> ModelChatState {
        modelChatStateLock.lock()
        let currentModelChatState = modelChatState
        modelChatStateLock.unlock()
        return currentModelChatState
    }
    
    func setModelChatState(newModelChatState: ModelChatState) {
        modelChatStateLock.lock()
        modelChatState = newModelChatState
        modelChatStateLock.unlock()
    }
    
    private func appendMessage(role: MessageRole, message: String) {
        messages.append(MessageData(role: role, message: message))
    }
    
    private func updateMessage(role: MessageRole, message: String) {
        messages[messages.count - 1] = MessageData(role: role, message: message)
    }
    
    private func clearHistory() {
        messages.removeAll()
        infoText = ""
    }
    
    private func switchToResetting() {
        setModelChatState(newModelChatState: .Resetting)
    }
    
    private func switchToGenerating() {
        setModelChatState(newModelChatState: .Generating)
    }
    
    private func switchToReloading() {
        setModelChatState(newModelChatState: .Reloading)
    }
    
    private func switchToReady() {
        setModelChatState(newModelChatState: .Ready)
    }
    
    private func switchToTerminating() {
        setModelChatState(newModelChatState: .Terminating)
    }
    
    private func switchToFailed() {
        setModelChatState(newModelChatState: .Failed)
    }
    
    func interruptable() -> Bool {
        return getModelChatState() == .Ready || getModelChatState() == .Generating || getModelChatState() == .Failed
    }
    
    func resettable() -> Bool {
        return getModelChatState() == .Ready || getModelChatState() == .Generating
    }
    
    func chattable() -> Bool {
        return getModelChatState() == .Ready
    }
    
    private func interruptChat(prologue: () -> Void, epilogue: @escaping () -> Void) {
        assert(interruptable())
        if getModelChatState() == .Ready || getModelChatState() == .Failed {
            prologue()
            epilogue()
        } else if getModelChatState() == .Generating {
            prologue()
            threadWorker.push {
                DispatchQueue.main.async {
                    epilogue()
                }
            }
        } else {
            assert(false)
        }
    }
    
    private func mainResetChat() {
        threadWorker.push {[self] in
            backend.resetChat()
            DispatchQueue.main.async { [self] in
                clearHistory()
                switchToReady()
            }
        }
    }
    
    func requestResetChat() {
        assert(resettable())
        interruptChat(prologue: {
            switchToResetting()
        }, epilogue: { [self] in
            mainResetChat()
        })
    }
    
    private func mainTerminateChat(callback: @escaping () -> Void) {
        threadWorker.push { [self] in
            backend.unload()
            DispatchQueue.main.async { [self] in
                clearHistory()
                self.localId = ""
                self.modelLib = ""
                self.modelPath = ""
                self.displayName = ""
                switchToReady()
                callback()
            }
        }
    }
    
    func requestTerminateChat(callback: @escaping () -> Void) {
        assert(interruptable())
        interruptChat(prologue: {
            switchToTerminating()
        }, epilogue: { [self] in
            mainTerminateChat(callback: callback)
        })
    }
    
    private func mainReloadChat(localId: String, modelLib: String, modelPath: String, estimatedVRAMReq: Int64, displayName: String) {
        clearHistory()
        self.localId = localId
        self.modelLib = modelLib
        self.modelPath = modelPath
        self.displayName = displayName
        threadWorker.push {[self] in
            DispatchQueue.main.async { [self] in
                appendMessage(role: .bot, message: "[System] Initalize...")
            }
            backend.unload()
            let vram = os_proc_available_memory()
            if (vram < estimatedVRAMReq) {
                let reqMem = String (
                    format: "%.1fMB", Double(estimatedVRAMReq) / Double(1 << 20)
                )
                let errMsg = (
                    "Sorry, the system cannot provide " + reqMem + " VRAM as requested to the app, " +
                    "so we cannot initialize this model on this device."
                )
                DispatchQueue.main.sync {
                    self.messages.append(MessageData(role: MessageRole.bot, message: errMsg))
                    self.switchToFailed()
                }
                return
            }
            backend.reload(modelLib, modelPath: modelPath)
            DispatchQueue.main.async { [self] in
                updateMessage(role: .bot, message: "[System] Ready to chat")
                switchToReady()
            }
        }
    }
    
    func requestReloadChat(localId: String, modelLib: String, modelPath: String, estimatedVRAMReq: Int64, displayName: String) {
        if (isCurrentModel(localId: localId)) {
            return
        }
        assert(interruptable())
        interruptChat(prologue: {
            switchToReloading()
        }, epilogue: { [self] in
            mainReloadChat(localId: localId, modelLib: modelLib, modelPath: modelPath, estimatedVRAMReq: estimatedVRAMReq, displayName: displayName)
        })
    }
    
    func requestGenerate(prompt: String) {
        assert(chattable())
        switchToGenerating()
        appendMessage(role: .user, message: prompt)
        appendMessage(role: .bot, message: "")
        threadWorker.push {[self] in
            backend.prefill(prompt)
            while !backend.stopped() {
                backend.decode()
                let newText = backend.getMessage()
                DispatchQueue.main.async { [self] in
                    updateMessage(role: .bot, message: newText!)
                }
                if getModelChatState() != .Generating {
                    break
                }
            }
            if getModelChatState() == .Generating {
                let runtimeStats = backend.runtimeStatsText()
                DispatchQueue.main.async { [self] in
                    infoText = runtimeStats!
                    switchToReady()
                }
            }
        }
    }
    
    func isCurrentModel(localId: String) -> Bool {
        return self.localId == localId
    }
}
