//
//  ChatState.swift
//  LLMChat
//
import Foundation
import MLCSwift
import SwiftUI

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
    case PendingImageUpload
    case ProcessingImage
}

class ChatState : ObservableObject {
    @Published var messages = [MessageData]()
    @Published var infoText = ""
    @Published var displayName = ""
    @Published var modelChatState = ModelChatState.Ready
    @Published var useVision = false
    
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
    
    private func switchToPendingImageUpload() {
        setModelChatState(newModelChatState: .PendingImageUpload)
    }

    private func switchToProcessingImage() {
        setModelChatState(newModelChatState: .ProcessingImage)
    }

    func interruptable() -> Bool {
        return getModelChatState() == .Ready || getModelChatState() == .Generating || getModelChatState() == .Failed || getModelChatState() == .PendingImageUpload
    }
    
    func resettable() -> Bool {
        return getModelChatState() == .Ready || getModelChatState() == .Generating
    }
    
    func chattable() -> Bool {
        return getModelChatState() == .Ready
    }

    func uploadable() -> Bool {
        return getModelChatState() == .PendingImageUpload
    }
    
    private func interruptChat(prologue: () -> Void, epilogue: @escaping () -> Void) {
        assert(interruptable())
        if getModelChatState() == .Ready || getModelChatState() == .Failed || getModelChatState() == .PendingImageUpload {
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
            if useVision {
                backend.resetImageModule()
            }
            DispatchQueue.main.async { [self] in
                clearHistory()
                if useVision {
                    appendMessage(role: .bot, message: "[System] Upload an image to chat")
                    switchToPendingImageUpload()
                } else {
                    switchToReady()
                }
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
            if useVision {
                backend.unloadImageModule()
            }
            backend.unload()
            DispatchQueue.main.async { [self] in
                clearHistory()
                self.localId = ""
                self.modelLib = ""
                self.modelPath = ""
                self.displayName = ""
                self.useVision = false
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
        let prevUseVision = useVision
        self.localId = localId
        self.modelLib = modelLib
        self.modelPath = modelPath
        self.displayName = displayName
        self.useVision = displayName.hasPrefix("minigpt")
        threadWorker.push {[self] in
            DispatchQueue.main.async { [self] in
                appendMessage(role: .bot, message: "[System] Initalize...")
            }
            if prevUseVision {
                backend.unloadImageModule()
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
            if useVision {
                // load vicuna model
                let dir = (modelPath as NSString).deletingLastPathComponent
                let vicunaModelLib = "vicuna-7b-v1.3-q3f16_0"
                let vicunaModelPath = dir + "/" + vicunaModelLib
                let appConfigJsonData = try? JSONSerialization.data(withJSONObject: ["conv_template": "minigpt"], options: [])
                let appConfigJson = String(data: appConfigJsonData!, encoding: .utf8)
                backend.reload(vicunaModelLib, modelPath: vicunaModelPath, appConfigJson: appConfigJson)
                // load image model
                backend.reloadImageModule(modelLib, modelPath: modelPath)
            } else {
                backend.reload(modelLib, modelPath: modelPath, appConfigJson: "")
            }
            DispatchQueue.main.async { [self] in
                if useVision {
                    updateMessage(role: .bot, message: "[System] Upload an image to chat")
                    switchToPendingImageUpload()
                } else {
                    updateMessage(role: .bot, message: "[System] Ready to chat")
                    switchToReady()
                }
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
                let runtimeStats = (backend.runtimeStatsText(useVision))!
                DispatchQueue.main.async { [self] in
                    infoText = runtimeStats
                    switchToReady()
                }
            }
        }
    }

    func requestProcessImage(image: UIImage) {
        assert(getModelChatState() == .PendingImageUpload)
        switchToProcessingImage()
        threadWorker.push {[self] in
            assert(messages.count > 0)
            DispatchQueue.main.async { [self] in
                updateMessage(role: .bot, message: "[System] Processing image")
            }
            // step 1. resize image
            let new_image = resizeImage(image: image, width: 112, height: 112)
            // step 2. prefill image by backend.prefillImage()
            backend.prefillImage(new_image, prevPlaceholder: "<Img>", postPlaceholder: "</Img> ")
            DispatchQueue.main.async { [self] in
                updateMessage(role: .bot, message: "[System] Ready to chat")
                switchToReady()
            }
        }
    }

    func isCurrentModel(localId: String) -> Bool {
        return self.localId == localId
    }
}
