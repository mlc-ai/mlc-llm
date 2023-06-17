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

class ChatState : ObservableObject {
    @Published var messages = [MessageData]();
    @Published var infoText = "";
    @Published var modelName = "";
    @Published var inProgress = false;
    @Published var unfinishedRespondRole = MessageRole.bot;
    @Published var unfinishedRespondMessage = "";
    private var threadWorker = ThreadWorker();
    private var backend = ChatModule();

    private var stopLock = NSLock();
    private var requestedReset = false;
    private var stopRequested = false;
    private var reloadReady = false;
    private var modelLib = "";
    private var modelPath = "";

    init() {
        threadWorker.qualityOfService = QualityOfService.userInteractive;
        threadWorker.start()
    }
    
    // reset all chat state
    func mainResetChat() {
        self.messages = [MessageData]()
        self.infoText = ""
        self.unfinishedRespondMessage = ""
        self.inProgress = false;
        self.requestedReset = false;
    }

    func mainReload(modelName: String, modelLib: String, modelPath: String, estimatedVRAMReq : Int64) {
        if (self.reloadReady &&
            self.modelLib == modelLib &&
            self.modelPath == modelPath &&
            self.modelName == modelName) {
            return;
        }
        // request stop regardless of the state
        // to previous action can finish soon
        if (self.inProgress) {
            self.stopLock.lock()
            self.stopRequested = true;
            self.stopLock.unlock()
        }
        self.mainResetChat();
        // we are not reload ready
        self.reloadReady = false;
        self.inProgress = true;
        self.modelName = modelName;
        self.modelLib = modelLib;
        self.modelPath = modelPath;

        threadWorker.push {[self] in
            self.updateReply(role: MessageRole.bot, message: "[System] Initalize...")
            backend.unload();
            let vram = os_proc_available_memory()
            if (vram < estimatedVRAMReq) {
                let reqMem = String (
                    format: "%.1fGB", Double(estimatedVRAMReq) / Double(1 << 20)
                )
                let errMsg = (
                    "Sorry, the system cannot provide " + reqMem + " VRAM as requested to the app, " +
                    "so we cannot initialize this model on this device."
                )
                DispatchQueue.main.sync {
                    self.messages.append(MessageData(role: MessageRole.bot, message: errMsg))
                    self.reloadReady = false
                    self.inProgress = true
                }
                return
            }
            backend.reload(modelLib, modelPath: modelPath)
            self.reloadReady = true
            self.updateReply(role: MessageRole.bot, message: "[System] Ready to chat")
            self.commitReply()
            self.markFinish()
        }
    }

    func dummyGenerate(prompt: String)  {
        threadWorker.push {
            Task {
                self.appendMessage(role: MessageRole.user, message: prompt)
                let testMessage = "I am a friendly bot. Please ask questions."
                var msg = ""
                for _ in stride(from: 0, to: 20, by: 1) {
                    for item in testMessage.split(separator: " ") {
                        do {
                            try await Task.sleep(nanoseconds: 100_000_000)
                        } catch {}
                        msg += " " + item
                        self.updateReply(role: MessageRole.bot, message: msg)
                    }
                    msg += "\n"
                }
                self.commitReply()
                self.reportSpeed(encodingSpeed: 1000, decodingSpeed: 1000)

                self.markFinish()
            }
        }
    }

    func backendGenerate(prompt: String) {
        assert(self.inProgress);
        // generation needs to run on thread worker
        threadWorker.push {[self] in
            self.appendMessage(role: MessageRole.user, message: prompt)

            backend.prefill(prompt);
            while (!backend.stopped()) {
                assert(self.inProgress);
                backend.decode();
                self.updateReply(role: MessageRole.bot, message: backend.getMessage())
                // use lock to pass in signal
                self.stopLock.lock()
                let needStop = self.stopRequested;
                self.stopLock.unlock()
                if (needStop) {
                    let forceStop = !self.reloadReady;
                    // if we are not reload ready
                    // this means we are forced stoped during reload
                    // do not do anything to refresh UX
                    if (forceStop) {
                        return
                    }
                    break;
                }
            }

            self.commitReply()
            let runtimeText: String = self.backend.runtimeStatsText()
            DispatchQueue.main.sync { [runtimeText] in
                self.infoText = runtimeText;
            }

            self.markFinish()
        };
    }

    func generate(prompt: String) {
        if (!self.reloadReady) {
            return
        }
        self.inProgress = true
        self.stopRequested = false
        self.backendGenerate(prompt: prompt)
    }

    func requestStop() {
        if (!self.reloadReady) {
            return
        }
        if (self.inProgress) {
            self.stopLock.lock()
            self.stopRequested = true;
            self.stopLock.unlock()
        }
    }

    func resetChat() {
        if (!self.reloadReady) {
            return
        }
        if (self.inProgress) {
            self.requestStop()
        }
        if (self.requestedReset) {
            return;
        }
        self.requestedReset = true;

        threadWorker.push {
            self.backend.resetChat();
            DispatchQueue.main.sync {
                self.mainResetChat();
            }
        }
    }

    func reportSpeed(encodingSpeed: Float, decodingSpeed: Float) {
        DispatchQueue.main.sync { [self, encodingSpeed, decodingSpeed] in
            self.infoText = String(
                format: "prefill: %.1f tok/s, decode: %.1f tok/s", encodingSpeed, decodingSpeed
            )
        }
    }

    func markFinish() {
        DispatchQueue.main.sync { [self] in
            self.inProgress = false
        }
    }

    func commitReply() {
        DispatchQueue.main.sync { [self] in
            self.messages.append(MessageData(
                role: unfinishedRespondRole, message: unfinishedRespondMessage))
            self.unfinishedRespondMessage = ""
        }
    }

    func updateReply(role: MessageRole, message: String) {
        DispatchQueue.main.sync { [self, role, message] in
            self.unfinishedRespondRole = role
            self.unfinishedRespondMessage = message;
        }
    }

    func appendMessage(role: MessageRole, message: String) {
        DispatchQueue.main.sync { [self, role, message] in
            self.messages.append(MessageData(role: role, message: message))
        }
    }
}
