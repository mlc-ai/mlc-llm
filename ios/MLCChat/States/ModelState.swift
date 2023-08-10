//
//  ModelState.swift
//  MLCChat
//

import Foundation

final class ModelState: ObservableObject, Identifiable {
    enum ModelInitState {
        case initializing
        case indexing
        case paused
        case downloading
        case pausing
        case verifying
        case finished
        case failed
        case clearing
        case deleting
    }

    fileprivate struct DownloadTask: Hashable {
        let remoteURL: URL
        let localURL: URL
    }

    @Published var modelConfig: ModelConfig
    @Published var modelInitState: ModelInitState = .initializing
    @Published var progress: Int = 0
    @Published var total: Int = 1

    private var modelLocalBaseURL: URL
    private var startState: StartState
    private var chatState: ChatState

    private let fileManager: FileManager = FileManager.default
    private let decoder = JSONDecoder()
    private var paramsConfig: ParamsConfig?
    private var modelRemoteBaseURL: URL?
    private var remainingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var downloadingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var maxDownloadingTasks: Int = 3

    init(modelConfig: ModelConfig,
         modelLocalBaseURL: URL,
         startState: StartState,
         chatState: ChatState) {
        self.modelConfig = modelConfig
        self.modelLocalBaseURL = modelLocalBaseURL
        self.startState = startState
        self.chatState = chatState
    }

    func loadModel(modelURL: URL?) {
        createModelFolderIfNeeded()

        guard let modelURL else {
            switchToVerifying()
            return
        }

        modelRemoteBaseURL = modelURL.appending(path: "resolve").appending(path: "main")

        // create local params dir
        let paramsConfigURL = modelLocalBaseURL.appending(path: StartState.ParamsConfigFileName)
        if fileManager.fileExists(atPath: paramsConfigURL.path()) {
            // ndarray-cache.json already downloaded
            loadParamsConfig()
            switchToIndexing()
        } else {
            // download ndarray-cache.json
            downloadParamsConfig()
        }
    }

    func startChat(chatState: ChatState) {
        chatState.requestReloadChat(
            localId: modelConfig.localID,
            modelLib: modelConfig.modelLib,
            modelPath: modelLocalBaseURL.path(),
            estimatedVRAMReq: modelConfig.estimatedVRAMReq,
            displayName: modelConfig.displayName
        )
    }
    
    func handleStart() {
        // start downloading
        switchToDownloading()
    }
    
    func handlePause() {
        // pause downloading
        switchToPausing()
    }
    
    func handleClear() {
        assert(modelInitState == .downloading || modelInitState == .paused || modelInitState == .finished)
        switchToClearing()
    }
    
    func handleDelete() {
        assert(modelInitState == .downloading || modelInitState == .paused || modelInitState == .finished || modelInitState == .failed)
        switchToDeleting()
    }
}

private extension ModelState {
    func createModelFolderIfNeeded() {
        if !fileManager.fileExists(atPath: modelLocalBaseURL.path()) {
            do {
                try fileManager.createDirectory(at: modelLocalBaseURL, withIntermediateDirectories: true)
            } catch {
                print(error.localizedDescription)
            }
        }
    }

    func loadParamsConfig() {
        let paramsConfigURL = modelLocalBaseURL.appending(path: StartState.ParamsConfigFileName)
        assert(fileManager.fileExists(atPath: paramsConfigURL.path()))
        do {
            let fileHandle = try FileHandle(forReadingFrom: paramsConfigURL)
            let data = fileHandle.readDataToEndOfFile()
            paramsConfig = try self.decoder.decode(ParamsConfig.self, from: data)
        } catch {
            print(error.localizedDescription)
        }
    }

    func downloadParamsConfig() {
        guard let modelRemoteBaseURL else {
            return
        }

        let paramsConfigURL = modelLocalBaseURL.appending(path: StartState.ParamsConfigFileName)
        let downloadTask = URLSession.shared.downloadTask(with: modelRemoteBaseURL.appending(path: StartState.ParamsConfigFileName)) {
            [weak self] urlOrNil, responseOrNil, errorOrNil in
            guard let self else { return }
            guard let fileURL = urlOrNil else { return }
            do {
                try? self.fileManager.removeItem(at: paramsConfigURL)
                try self.fileManager.moveItem(at: fileURL, to: paramsConfigURL)
                DispatchQueue.main.async {
                    self.loadParamsConfig()
                    self.switchToIndexing()
                }
            } catch {
                print(error.localizedDescription)
            }
        }
        downloadTask.resume()
    }

    func switchToIndexing() {
        guard let paramsConfig, let modelRemoteBaseURL else {
            return
        }

        modelInitState = .indexing
        progress = 0
        total = modelConfig.tokenizerFiles.count + paramsConfig.records.count

        // collect tokenizer download tasks
        for tokenizerFile in modelConfig.tokenizerFiles {
            let remoteURL = modelRemoteBaseURL.appending(path: tokenizerFile)
            let localURL = modelLocalBaseURL.appending(path: tokenizerFile)

            if fileManager.fileExists(atPath: localURL.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteURL: remoteURL, localURL: localURL))
            }
        }

        // collect params download tasks
        for paramsRecord in paramsConfig.records {
            let remoteURL = modelRemoteBaseURL.appending(path: paramsRecord.dataPath)
            let localURL = modelLocalBaseURL.appending(path: paramsRecord.dataPath)

            if fileManager.fileExists(atPath: localURL.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteURL: remoteURL, localURL: localURL))
            }
        }

        if progress < total {
            switchToPaused()
        } else {
            switchToFinished()
        }
    }

    func handleNewDownload(downloadTask: DownloadTask) {
        // start one download task
        assert(downloadingTasks.count < maxDownloadingTasks)
        let task = URLSession.shared.downloadTask(with: downloadTask.remoteURL) {
            [weak self] urlOrNil, responseOrNil, errorOrNil in
            guard let self else { return }
            guard let fileUrl = urlOrNil else {
                DispatchQueue.main.async {
                    self.handleCancelDownload(downloadTask: downloadTask)
                }
                return
            }

            do {
                try self.fileManager.createDirectory(at: downloadTask.localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? self.fileManager.removeItem(at: downloadTask.localURL)
                try self.fileManager.moveItem(at: fileUrl, to: downloadTask.localURL)
            } catch {
                print(error.localizedDescription)
            }
            DispatchQueue.main.async {
                self.handleFinishDownload(downloadTask: downloadTask)
            }
        }
        downloadingTasks.insert(downloadTask)
        task.resume()
    }

    func handleFinishDownload(downloadTask: DownloadTask) {
        // update the finished download task
        remainingTasks.remove(downloadTask)
        downloadingTasks.remove(downloadTask)
        progress += 1
        assert(modelInitState == .downloading ||
               modelInitState == .pausing ||
               modelInitState == .clearing ||
               modelInitState == .deleting
        )
        if modelInitState == .downloading {
            if remainingTasks.isEmpty && downloadingTasks.isEmpty {
                switchToFinished()
            } else {
                handleNextDownload()
            }
        } else if modelInitState == .pausing && downloadingTasks.isEmpty {
            switchToPaused()
        } else if modelInitState == .clearing && downloadingTasks.isEmpty {
            clear()
        } else if modelInitState == .deleting && downloadingTasks.isEmpty {
            delete()
        }
    }

    func handleCancelDownload(downloadTask: DownloadTask) {
        // withdraw the failed download task
        assert(modelInitState == .downloading || modelInitState == .pausing)
        downloadingTasks.remove(downloadTask)
        if modelInitState == .downloading {
            handleNextDownload()
        } else if modelInitState == .pausing && downloadingTasks.count == 0 {
            switchToPaused()
        }
    }

    func handleNextDownload() {
        // start next download task
        assert(modelInitState == .downloading)
        for downloadTask in remainingTasks {
            if !downloadingTasks.contains(downloadTask) {
                handleNewDownload(downloadTask: downloadTask)
                break
            }
        }
    }

    func switchToPaused() {
        modelInitState = .paused
    }

    func switchToPausing() {
        modelInitState = .pausing
    }

    func switchToVerifying() {
        modelInitState = .verifying

        let paramsConfigURL = modelLocalBaseURL.appending(path: StartState.ParamsConfigFileName)
        guard fileManager.fileExists(atPath: paramsConfigURL.path()) else {
            switchToFailed()
            return
        }

        loadParamsConfig()
        guard let paramsConfig else {
            switchToFailed()
            return
        }
        progress = 0
        total = modelConfig.tokenizerFiles.count + paramsConfig.records.count

        if !verifyTokenizers() {
            switchToFailed()
            return
        }

        if !verifyParams() {
            switchToFailed()
            return
        }

        switchToFinished()
    }

    func verifyTokenizers() -> Bool {
        for tokenizerFile in modelConfig.tokenizerFiles {
            let localURL = modelLocalBaseURL.appending(path: tokenizerFile)

            if !fileManager.fileExists(atPath: localURL.path()) {
                switchToFailed()
                return false
            }
            progress += 1
        }
        return true
    }

    func verifyParams() -> Bool {
        guard let paramsConfig else {
            return false
        }

        for paramsRecord in paramsConfig.records {
            let localUrl = modelLocalBaseURL.appending(path: paramsRecord.dataPath)

            if !fileManager.fileExists(atPath: localUrl.path()) {
                switchToFailed()
                return false
            }

            progress += 1
        }
        return true
    }

    func switchToClearing() {
        if modelInitState == .paused {
            modelInitState = .clearing
            clear()
        } else if modelInitState == .finished {
            if chatState.localId == modelConfig.localID {
                chatState.requestTerminateChat { [weak self] in
                    self?.clear()
                }
            } else {
                clear()
            }
        } else {
            modelInitState = .clearing
        }
    }

    func switchToDeleting() {
        if modelInitState == .paused || modelInitState == .failed {
            modelInitState = .deleting
            delete()
        } else if modelInitState == .finished {
            if chatState.localId == modelConfig.localID {
                chatState.requestTerminateChat { [weak self] in
                    self?.delete()
                }
            } else {
                delete()
            }
        } else {
            modelInitState = .deleting
        }
    }

    func switchToFinished() {
        modelInitState = .finished
    }

    func switchToFailed() {
        modelInitState = .failed
    }

    func switchToDownloading() {
        modelInitState = .downloading
        for downloadTask in remainingTasks {
            if downloadingTasks.count < maxDownloadingTasks {
                handleNewDownload(downloadTask: downloadTask)
            } else {
                return
            }
        }
    }

    func clear() {
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: modelLocalBaseURL, includingPropertiesForKeys: nil)
            for fileURL in fileURLs where fileURL.lastPathComponent != StartState.ModelConfigFileName {
                try fileManager.removeItem(at: fileURL)
                assert(!fileManager.fileExists(atPath: fileURL.path()))
            }
            assert(fileManager.fileExists(atPath: modelLocalBaseURL.appending(path: StartState.ModelConfigFileName).path()))
            switchToIndexing()
        } catch {
            print(error.localizedDescription)
        }
    }

    func delete() {
        do {
            try fileManager.removeItem(at: modelLocalBaseURL)
            assert(!fileManager.fileExists(atPath: modelLocalBaseURL.path()))
            startState.requestDeleteModel(localId: modelConfig.localID)
        } catch {
            print(error.localizedDescription)
        }
    }
}
