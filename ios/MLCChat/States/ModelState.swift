//
//  ModelState.swift
//  MLCChat
//

import Foundation

final class ModelState : ObservableObject, Identifiable {
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

    private struct DownloadTask: Hashable {
        let remoteURL: URL
        let localURL: URL
    }

    @Published var modelConfig: ModelConfig!
    @Published var modelInitState: ModelInitState = .initializing
    @Published var progress: Int = 0
    @Published var total: Int = 1

    private let fileManager: FileManager = FileManager.default
    private let decoder = JSONDecoder()
    private var paramsConfig: ParamsConfig!
    private var modelDirUrl: URL!
    private var remainingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var downloadingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var maxDownloadingTasks: Int = 3
    private var baseRemoteUrl: URL! = nil
    private var chatState: ChatState!
    private var startState: StartState!

    init(modelConfig: ModelConfig, modelUrl: URL?, modelDirUrl: URL, startState: StartState, chatState: ChatState) {
        switchToInitializing(modelConfig: modelConfig, modelUrl: modelUrl, modelDirUrl: modelDirUrl, startState: startState, chatState: chatState)
    }
    
    func startChat(chatState: ChatState) {
        chatState.requestReloadChat(
            localId: modelConfig.localID,
            modelLib: modelConfig.modelLib,
            modelPath: modelDirUrl.path(),
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
    
    private func handleNewDownload(downloadTask: DownloadTask) {
        // start one download task
        assert(downloadingTasks.count < maxDownloadingTasks)
        let task = URLSession.shared.downloadTask(with: downloadTask.remoteURL) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileUrl = urlOrNil else {
                DispatchQueue.main.async { [self] in
                    handleCancelDownload(downloadTask: downloadTask)
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
            DispatchQueue.main.async { [self] in
                handleFinishDownload(downloadTask: downloadTask)
            }
        }
        downloadingTasks.insert(downloadTask)
        task.resume()
    }
    
    private func handleFinishDownload(downloadTask: DownloadTask) {
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
            if remainingTasks.isEmpty {
                if downloadingTasks.isEmpty {
                    switchToFinished()
                }
            } else {
                handleNextDownload()
            }
        } else if modelInitState == .pausing {
            if downloadingTasks.isEmpty {
                switchToPaused()
            }
        } else if modelInitState == .clearing {
            if downloadingTasks.isEmpty {
                clear()
            }
        } else if modelInitState == .deleting {
            if downloadingTasks.isEmpty {
                delete()
            }
        }
    }
    
    private func handleCancelDownload(downloadTask: DownloadTask) {
        // withdraw the failed download task
        assert(modelInitState == .downloading || modelInitState == .pausing)
        downloadingTasks.remove(downloadTask)
        if modelInitState == .downloading {
            handleNextDownload()
        } else if modelInitState == .pausing {
            if downloadingTasks.count == 0 {
                switchToPaused()
            }
        }
    }
    
    private func handleNextDownload() {
        // start next download task
        assert(modelInitState == .downloading)
        for downloadTask in remainingTasks {
            if !downloadingTasks.contains(downloadTask) {
                handleNewDownload(downloadTask: downloadTask)
                break
            }
        }
    }
    
    private func switchToPaused() {
        modelInitState = .paused
    }
    
    private func switchToPausing() {
        modelInitState = .pausing
    }
    
    private func switchToVerifying() {
        modelInitState = .verifying
        let paramsConfigUrl = modelDirUrl.appending(path: StartState.ParamsConfigFileName)
        if !fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            switchToFailed()
            return
        }
        loadParamsConfig()
        progress = 0
        total = modelConfig.tokenizerFiles.count + paramsConfig.records.count
        // verify tokenizer
        for tokenizerFile in modelConfig.tokenizerFiles {
            let localUrl = modelDirUrl.appending(path: tokenizerFile)

            if !fileManager.fileExists(atPath: localUrl.path()) {
                switchToFailed()
                return
            }
            progress += 1

        }
        
        // verify params
        for paramsRecord in paramsConfig.records {
            let localUrl = modelDirUrl.appending(path: paramsRecord.dataPath)
            
            if !fileManager.fileExists(atPath: localUrl.path()) {
                switchToFailed()
                return
            }
            
            progress += 1
        }
        switchToFinished()
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
    func switchToInitializing(modelConfig: ModelConfig, modelUrl: URL?, modelDirUrl: URL, startState: StartState, chatState: ChatState) {
        self.modelConfig = modelConfig
        self.modelDirUrl = modelDirUrl
        self.startState = startState
        self.chatState = chatState
        // switchToInitializing should only be called in init
        assert(modelInitState == .initializing)
        if !fileManager.fileExists(atPath: modelDirUrl.path()) {
            do {
                try fileManager.createDirectory(at: modelDirUrl, withIntermediateDirectories: true)
            } catch {
                print(error.localizedDescription)
            }
        }

        if modelUrl == nil {
            // verify local model
            switchToVerifying()
            return
        } else {
            baseRemoteUrl = modelUrl!.appending(path: "resolve").appending(path: "main")
        }

        // create local params dir
        let paramsConfigUrl = modelDirUrl.appending(path: StartState.ParamsConfigFileName)

        if fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            // ndarray-cache.json already downloaded
            loadParamsConfig()
            switchToIndexing()
        } else {
            // download ndarray-cache.json
            downloadParamsConfig()
        }
    }

    func loadParamsConfig() {
        let paramsConfigUrl = modelDirUrl.appending(path: StartState.ParamsConfigFileName)
        assert(fileManager.fileExists(atPath: paramsConfigUrl.path()))
        do {
            let fileHandle = try FileHandle(forReadingFrom: paramsConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            paramsConfig = try self.decoder.decode(ParamsConfig.self, from: data)
        } catch {
            print(error.localizedDescription)
        }
    }

    func downloadParamsConfig() {
        let paramsConfigUrl = modelDirUrl.appending(path: StartState.ParamsConfigFileName)
        let downloadTask = URLSession.shared.downloadTask(with: baseRemoteUrl.appending(path: StartState.ParamsConfigFileName)) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileUrl = urlOrNil else { return }
            do {
                try? self.fileManager.removeItem(at: paramsConfigUrl)
                try self.fileManager.moveItem(at: fileUrl, to: paramsConfigUrl)
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
        modelInitState = .indexing
        progress = 0
        total = modelConfig.tokenizerFiles.count + paramsConfig.records.count

        // collect tokenizer download tasks
        for tokenizerFile in modelConfig.tokenizerFiles {
            let remoteURL = baseRemoteUrl.appending(path: tokenizerFile)
            let localURL = modelDirUrl.appending(path: tokenizerFile)

            if fileManager.fileExists(atPath: localURL.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteURL: remoteURL, localURL: localURL))
            }
        }

        // collect params download tasks
        for paramsRecord in paramsConfig.records {
            let remoteURL = baseRemoteUrl.appending(path: paramsRecord.dataPath)
            let localURL = modelDirUrl.appending(path: paramsRecord.dataPath)

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

    func switchToClearing() {
        if modelInitState == .paused {
            modelInitState = .clearing
            clear()
        } else if modelInitState == .finished {
            if chatState.localId == modelConfig.localID {
                chatState.requestTerminateChat {
                    self.clear()
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
                chatState.requestTerminateChat {
                    self.delete()
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
            let fileUrls = try fileManager.contentsOfDirectory(at: modelDirUrl, includingPropertiesForKeys: nil)
            for fileUrl in fileUrls where fileUrl.lastPathComponent != StartState.ModelConfigFileName {
                try fileManager.removeItem(at: fileUrl)
                assert(!fileManager.fileExists(atPath: fileUrl.path()))
            }
            assert(fileManager.fileExists(atPath: modelDirUrl.appending(path: StartState.ModelConfigFileName).path()))
            switchToIndexing()
        } catch {
            print(error.localizedDescription)
        }
    }

    func delete() {
        do {
            try fileManager.removeItem(at: modelDirUrl)
            assert(!fileManager.fileExists(atPath: modelDirUrl.path()))
            startState.requestDeleteModel(localId: modelConfig.localID)
        } catch {
            print(error.localizedDescription)
        }
    }
}
