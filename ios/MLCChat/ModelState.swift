//
//  ModelState.swift
//  MLCChat
//

import Foundation

enum ModelInitState {
    case Initializing
    case Indexing
    case Paused
    case Downloading
    case Pausing
    case Verifying
    case Finished
    case Failed
    case Clearing
    case Deleting
}

struct DownloadTask: Hashable {
    let remoteUrl: URL
    let localUrl: URL
}


class ModelState : ObservableObject, Identifiable {
    @Published var modelConfig: ModelConfig!
    @Published var modelInitState: ModelInitState = .Initializing
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
            localId: modelConfig.local_id,
            modelLib: modelConfig.model_lib,
            modelPath: modelDirUrl.path(),
            estimatedVRAMReq: modelConfig.estimated_vram_req ?? 4000000000,
            displayName: modelConfig.display_name ?? modelConfig.local_id.components(separatedBy: "-")[0]
        )
    }
    
    private func switchToInitializing(modelConfig: ModelConfig, modelUrl: URL?, modelDirUrl: URL, startState: StartState, chatState: ChatState) {
        self.modelConfig = modelConfig
        self.modelDirUrl = modelDirUrl
        self.startState = startState
        self.chatState = chatState
        // switchToInitializing should only be called in init
        assert(modelInitState == .Initializing)
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
    
    private func loadParamsConfig() {
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
    
    private func downloadParamsConfig() {
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
    
    private func switchToIndexing() {
        modelInitState = .Indexing
        progress = 0
        total = modelConfig.tokenizer_files.count + paramsConfig.records.count
        
        // collect tokenizer download tasks
        for tokenizerFile in modelConfig.tokenizer_files {
            let remoteUrl = baseRemoteUrl.appending(path: tokenizerFile)
            let localUrl = modelDirUrl.appending(path: tokenizerFile)
         
            if fileManager.fileExists(atPath: localUrl.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteUrl: remoteUrl, localUrl: localUrl))
            }
        }
        
        // collect params download tasks
        for paramsRecord in paramsConfig.records {
            let remoteUrl = baseRemoteUrl.appending(path: paramsRecord.dataPath)
            let localUrl = modelDirUrl.appending(path: paramsRecord.dataPath)

            if fileManager.fileExists(atPath: localUrl.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteUrl: remoteUrl, localUrl: localUrl))
            }
        }
        if progress < total {
            switchToPaused()
        } else {
            switchToFinished()
        }
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
        let task = URLSession.shared.downloadTask(with: downloadTask.remoteUrl) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileUrl = urlOrNil else {
                DispatchQueue.main.async { [self] in
                    handleCancelDownload(downloadTask: downloadTask)
                }
                return
            }
            
            do {
                try self.fileManager.createDirectory(at: downloadTask.localUrl.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? self.fileManager.removeItem(at: downloadTask.localUrl)
                try self.fileManager.moveItem(at: fileUrl, to: downloadTask.localUrl)
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
        assert(modelInitState == .Downloading ||
               modelInitState == .Pausing ||
               modelInitState == .Clearing ||
               modelInitState == .Deleting
        )
        if modelInitState == .Downloading {
            if remainingTasks.isEmpty {
                if downloadingTasks.isEmpty {
                    switchToFinished()
                }
            } else {
                handleNextDownload()
            }
        } else if modelInitState == .Pausing {
            if downloadingTasks.isEmpty {
                switchToPaused()
            }
        } else if modelInitState == .Clearing {
            if downloadingTasks.isEmpty {
                clear()
            }
        } else if modelInitState == .Deleting {
            if downloadingTasks.isEmpty {
                delete()
            }
        }
    }
    
    private func handleCancelDownload(downloadTask: DownloadTask) {
        // withdraw the failed download task
        assert(modelInitState == .Downloading || modelInitState == .Pausing)
        downloadingTasks.remove(downloadTask)
        if modelInitState == .Downloading {
            handleNextDownload()
        } else if modelInitState == .Pausing {
            if downloadingTasks.count == 0 {
                switchToPaused()
            }
        }
    }
    
    private func handleNextDownload() {
        // start next download task
        assert(modelInitState == .Downloading)
        for downloadTask in remainingTasks {
            if !downloadingTasks.contains(downloadTask) {
                handleNewDownload(downloadTask: downloadTask)
                break
            }
        }
    }
    
    private func switchToPaused() {
        modelInitState = .Paused
    }
    
    private func switchToPausing() {
        modelInitState = .Pausing
    }
    
    private func switchToVerifying() {
        modelInitState = .Verifying
        let paramsConfigUrl = modelDirUrl.appending(path: StartState.ParamsConfigFileName)
        if !fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            switchToFailed()
            return
        }
        loadParamsConfig()
        progress = 0
        total = modelConfig.tokenizer_files.count + paramsConfig.records.count
        // verify tokenizer
        for tokenizerFile in modelConfig.tokenizer_files {
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
    
    private func switchToFinished() {
        modelInitState = .Finished
    }
    
    private func switchToFailed() {
        modelInitState = .Failed
    }
    
    private func switchToDownloading() {
        modelInitState = .Downloading
        for downloadTask in remainingTasks {
            if downloadingTasks.count < maxDownloadingTasks {
                handleNewDownload(downloadTask: downloadTask)
            } else {
                return
            }
        }
    }
    
    
    func handleClear() {
        assert(modelInitState == .Downloading || modelInitState == .Paused || modelInitState == .Finished)
        switchToClearing()
    }
    
    func handleDelete() {
        assert(modelInitState == .Downloading || modelInitState == .Paused || modelInitState == .Finished || modelInitState == .Failed)
        switchToDeleting()
    }
    
    private func switchToClearing() {
        if modelInitState == .Paused {
            modelInitState = .Clearing
            clear()
        } else if modelInitState == .Finished {
            if chatState.localId == modelConfig.local_id {
                chatState.requestTerminateChat {
                    self.clear()
                }
            } else {
                clear()
            }
        } else {
            modelInitState = .Clearing
        }
    }
    
    private func switchToDeleting() {
        if modelInitState == .Paused || modelInitState == .Failed {
            modelInitState = .Deleting
            delete()
        } else if modelInitState == .Finished {
            if chatState.localId == modelConfig.local_id {
                chatState.requestTerminateChat {
                    self.delete()
                }
            } else {
                delete()
            }
        } else {
            modelInitState = .Deleting
        }
    }
    
    private func clear() {
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
    
    private func delete() {
        do {
            try fileManager.removeItem(at: modelDirUrl)
            assert(!fileManager.fileExists(atPath: modelDirUrl.path()))
            startState.requestDeleteModel(localId: modelConfig.local_id)
        } catch {
            print(error.localizedDescription)
        }
    }
}
