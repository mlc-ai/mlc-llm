//
//  ModelState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/15/23.
//

import Foundation

enum ModelInitState{
    case Initializing
    case Indexing
    case Paused
    case Downloading
    case Pausing
    case Verifying
    case Finished
    case Failed
}

struct DownloadTask: Hashable {
    let remoteUrl: URL
    let localUrl: URL
}


class ModelState : ObservableObject, Identifiable{
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
    private var baseRemoteUrl: URL!
    public let chatState: ChatState;
    
    init(modelConfig: ModelConfig, modelUrl: URL?, modelDirUrl: URL, chatState: ChatState) {
        self.chatState = chatState
        switchToInitializing(modelConfig: modelConfig, modelUrl: modelUrl, modelDirUrl: modelDirUrl)
    }
    
    func reloadChatStateWithThisModel() {
        // TODO(tvm-team) consider log optional model name
        let estimatedMemReq = modelConfig.estimated_memory_req ?? 4000000000;
        let modelName = modelConfig.display_name ?? modelConfig.local_id.components(separatedBy: "-")[0];
        self.chatState.mainReload(
            modelName: modelName,
            modelLib: modelConfig.model_lib,
            modelPath: modelDirUrl.path(),
            estimatedMemReq: estimatedMemReq)
    }
    
    func switchToInitializing(modelConfig: ModelConfig, modelUrl: URL?, modelDirUrl: URL) {
        self.modelConfig = modelConfig
        self.modelDirUrl = modelDirUrl
        // switchToInitializing should only be called in init
        assert(modelInitState == .Initializing)
        if !fileManager.fileExists(atPath: modelDirUrl.path()) {
            do {
                try fileManager.createDirectory(at: modelDirUrl, withIntermediateDirectories: true)
            } catch {
                print(error.localizedDescription)
            }
        }
        
       
        // remote base url
        baseRemoteUrl = modelUrl
        
        if baseRemoteUrl == nil {
            // verify local model
            switchToVerifying()
            return
        }
        
        // create local params dir
        let paramsConfigUrl = modelDirUrl.appending(path: "ndarray-cache.json")
        
        if fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            // ndarray-cache.json already downloaded
            self.loadParamsConfig()
            switchToIndexing()
        } else {
            // download ndarray-cache.json
            let downloadTask = URLSession.shared.downloadTask(with: baseRemoteUrl.appending(path: "ndarray-cache.json")) {
                urlOrNil, responseOrNil, errorOrNil in
                guard let fileUrl = urlOrNil else { return }
                do {
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
    }
    
    func switchToIndexing() {
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
    
    func loadParamsConfig() {
        let paramsConfigUrl = modelDirUrl.appending(path: "ndarray-cache.json")
        assert(fileManager.fileExists(atPath: paramsConfigUrl.path()))
        do {
            let fileHandle = try FileHandle(forReadingFrom: paramsConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            paramsConfig = try self.decoder.decode(ParamsConfig.self, from: data)
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func handleNewDownload(downloadTask: DownloadTask) {
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
    
    func handleFinishDownload(downloadTask: DownloadTask) {
        // update the finished download task
        remainingTasks.remove(downloadTask)
        downloadingTasks.remove(downloadTask)
        progress += 1
        assert(modelInitState == .Downloading || modelInitState == .Pausing)
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
        }
    }
    
    func handleCancelDownload(downloadTask: DownloadTask) {
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
    
    func handleNextDownload() {
        // start next download task
        assert(modelInitState == .Downloading)
        for downloadTask in remainingTasks {
            if !downloadingTasks.contains(downloadTask) {
                handleNewDownload(downloadTask: downloadTask)
                break
            }
        }
    }
    
    func switchToPaused() {
        modelInitState = .Paused
    }
    
    func switchToPausing() {
        modelInitState = .Pausing
    }
    
    func switchToVerifying() {
        modelInitState = .Verifying
        let paramsConfigUrl = modelDirUrl.appending(path: "ndarray-cache.json")
        if !fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            switchToFailed()
            return
        }
        loadParamsConfig()
        // verify tokenizer
        for tokenizerFile in modelConfig.tokenizer_files {
            let localUrl = modelDirUrl.appending(path: tokenizerFile)
         
            if !fileManager.fileExists(atPath: localUrl.path()) {
                switchToFailed()
                return
            }
                
        }
        
        // verify params
        for paramsRecord in paramsConfig.records {
            let localUrl = modelDirUrl.appending(path: paramsRecord.dataPath)
            
            if !fileManager.fileExists(atPath: localUrl.path()) {
                switchToFailed()
                return
            }
        }
        switchToFinished()
    }
    
    func switchToFinished() {
        modelInitState = .Finished
    }
    
    func switchToFailed() {
        modelInitState = .Failed
    }
    
    func switchToDownloading() {
        modelInitState = .Downloading
        for downloadTask in remainingTasks {
            if downloadingTasks.count < maxDownloadingTasks {
                handleNewDownload(downloadTask: downloadTask)
            } else {
                return
            }
        }
    }
}
