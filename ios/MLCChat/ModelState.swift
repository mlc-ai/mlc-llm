//
//  ModelState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/15/23.
//

import Foundation

enum ModelInitState{
    case Initializing
    case Stopped
    case Downloading
    case Stopping
    case Finished
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
    private var paramsDirUrl: URL!
    private var remainingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var downloadingTasks: Set<DownloadTask> = Set<DownloadTask>()
    private var maxDownloadingTasks: Int = 3
    private var baseRemoteUrl: URL!
    
    
    init(modelConfig: ModelConfig, modelDirUrl: URL) {
        print(modelConfig)
        self.modelConfig = modelConfig
        self.modelDirUrl = modelDirUrl
        if !fileManager.fileExists(atPath: modelDirUrl.path()) {
            do {
                try fileManager.createDirectory(at: modelDirUrl, withIntermediateDirectories: true)
            } catch {
                print(error.localizedDescription)
            }
        }
        
       
        // remote base url
        baseRemoteUrl = URL(string: modelConfig.model_url)
        
        // create local params dir
        let paramsConfigRemoteUrl = baseRemoteUrl.appending(path: modelConfig.ndarray_file)
        paramsDirUrl = modelDirUrl.appending(path: modelConfig.ndarray_file).deletingLastPathComponent()
        do {
            try fileManager.createDirectory(at: paramsDirUrl, withIntermediateDirectories: true)
        } catch {
            print(error.localizedDescription)
        }
        let paramsConfigUrl = modelDirUrl.appending(path: modelConfig.ndarray_file)
        
        if fileManager.fileExists(atPath: paramsConfigUrl.path()) {
            // ndarray-cache.json already downloaded
            do {
                let fileHandle = try FileHandle(forReadingFrom: paramsConfigUrl)
                let data = fileHandle.readDataToEndOfFile()
                paramsConfig = try self.decoder.decode(ParamsConfig.self, from: data)
            } catch {
                print(error.localizedDescription)
            }
            print("ndarray-cache.json exists")
            prepareDownload()
        } else {
            // download ndarray-cache.json
            let downloadTask = URLSession.shared.downloadTask(with: paramsConfigRemoteUrl) {
                urlOrNil, responseOrNil, errorOrNil in
                guard let fileUrl = urlOrNil else { return }
                do {
                    print("ndarray-cache.json downloaded")
                    try self.fileManager.moveItem(at: fileUrl, to: paramsConfigUrl)
                    let fileHandle = try FileHandle(forReadingFrom: paramsConfigUrl)
                    let data = fileHandle.readDataToEndOfFile()
                    let paramsConfig = try self.decoder.decode(ParamsConfig.self, from: data)
                    DispatchQueue.main.async {
                        self.paramsConfig = paramsConfig
                        self.prepareDownload()
                    }
                    
                } catch {
                    print(error.localizedDescription)
                }
            }
            downloadTask.resume()
            print("ndarray-cache.json downloading")
        }
    }
    
    func prepareDownload(){
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
        let baseParamsRemoteUrl = baseRemoteUrl.appending(path: modelConfig.ndarray_file).deletingLastPathComponent()
        for paramsRecord in paramsConfig.records {
            let remoteUrl = baseParamsRemoteUrl.appending(path: paramsRecord.dataPath)
            let localUrl = paramsDirUrl.appending(path: paramsRecord.dataPath)
            
            if fileManager.fileExists(atPath: localUrl.path()) {
                progress += 1
            } else {
                remainingTasks.insert(DownloadTask(remoteUrl: remoteUrl, localUrl: localUrl))
            }
        }
        modelInitState = progress < total ? .Stopped : .Finished
    }
    
    func start() {
        // start downloading
        modelInitState = .Downloading
        for downloadTask in remainingTasks {
            if downloadingTasks.count < maxDownloadingTasks {
                startDownload(downloadTask: downloadTask)
            } else {
                return
            }
        }
    }
    
    func stop() {
        // stop downloading
        modelInitState = .Stopping
    }
    
    func startDownload(downloadTask: DownloadTask) {
        // start one download task
        assert(downloadingTasks.count < maxDownloadingTasks)
        let task = URLSession.shared.downloadTask(with: downloadTask.remoteUrl) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileUrl = urlOrNil else {
                DispatchQueue.main.async { [self] in
                    cancelDownload(downloadTask: downloadTask)
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
                finishDownload(downloadTask: downloadTask)
            }
        }
        downloadingTasks.insert(downloadTask)
        task.resume()
    }
    
    func finishDownload(downloadTask: DownloadTask) {
        // update the finished download task
        remainingTasks.remove(downloadTask)
        downloadingTasks.remove(downloadTask)
        progress += 1
        assert(modelInitState == .Downloading || modelInitState == .Stopping)
        if modelInitState == .Downloading {
            if remainingTasks.isEmpty {
                if downloadingTasks.isEmpty {
                    modelInitState = .Finished
                }
            } else {
                nextDownload()
            }
        } else if modelInitState == .Stopping {
            if downloadingTasks.isEmpty {
                modelInitState = .Stopped
            }
        }
    }
    
    func cancelDownload(downloadTask: DownloadTask) {
        // withdraw the failed download task
        assert(modelInitState == .Downloading || modelInitState == .Stopping)
        downloadingTasks.remove(downloadTask)
        if modelInitState == .Downloading {
            nextDownload()
        } else if modelInitState == .Stopping {
            if downloadingTasks.count == 0 {
                modelInitState = .Stopped
            }
        }
    }
    
    func nextDownload() {
        // start next download task
        assert(modelInitState == .Downloading)
        for downloadTask in remainingTasks {
            if !downloadingTasks.contains(downloadTask) {
                startDownload(downloadTask: downloadTask)
                break
            }
        }
    }
}
