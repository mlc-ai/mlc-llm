//
//  StartState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/13/23.
//

import Foundation


class StartState : ObservableObject {
    @Published var models = [ModelState]()
    private var appConfig: AppConfig!
    private let chatState = ChatState()
    private var cacheDirUrl: URL!
    private let fileManager: FileManager = FileManager.default
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private var prebuiltLocals = Set<String>();
    
    init() {
        let bundleUrl = Bundle.main.bundleURL
        // models in dist
        do {
            let distDirUrl = bundleUrl.appending(path: "dist")
            let contents = try fileManager.contentsOfDirectory(at: distDirUrl, includingPropertiesForKeys: nil)
            let modelDirs = contents.filter{$0.hasDirectoryPath}
            for modelDir in modelDirs {
                let modelConfigUrl = modelDir.appending(path: "mlc-chat-config.json")
                if fileManager.fileExists(atPath: modelConfigUrl.path()) {
                    let fileHandle = try FileHandle(forReadingFrom: modelConfigUrl)
                    let data = fileHandle.readDataToEndOfFile()
                    let modelConfig = try decoder.decode(ModelConfig.self, from: data)
                    assert(modelDir.lastPathComponent == modelConfig.local_id)
                    models.append(
                        ModelState(modelConfig: modelConfig, modelUrl: nil, modelDirUrl: modelDir, chatState: chatState))
                    prebuiltLocals.insert(modelConfig.local_id)
                }
            }
        } catch {
            print(error.localizedDescription)
        }
        
        // models in cache to download
        do {
            cacheDirUrl = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            let appConfigUrl = bundleUrl.appending(path: "app-config.json")
            assert(fileManager.fileExists(atPath: appConfigUrl.path()))
            let fileHandle = try FileHandle(forReadingFrom: appConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            self.appConfig = try decoder.decode(AppConfig.self, from: data)
            
            for model in self.appConfig.model_list {
                if self.prebuiltLocals.contains(model.local_id) {
                    continue
                }
                let configUrl = cacheDirUrl
                    .appending(path:  model.local_id)
                    .appending(path: "mlc-chat-config.json")
                
                if fileManager.fileExists(atPath: configUrl.path()) {
                    loadConfig(modelRecord: model)
                } else {
                    downloadConfig(modelUrl: URL(string: model.model_url)!, newRecord: false)
                }
            }
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func addModel(modelRemoteBaseUrl: String) {
        for model in self.appConfig.model_list {
            if model.model_url == modelRemoteBaseUrl {
                return
            }
        }
        downloadConfig(modelUrl: URL(string: modelRemoteBaseUrl)!, newRecord: true)
    }
    
    func downloadConfig(modelUrl: URL, newRecord: Bool) {
        let downloadTask = URLSession.shared.downloadTask(with: modelUrl.appending(path: "mlc-chat-config.json")) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileUrl = urlOrNil else { return }
            do {
                let fileHandle = try FileHandle(forReadingFrom: fileUrl)
                let data = fileHandle.readDataToEndOfFile()
                let modelConfig = try self.decoder.decode(ModelConfig.self, from: data)
                let modelBaseUrl = self.cacheDirUrl.appending(path: modelConfig.local_id)
                
                if (newRecord) {
                    // TODO(tvm-team) add error message about duolicate
                    if self.prebuiltLocals.contains(modelConfig.local_id) {
                        return;
                    }
                    for model in self.appConfig.model_list {
                        if (model.local_id == modelConfig.local_id) {
                            return
                        }
                    }
                }
                
                try self.fileManager.createDirectory(at: modelBaseUrl, withIntermediateDirectories: true)
                try self.fileManager.moveItem(at: fileUrl, to: modelBaseUrl.appending(path: "mlc-chat-config.json"))
                DispatchQueue.main.async {
                    let record = ModelRecord(model_url: modelUrl.absoluteString, local_id: modelConfig.local_id)
                    self.loadConfig(modelRecord: record)
                    if (!newRecord) {
                        self.appConfig.model_list.append(record)
                        self.commitUpdate()
                    }
                }
            } catch {
                print(error.localizedDescription)
            }
        }
        downloadTask.resume()
    }
    
    func loadConfig(modelRecord: ModelRecord) {
        // local-id dir should exist
        let modelBaseUrl = cacheDirUrl.appending(path: modelRecord.local_id)
        assert(fileManager.fileExists(atPath: modelBaseUrl.path()))
        
        // mlc-chat-config.json should exist
        let modelConfigUrl = modelBaseUrl.appending(path: "mlc-chat-config.json")
        assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
        
        do {
            let fileHandle = try FileHandle(forReadingFrom: modelConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            let modelConfig = try decoder.decode(ModelConfig.self, from: data)
            models.append(ModelState(
                modelConfig: modelConfig,
                modelUrl: URL(string: modelRecord.model_url)!,
                modelDirUrl: cacheDirUrl.appending(path: modelConfig.local_id),
                chatState: chatState
            ))
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func commitUpdate(){
        // TODO(tvm-team): atomic switch over
        let appConfigUrl = Bundle.main.bundleURL.appending(path: "app-config.json")
        assert(fileManager.fileExists(atPath: appConfigUrl.path()))
        do {
            let fileHandle = try FileHandle(forWritingTo: appConfigUrl)
            let data = try encoder.encode(appConfig)
            try fileHandle.write(contentsOf: data)
        } catch {
            print(error.localizedDescription)
        }
    }
}
