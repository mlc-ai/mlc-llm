//
//  StartState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/13/23.
//

import Foundation


class StartState : ObservableObject {
    @Published var models = [ModelState]()
    @Published var exampleModelUrls = [ExampleModelUrl]()
    @Published var alertMessage = ""
    @Published var alertDisplayed = false
    @Published var chatState = ChatState()
    
    private var appConfig: AppConfig!
    private var cacheDirUrl: URL!
    private let fileManager: FileManager = FileManager.default
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private var prebuiltLocalIds = Set<String>()
    private var localIds = Set<String>()
    
    static let PrebuiltModelDir = "dist"
    static let AppConfigFileName = "app-config.json"
    static let ModelConfigFileName = "mlc-chat-config.json"
    static let ParamsConfigFileName = "ndarray-cache.json"
    
    init() {
        loadPrebuiltModels()
        loadAppConfig()
    }
    
    private func loadPrebuiltModels() {
        let bundleUrl = Bundle.main.bundleURL
        // models in dist
        do {
            let distDirUrl = bundleUrl.appending(path: StartState.PrebuiltModelDir)
            let contents = try fileManager.contentsOfDirectory(at: distDirUrl, includingPropertiesForKeys: nil)
            let modelDirs = contents.filter{$0.hasDirectoryPath}
            for modelDir in modelDirs {
                let modelConfigUrl = modelDir.appending(path: StartState.ModelConfigFileName)
                if fileManager.fileExists(atPath: modelConfigUrl.path()) {
                    if let modelConfig = loadModelConfig(modelConfigUrl: modelConfigUrl) {
                        assert(modelDir.lastPathComponent == modelConfig.local_id)
                        addModelConfig(modelConfig: modelConfig, modelUrl: nil, isBuiltin: true)
                    }
                }
            }
        } catch {
            showAlert(message: "Failed to load prebuilt models: \(error.localizedDescription)")
        }
        
    }
    
    private func loadAppConfig() {
        let bundleUrl = Bundle.main.bundleURL
        // models in cache to download
        do {
            cacheDirUrl = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            var appConfigUrl = cacheDirUrl.appending(path: StartState.AppConfigFileName)
            if !fileManager.fileExists(atPath: appConfigUrl.path()) {
                appConfigUrl = bundleUrl.appending(path: StartState.AppConfigFileName)
            }
            assert(fileManager.fileExists(atPath: appConfigUrl.path()))
            let fileHandle = try FileHandle(forReadingFrom: appConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            appConfig = try decoder.decode(AppConfig.self, from: data)
            for model in self.appConfig.model_list {
                if self.prebuiltLocalIds.contains(model.local_id) {
                    continue
                }
                let modelConfigUrl = cacheDirUrl
                    .appending(path:  model.local_id)
                    .appending(path: StartState.ModelConfigFileName)
                
                if fileManager.fileExists(atPath: modelConfigUrl.path()) {
                    if let modelConfig = loadModelConfig(modelConfigUrl: modelConfigUrl) {
                        addModelConfig(
                            modelConfig: modelConfig,
                            modelUrl: URL(string: model.model_url)!,
                            isBuiltin: true
                        )
                    }
                } else {
                    downloadConfig(modelUrl: URL(string: model.model_url)!, localId: model.local_id, isBuiltin: true)
                }
            }
            for sample in appConfig.add_model_samples {
                exampleModelUrls.append(ExampleModelUrl(model_url: sample.model_url, local_id: sample.local_id))
            }
        } catch {
            showAlert(message: "Failed to load app config: \(error.localizedDescription)")
        }
    }
    
    private func loadModelConfig(modelConfigUrl: URL) -> ModelConfig? {
        do {
            assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
            let fileHandle = try FileHandle(forReadingFrom: modelConfigUrl)
            let data = fileHandle.readDataToEndOfFile()
            return try decoder.decode(ModelConfig.self, from: data)
        } catch {
            showAlert(message: "Failed to resolve model config: \(error.localizedDescription)")
        }
        return nil
    }
    
    func requestAddModel(url: String, localId: String?) {
        if localId != nil && localIds.contains(localId!) {
            showAlert(message: "Local ID: \(localId!) has been occupied")
        } else {
            if let modelUrl = URL(string: url) {
                downloadConfig(modelUrl: modelUrl, localId: localId, isBuiltin: false)
            } else {
                showAlert(message: "Failed to resolve the given url")
            }
        }
    }
    
    func requestDeleteModel(localId: String) {
        // model dir should have been deleted in ModelState
        assert(!fileManager.fileExists(atPath: cacheDirUrl.appending(path: localId).path()))
        localIds.remove(localId)
        models.removeAll(where: {$0.modelConfig.local_id == localId})
        updateAppConfig {
            appConfig.model_list.removeAll(where: {$0.local_id == localId})
        }
    }
    
    private func showAlert(message: String) {
        if !alertDisplayed {
            alertMessage = message
            alertDisplayed = true
        } else {
            alertMessage.append("\n" + message)
        }
    }
    
    
    private func downloadConfig(modelUrl: URL, localId: String?, isBuiltin: Bool) {
        let downloadTask = URLSession.shared.downloadTask(with: modelUrl.appending(path: "resolve").appending(path: "main").appending(path: StartState.ModelConfigFileName)) {
            urlOrNil, responseOrNil, errorOrNil in
            if let error = errorOrNil {
                DispatchQueue.main.sync {
                    self.showAlert(message: "Failed to download model config: \(error.localizedDescription)")
                }
                return
            }
            guard let fileUrl = urlOrNil else {
                DispatchQueue.main.sync {
                    self.showAlert(message: "Failed to download model config")
                }
                return
            }
            
            // cache temp file to avoid being deleted by system automatically
            let tempName = UUID().uuidString
            let tempFileUrl = self.cacheDirUrl.appending(path: tempName)
            
            do {
                try self.fileManager.moveItem(at: fileUrl, to: tempFileUrl)
            } catch {
                DispatchQueue.main.sync {
                    self.showAlert(message: "Failed to cache downloaded file: \(error.localizedDescription)")
                }
                return
            }
            
            DispatchQueue.main.async { [self] in
                do {
                    guard let modelConfig = loadModelConfig(modelConfigUrl: tempFileUrl) else {
                        try fileManager.removeItem(at: tempFileUrl)
                        return
                    }
                    
                    
                    if localId != nil {
                        assert(localId == modelConfig.local_id)
                    }
                    
                    if localIds.contains(modelConfig.local_id) {
                        try fileManager.removeItem(at: tempFileUrl)
                        return
                    }
                    
                    let modelBaseUrl = cacheDirUrl.appending(path: modelConfig.local_id)
                    try fileManager.createDirectory(at: modelBaseUrl, withIntermediateDirectories: true)
                    let modelConfigUrl = modelBaseUrl.appending(path: StartState.ModelConfigFileName)
                    try fileManager.moveItem(at: tempFileUrl, to: modelConfigUrl)
                    assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
                    assert(!fileManager.fileExists(atPath: tempFileUrl.path()))
                    addModelConfig(modelConfig: modelConfig, modelUrl: modelUrl, isBuiltin: isBuiltin)
                } catch {
                    showAlert(message: "Failed to import model: \(error.localizedDescription)")
                }
            }
        }
        downloadTask.resume()
    }
    
    private func addModelConfig(modelConfig: ModelConfig, modelUrl: URL?, isBuiltin: Bool) {
        assert(!localIds.contains(modelConfig.local_id))
        localIds.insert(modelConfig.local_id)
        var modelBaseUrl: URL
        
        // local-id dir should exist
        if modelUrl == nil {
            // prebuilt model in dist
            modelBaseUrl = Bundle.main.bundleURL.appending(path: StartState.PrebuiltModelDir).appending(path: modelConfig.local_id)
        } else {
            // download model in cache
            modelBaseUrl = cacheDirUrl.appending(path: modelConfig.local_id)
        }
        assert(fileManager.fileExists(atPath: modelBaseUrl.path()))
        
        // mlc-chat-config.json should exist
        let modelConfigUrl = modelBaseUrl.appending(path: StartState.ModelConfigFileName)
        assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
        
        models.append(ModelState(modelConfig: modelConfig, modelUrl: modelUrl, modelDirUrl: modelBaseUrl, startState: self, chatState: chatState))
        if modelUrl != nil && !isBuiltin {
            updateAppConfig {
                appConfig.model_list.append(ModelRecord(model_url: modelUrl!.absoluteString, local_id: modelConfig.local_id))
            }
        }
    }
    
    private func updateAppConfig(action: () -> Void) {
        action()
        let appConfigUrl = cacheDirUrl.appending(path: StartState.AppConfigFileName)
        do {
            let data = try encoder.encode(appConfig)
            try data.write(to: appConfigUrl, options: Data.WritingOptions.atomic)
        } catch {
            print(error.localizedDescription)
        }
    }
}
