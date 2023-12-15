//
//  AppState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/13/23.
//

import Foundation

final class AppState: ObservableObject {
    @Published var models = [ModelState]()
    @Published var exampleModels = [ExampleModelConfig]()
    @Published var chatState = ChatState()

    @Published var alertMessage = "" // TODO: Should move out
    @Published var alertDisplayed = false // TODO: Should move out
    
    private var appConfig: AppConfig?
    private var localModelIDs = Set<String>()

    private let fileManager: FileManager = FileManager.default
    private lazy var cacheDirectoryURL: URL = {
        fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    }()

    private let jsonDecoder = JSONDecoder()
    private let jsonEncoder = JSONEncoder()

    func loadAppConfigAndModels() {
        appConfig = loadAppConfig()
        // Can't do anything without a valid app config
        guard let appConfig else {
            return
        }
        loadPrebuiltModels()
        loadModelsConfig(modelList: appConfig.modelList)
        loadExampleModelsConfig(exampleModels: appConfig.exampleModels)
    }

    func requestAddModel(url: String, localID: String?) {
        if let localID, localModelIDs.contains(localID) {
            showAlert(message: "Local ID: \(localID) has been occupied")
        } else {
            if let modelURL = URL(string: url) {
                downloadConfig(modelURL: modelURL, localID: localID, isBuiltin: false)
            } else {
                showAlert(message: "Failed to resolve the given url")
            }
        }
    }
    
    func requestDeleteModel(localId: String) {
        // model dir should have been deleted in ModelState
        assert(!fileManager.fileExists(atPath: cacheDirectoryURL.appending(path: localId).path()))
        localModelIDs.remove(localId)
        models.removeAll(where: {$0.modelConfig.localID == localId})
        updateAppConfig {
            appConfig?.modelList.removeAll(where: {$0.localID == localId})
        }
    }
}

private extension AppState {
    func loadAppConfig() -> AppConfig? {
        // models in cache to download
        var appConfigFileURL = cacheDirectoryURL.appending(path: Constants.appConfigFileName)
        if !fileManager.fileExists(atPath: appConfigFileURL.path()) {
            appConfigFileURL = Bundle.main.bundleURL.appending(path: Constants.appConfigFileName)
        }
        assert(fileManager.fileExists(atPath: appConfigFileURL.path()))

        do {
            let fileHandle = try FileHandle(forReadingFrom: appConfigFileURL)
            let data = fileHandle.readDataToEndOfFile()

            let appConfig = try jsonDecoder.decode(AppConfig.self, from: data)
            return appConfig
        } catch {
            showAlert(message: "Failed to load app config: \(error.localizedDescription)")
            return nil
        }
    }

    func loadModelsConfig(modelList: [AppConfig.ModelRecord]) {
        for model in modelList {
            let modelConfigFileURL = cacheDirectoryURL
                .appending(path: model.localID)
                .appending(path: Constants.modelConfigFileName)
            if fileManager.fileExists(atPath: modelConfigFileURL.path()) {
                if let modelConfig = loadModelConfig(modelConfigURL: modelConfigFileURL) {
                    addModelConfig(
                        modelConfig: modelConfig,
                        modelURL: URL(string: model.modelURL),
                        isBuiltin: true
                    )
                }
            } else {
                downloadConfig(
                    modelURL: URL(string: model.modelURL),
                    localID: model.localID,
                    isBuiltin: true)
            }
        }
    }

    func loadExampleModelsConfig(exampleModels: [AppConfig.ModelRecord]) {
        self.exampleModels = exampleModels.map{
            ExampleModelConfig(modelURL: $0.modelURL, localID: $0.localID)
        }
    }

    func loadPrebuiltModels() {
        // models in dist
        do {
            let distDirURL = Bundle.main.bundleURL.appending(path: Constants.prebuiltModelDir)
            let contents = try fileManager.contentsOfDirectory(at: distDirURL, includingPropertiesForKeys: nil)
            let modelDirs = contents.filter{ $0.hasDirectoryPath }
            for modelDir in modelDirs {
                let modelConfigURL = modelDir.appending(path: Constants.modelConfigFileName)
                if fileManager.fileExists(atPath: modelConfigURL.path()) {
                    if let modelConfig = loadModelConfig(modelConfigURL: modelConfigURL) {
                        assert(modelDir.lastPathComponent == modelConfig.localID)
                        addModelConfig(modelConfig: modelConfig, modelURL: nil, isBuiltin: true)
                    }
                }
            }
        } catch {
            showAlert(message: "Failed to load prebuilt models: \(error.localizedDescription)")
        }
    }

    func loadModelConfig(modelConfigURL: URL) -> ModelConfig? {
        do {
            assert(fileManager.fileExists(atPath: modelConfigURL.path()))
            let fileHandle = try FileHandle(forReadingFrom: modelConfigURL)
            let data = fileHandle.readDataToEndOfFile()
            let modelConfig = try jsonDecoder.decode(ModelConfig.self, from: data)
            if !isModelConfigAllowed(modelConfig: modelConfig) {
                return nil
            }
            return modelConfig
        } catch {
            showAlert(message: "Failed to resolve model config: \(error.localizedDescription)")
        }
        return nil
    }

    func showAlert(message: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if !self.alertDisplayed {
                self.alertMessage = message
                self.alertDisplayed = true
            } else {
                self.alertMessage.append("\n" + message)
            }
        }
    }

    func isModelConfigAllowed(modelConfig: ModelConfig) -> Bool {
        if appConfig?.modelLibs.contains(modelConfig.modelLib) ?? true {
            return true
        }
        showAlert(message: "Model lib \(modelConfig.modelLib) is not supported")
        return false
    }

    func downloadConfig(modelURL: URL?, localID: String?, isBuiltin: Bool) {
        guard let modelConfigURL = modelURL?.appending(path: "resolve").appending(path: "main").appending(path: Constants.modelConfigFileName) else {
            return
        }

        let downloadTask = URLSession.shared.downloadTask(with: modelConfigURL) {
            [weak self] urlOrNil, responseOrNil, errorOrNil in
            guard let self else {
                return
            }
            if let error = errorOrNil {
                self.showAlert(message: "Failed to download model config: \(error.localizedDescription)")
                return
            }
            guard let fileUrl = urlOrNil else {
                self.showAlert(message: "Failed to download model config")
                return
            }

            // cache temp file to avoid being deleted by system automatically
            let tempName = UUID().uuidString
            let tempFileURL = self.cacheDirectoryURL.appending(path: tempName)

            do {
                try self.fileManager.moveItem(at: fileUrl, to: tempFileURL)
            } catch {
                self.showAlert(message: "Failed to cache downloaded file: \(error.localizedDescription)")
                return
            }

            do {
                guard let modelConfig = loadModelConfig(modelConfigURL: tempFileURL) else {
                    try fileManager.removeItem(at: tempFileURL)
                    return
                }

                if localID != nil {
                    assert(localID == modelConfig.localID)
                }

                if localModelIDs.contains(modelConfig.localID) {
                    try fileManager.removeItem(at: tempFileURL)
                    return
                }

                let modelBaseUrl = cacheDirectoryURL.appending(path: modelConfig.localID)
                try fileManager.createDirectory(at: modelBaseUrl, withIntermediateDirectories: true)
                let modelConfigUrl = modelBaseUrl.appending(path: Constants.modelConfigFileName)
                try fileManager.moveItem(at: tempFileURL, to: modelConfigUrl)
                assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
                assert(!fileManager.fileExists(atPath: tempFileURL.path()))
                addModelConfig(modelConfig: modelConfig, modelURL: modelURL, isBuiltin: isBuiltin)
            } catch {
                showAlert(message: "Failed to import model: \(error.localizedDescription)")
            }
        }
        downloadTask.resume()
    }

    func addModelConfig(modelConfig: ModelConfig, modelURL: URL?, isBuiltin: Bool) {
        assert(!localModelIDs.contains(modelConfig.localID))
        localModelIDs.insert(modelConfig.localID)
        let modelBaseURL: URL

        // local-id dir should exist
        if modelURL == nil {
            // prebuilt model in dist
            modelBaseURL = Bundle.main.bundleURL.appending(path: Constants.prebuiltModelDir).appending(path: modelConfig.localID)
        } else {
            // download model in cache
            modelBaseURL = cacheDirectoryURL.appending(path: modelConfig.localID)
        }
        assert(fileManager.fileExists(atPath: modelBaseURL.path()))

        // mlc-chat-config.json should exist
        let modelConfigURL = modelBaseURL.appending(path: Constants.modelConfigFileName)
        assert(fileManager.fileExists(atPath: modelConfigURL.path()))

        let model = ModelState(modelConfig: modelConfig, modelLocalBaseURL: modelBaseURL, startState: self, chatState: chatState)
        model.checkModelDownloadState(modelURL: modelURL)
        models.append(model)
        if modelURL != nil && !isBuiltin {
            updateAppConfig {
                appConfig?.modelList.append(AppConfig.ModelRecord(modelURL: modelURL!.absoluteString, localID: modelConfig.localID))
            }
        }
    }

    func updateAppConfig(action: () -> Void) {
        action()
        let appConfigURL = cacheDirectoryURL.appending(path: Constants.appConfigFileName)
        do {
            let data = try jsonEncoder.encode(appConfig)
            try data.write(to: appConfigURL, options: Data.WritingOptions.atomic)
        } catch {
            print(error.localizedDescription)
        }
    }
}
