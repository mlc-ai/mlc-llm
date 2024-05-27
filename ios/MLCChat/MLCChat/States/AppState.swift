//
//  AppState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/13/23.
//

import Foundation

final class AppState: ObservableObject {
    @Published var models = [ModelState]()
    @Published var chatState = ChatState()

    @Published var alertMessage = "" // TODO: Should move out
    @Published var alertDisplayed = false // TODO: Should move out

    private var appConfig: AppConfig?
    private var modelIDs = Set<String>()

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
        loadModelsConfig(modelList: appConfig.modelList)
    }

    func requestDeleteModel(modelID: String) {
        // model dir should have been deleted in ModelState
        assert(!fileManager.fileExists(atPath: cacheDirectoryURL.appending(path: modelID).path()))
        modelIDs.remove(modelID)
        models.removeAll(where: {$0.modelConfig.modelID == modelID})
        updateAppConfig {
            appConfig?.modelList.removeAll(where: {$0.modelID == modelID})
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
            if model.modelPath != nil {
                // local model
                let modelDir = Bundle.main.bundleURL.appending(path: Constants.prebuiltModelDir).appending(path: model.modelPath!)
                let modelConfigURL = modelDir.appending(path: Constants.modelConfigFileName)
                if fileManager.fileExists(atPath: modelConfigURL.path()) {
                    if let modelConfig = loadModelConfig(
                        modelConfigURL: modelConfigURL,
                        modelLib: model.modelLib,
                        modelID: model.modelID,
                        estimatedVRAMReq: model.estimatedVRAMReq
                    ) {
                        addModelConfig(
                            modelConfig: modelConfig,
                            modelPath: model.modelPath!,
                            modelURL: nil,
                            isBuiltin: true
                        )
                    } else {
                        showAlert(message: "Failed to load prebuilt model: \(model.modelPath!)")
                    }
                } else {
                    showAlert(message: "Prebuilt mlc-chat-config.json file not found: \(model.modelPath!)")
                }
            } else if model.modelURL != nil {
                // remote model
                let modelConfigFileURL = cacheDirectoryURL
                    .appending(path: model.modelID)
                    .appending(path: Constants.modelConfigFileName)
                if fileManager.fileExists(atPath: modelConfigFileURL.path()) {
                    if let modelConfig = loadModelConfig(
                        modelConfigURL: modelConfigFileURL,
                        modelLib: model.modelLib,
                        modelID: model.modelID,
                        estimatedVRAMReq: model.estimatedVRAMReq
                    ) {
                        addModelConfig(
                            modelConfig: modelConfig,
                            modelPath: nil,
                            modelURL: URL(string: model.modelURL!),
                            isBuiltin: true
                        )
                    }
                } else {
                    downloadConfig(
                        modelURL: URL(string: model.modelURL!),
                        modelLib: model.modelLib,
                        modelID: model.modelID,
                        estimatedVRAMReq: model.estimatedVRAMReq,
                        isBuiltin: true
                    )
                }
            } else {
                showAlert(message: "Path or URL should be provided in app config: \(model.modelID)")
            }
        }
    }

    func loadModelConfig(modelConfigURL: URL, modelLib: String, modelID: String, estimatedVRAMReq: Int) -> ModelConfig? {
        do {
            assert(fileManager.fileExists(atPath: modelConfigURL.path()))
            let fileHandle = try FileHandle(forReadingFrom: modelConfigURL)
            let data = fileHandle.readDataToEndOfFile()
            var modelConfig = try jsonDecoder.decode(ModelConfig.self, from: data)
            modelConfig.modelLib = modelLib
            modelConfig.modelID = modelID
            modelConfig.estimatedVRAMReq = estimatedVRAMReq
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

    func downloadConfig(modelURL: URL?, modelLib: String, modelID: String, estimatedVRAMReq: Int, isBuiltin: Bool) {
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
                guard let modelConfig = loadModelConfig(
                    modelConfigURL: tempFileURL,
                    modelLib: modelLib,
                    modelID: modelID,
                    estimatedVRAMReq: estimatedVRAMReq
                ) else {
                    try fileManager.removeItem(at: tempFileURL)
                    return
                }

                if modelIDs.contains(modelConfig.modelID!) {
                    try fileManager.removeItem(at: tempFileURL)
                    return
                }

                let modelBaseUrl = cacheDirectoryURL.appending(path: modelConfig.modelID!)
                try fileManager.createDirectory(at: modelBaseUrl, withIntermediateDirectories: true)
                let modelConfigUrl = modelBaseUrl.appending(path: Constants.modelConfigFileName)
                try fileManager.moveItem(at: tempFileURL, to: modelConfigUrl)
                assert(fileManager.fileExists(atPath: modelConfigUrl.path()))
                assert(!fileManager.fileExists(atPath: tempFileURL.path()))
                addModelConfig(
                    modelConfig: modelConfig,
                    modelPath: nil,
                    modelURL: modelURL,
                    isBuiltin: isBuiltin
                )
            } catch {
                showAlert(message: "Failed to import model: \(error.localizedDescription)")
            }
        }
        downloadTask.resume()
    }

    func addModelConfig(modelConfig: ModelConfig, modelPath: String?, modelURL: URL?, isBuiltin: Bool) {
        assert(!modelIDs.contains(modelConfig.modelID!))
        modelIDs.insert(modelConfig.modelID!)
        let modelBaseURL: URL

        // model_id dir should exist
        if modelURL == nil {
            // prebuilt model in bundle
            modelBaseURL = Bundle.main.bundleURL.appending(path: Constants.prebuiltModelDir).appending(path: modelPath!)
        } else {
            // download model in cache
            modelBaseURL = cacheDirectoryURL.appending(path: modelConfig.modelID!)
        }
        assert(fileManager.fileExists(atPath: modelBaseURL.path()))

        // mlc-chat-config.json should exist
        let modelConfigURL = modelBaseURL.appending(path: Constants.modelConfigFileName)
        assert(fileManager.fileExists(atPath: modelConfigURL.path()))

        let model = ModelState(modelConfig: modelConfig, modelLocalBaseURL: modelBaseURL, startState: self, chatState: chatState)
        model.checkModelDownloadState(modelURL: modelURL)

        // addModelConfig is not called from main thread, update to models needs to be performed on main
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            models.append(model)
        }

        if modelURL != nil && !isBuiltin {
            updateAppConfig {
                appConfig?.modelList.append(
                    AppConfig.ModelRecord(
                        modelPath: nil,
                        modelURL: modelURL!.absoluteString,
                        modelLib: modelConfig.modelLib!,
                        estimatedVRAMReq: modelConfig.estimatedVRAMReq!,
                        modelID: modelConfig.modelID!
                    )
                )
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
