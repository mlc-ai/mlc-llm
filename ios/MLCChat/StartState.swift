//
//  StartState.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/13/23.
//

import Foundation


class StartState : ObservableObject{
    @Published var models = [ModelState]()
    
    init() {
        let fileManager: FileManager = FileManager.default
        let bundleUrl = Bundle.main.bundleURL
        let decoder = JSONDecoder()
        // models in dist
        do {
            let distDirUrl = bundleUrl.appending(path: "dist")
            let contents = try fileManager.contentsOfDirectory(at: distDirUrl, includingPropertiesForKeys: nil)
            let modelDirs = contents.filter{$0.hasDirectoryPath}
            for modelDir in modelDirs {
                if fileManager.fileExists(atPath: modelDir.appending(path: "mlc-llm-config.json").path()) {
                    let fileHandle = try FileHandle(forReadingFrom: modelDir.appending(path: "mlc-llm-config.json"))
                    let data = fileHandle.readDataToEndOfFile()
                    let modelConfig = try decoder.decode(ModelConfig.self, from: data)
                    assert(modelDir.lastPathComponent == modelConfig.local_id)
                    models.append(ModelState(modelConfig: modelConfig, modelDirUrl: modelDir))
                }
            }
        } catch {
            print(error.localizedDescription)
        }
        
        // models in cache to download
        do {
            let cacheDirUrl = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            let modelConfigDirUrl = bundleUrl.appending(path: "ModelConfig")
            let contents = try fileManager.contentsOfDirectory(at: modelConfigDirUrl, includingPropertiesForKeys: nil)
            let jsonFiles = contents.filter{$0.pathExtension == "json"}
            for jsonFile in jsonFiles{
                assert(fileManager.fileExists(atPath: jsonFile.path()))
                let fileHandle = try FileHandle(forReadingFrom: jsonFile)
                let data = fileHandle.readDataToEndOfFile()
                let modelConfig = try decoder.decode(ModelConfig.self, from: data)
                models.append(ModelState(modelConfig: modelConfig, modelDirUrl: cacheDirUrl.appending(path: modelConfig.local_id)))
            }
        } catch {
            print(error.localizedDescription)
        }
    }
}
