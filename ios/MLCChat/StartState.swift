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
        do {
            let modelConfigDirUrl = bundleUrl.appending(path: "ModelConfig")
            let contents = try fileManager.contentsOfDirectory(at: modelConfigDirUrl, includingPropertiesForKeys: nil)
            let jsonFiles = contents.filter{$0.pathExtension == "json"}
            for jsonFile in jsonFiles{
                print(jsonFile)
                assert(fileManager.fileExists(atPath: jsonFile.path()))
                let fileHandle = try FileHandle(forReadingFrom: jsonFile)
                let data = fileHandle.readDataToEndOfFile()
                let modelConfig = try decoder.decode(ModelConfig.self, from: data)
                models.append(ModelState(modelConfig: modelConfig))
            }
        } catch {
            print(error)
        }
    }
}
