//
//  AppConfig.swift
//  MLCChat
//

struct AppConfig: Codable {
    struct ModelRecord: Codable {
        let modelURL: String
        let localID: String

        enum CodingKeys: String, CodingKey {
            case modelURL = "model_url"
            case localID = "local_id"
        }
    }

    let modelLibs: [String]
    var modelList: [ModelRecord]
    let exampleModels: [ModelRecord]

    enum CodingKeys: String, CodingKey {
        case modelLibs = "model_libs"
        case modelList = "model_list"
        case exampleModels = "add_model_samples"
    }
}
