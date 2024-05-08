//
//  AppConfig.swift
//  MLCChat
//

struct AppConfig: Codable {
    struct ModelRecord: Codable {
        let modelPath: String?
        let modelURL: String?
        let modelLib: String
        let estimatedVRAMReq: Int
        let modelID: String

        enum CodingKeys: String, CodingKey {
            case modelPath = "model_path"
            case modelURL = "model_url"
            case modelLib = "model_lib"
            case estimatedVRAMReq = "estimated_vram_bytes"
            case modelID = "model_id"
        }
    }

    var modelList: [ModelRecord]

    enum CodingKeys: String, CodingKey {
        case modelList = "model_list"
    }
}
