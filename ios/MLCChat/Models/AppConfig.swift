//
//  AppConfig.swift
//  MLCChat
//

struct AppConfig: Codable {
    struct ModelRecord: Codable {
        let model_url: String
        let local_id: String
    }

    let model_libs: [String]
    var model_list: [ModelRecord]
    let add_model_samples: [ModelRecord]
}
