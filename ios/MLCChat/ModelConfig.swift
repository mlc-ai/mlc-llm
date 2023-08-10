//
//  ModelConfig.swift
//  MLCChat
//

import Foundation

struct ModelConfig: Decodable {
    let model_lib: String
    let local_id: String
    let tokenizer_files: [String]
    let display_name: String!
    let estimated_vram_req: Int64!
}

struct ParamsRecord: Decodable {
    let dataPath: String
}

struct ParamsConfig: Decodable {
    let records: [ParamsRecord]
}

struct ModelRecord: Codable {
    let model_url: String
    let local_id: String
}

struct AppConfig: Codable {
    let model_libs: [String]
    var model_list: [ModelRecord]
    let add_model_samples: [ModelRecord]
}

struct ExampleModelUrl: Identifiable {
    let id = UUID()
    let model_url: String
    let local_id: String
}
