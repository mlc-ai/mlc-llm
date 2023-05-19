//
//  ModelConfig.swift
//  MLCChat
//

import Foundation

struct ModelConfig: Codable, Hashable {
    let model_lib: String
    let local_id: String
    let tokenizer_files: [String]
    let display_name: String!
    let estimated_memory_req: Int64!
}

struct ParamsRecord: Codable, Hashable {
    let dataPath: String
}

struct ParamsConfig: Codable, Hashable {
    let records: [ParamsRecord]
}

struct ModelRecord: Codable, Hashable {
    let model_url: String
    let local_id: String
}

struct AppConfig: Codable, Hashable {
    let model_libs: [String]
    var model_list: [ModelRecord]
}
