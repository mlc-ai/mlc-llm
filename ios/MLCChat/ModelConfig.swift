//
//  ModelConfig.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/15/23.
//

import Foundation

struct ModelConfig: Codable, Hashable {
    let model_url: String
    let model_lib: String
    let local_id: String
    let tokenizer_files: [String]
    let ndarray_file: String
}

struct ParamsRecord: Codable, Hashable {
    let dataPath: String
}

struct ParamsConfig: Codable, Hashable {
    let records: [ParamsRecord]
}
