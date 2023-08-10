//
//  ModelConfig.swift
//  MLCChat
//

struct ModelConfig: Decodable {
    let model_lib: String
    let local_id: String
    let tokenizer_files: [String]
    let display_name: String!
    let estimated_vram_req: Int64!
}
