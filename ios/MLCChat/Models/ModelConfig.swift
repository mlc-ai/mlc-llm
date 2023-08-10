//
//  ModelConfig.swift
//  MLCChat
//

struct ModelConfig: Decodable {
    let modelLib: String
    let localID: String
    let tokenizerFiles: [String]
    let displayName: String!
    let estimatedVRAMReq: Int64!

    enum CodingKeys: String, CodingKey {
        case modelLib = "model_lib"
        case localID = "local_id"
        case tokenizerFiles = "tokenizer_files"
        case displayName = "display_name"
        case estimatedVRAMReq = "estimated_vram_req"
    }
}
