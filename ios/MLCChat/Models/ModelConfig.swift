//
//  ModelConfig.swift
//  MLCChat
//

struct ModelConfig: Decodable {
    let tokenizerFiles: [String]
    var modelLib: String?
    var modelID: String?
    var estimatedVRAMReq: Int?

    enum CodingKeys: String, CodingKey {
        case tokenizerFiles = "tokenizer_files"
        case modelLib = "model_lib"
        case modelID = "model_id"
        case estimatedVRAMReq = "estimated_vram_req"
    }
}
