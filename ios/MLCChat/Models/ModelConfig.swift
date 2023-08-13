//
//  ModelConfig.swift
//  MLCChat
//

struct ModelConfig: Decodable {
    let modelLib: String
    let localID: String
    let tokenizerFiles: [String]
    private let internalDisplayName: String?
    private let internalEstimatedVRAMReq: Int?

    enum CodingKeys: String, CodingKey {
        case modelLib = "model_lib"
        case localID = "local_id"
        case tokenizerFiles = "tokenizer_files"
        case internalDisplayName = "display_name"
        case internalEstimatedVRAMReq = "estimated_vram_req"
    }

    var displayName: String {
        internalDisplayName ?? localID.components(separatedBy: "-")[0]
    }

    var estimatedVRAMReq: Int {
        internalEstimatedVRAMReq ?? 4000000000
    }
}
