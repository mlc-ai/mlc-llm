//
//  ExampleModelConfig.swift
//  MLCChat
//

import Foundation

struct ExampleModelConfig: Identifiable {
    let id = UUID()
    let modelURL: String
    let localID: String

    enum CodingKeys: String, CodingKey {
        case modelURL = "model_url"
        case localID = "local_id"
    }
}
