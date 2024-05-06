// NOTE: This example is still work in progress
//
// This is a minimum example App to interact with MLC Engine
// This app is mainly created with minimalism in mind for
// example and quick testing purposes.
//
// To build this app, select target My Mac(Designed for iPad) and run
// Make sure you run prepare_libs.sh and prepare_params.sh first
// to ensure the dist folder populates with the right model file
// and we have the model lib packaged correctly
import Foundation
import SwiftUI

// Import MLCSwift
import MLCSwift

class AppState: ObservableObject {
    // the MLC engine instance
    private let engine = MLCEngine()
    // obtain the local path to store models
    // this that stores the model files in the dist folder
    private let distURL = Bundle.main.bundleURL.appending(path: "dist")
    // NOTE: this does not yet work out of box
    // need to supply the Llama-3-8B-Instruct-q3f16_1-MLC and llama_q3f16_1
    // via manual local compile
    // TODO(mlc-team): update prebuild so it can be used out of box
    //
    // model path, this must match a builtin
    // file name in prepare_params.sh
    private let modelPath = "Llama-3-8B-Instruct-q3f16_1-MLC"
    // model lib identifier of within the packaged library
    // this must match a config in MLCChat/app-config.json
    // make sure we run prepare_libs.sh
    private let modelLib = "llama_q3f16_1"

    // this is a message to be displayed in app
    @Published var displayText = ""

    public func runExample() {
        // MLCEngine is a actor that can be called in an async context
        Task {
            let modelLocalPath = distURL.appending(path: modelPath).path()
            // Step 0: load the engine
            await engine.reload(modelPath: modelLocalPath, modelLib: modelLib)

            // TODO(mlc-team) update request so it is also structure based
            // as in open ai api
            // sent a request
            let jsonRequest = """
            {
                "model": "llama3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": "What is the meaning of life?" }
                        ]
                    }
                ]
            }
            """
            // run chat completion as in OpenAI API style
            for await res in await engine.chatCompletion(jsonRequest: jsonRequest) {
                // publish at main event loop
                DispatchQueue.main.async {
                    // parse the result content in structured form
                    // and stream back to the display
                    self.displayText += res.choices[0].delta.content![0]["text"]!
                }
            }
        }
    }
}


@main
struct MLCEngineExampleApp: App {
    private let appState = AppState()

    init() {
        // we simply run test
        // please checkout output in console
        appState.runExample()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}
