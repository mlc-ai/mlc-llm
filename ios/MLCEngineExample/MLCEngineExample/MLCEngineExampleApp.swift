// This is a minimum example App to interact with MLC Engine
// This app is mainly created with minimalism in mind for
// example and quick testing purposes.
//
// To build this app, select target My Mac(Designed for iPad) and run
// Make sure you run "mlc_llm package" first with "MLCChat"
// replaced by "MLCEngineExample"
// to ensure the "dist/bundle" folder populates with the right model file
// and we have the model lib packaged correctly
import Foundation
import SwiftUI

import MLCSwift

class AppState: ObservableObject {
    // the MLC engine instance
    private let engine = MLCEngine()
    // obtain the local path to store models
    // this that stores the model files in the dist folder
    private let bundleURL = Bundle.main.bundleURL.appending(path: "bundle")
    // model path, this must match a builtin
    // file name in prepare_params.sh
    private let modelPath = "Llama-3-8B-Instruct-q3f16_1-MLC"
    // model lib identifier of within the packaged library
    // make sure we run "mlc_llm package"
    private let modelLib = "llama_q3f16_1"

    // this is a message to be displayed in app
    @Published var displayText = ""

    public func runExample() {
        // MLCEngine is a actor that can be called in an async context
        Task {
            let modelLocalPath = bundleURL.appending(path: modelPath).path()
            // Step 0: load the engine
            await engine.reload(modelPath: modelLocalPath, modelLib: modelLib)

            // run chat completion as in OpenAI API style
            for await res in await engine.chat.completions.create(
                messages: [
                    ChatCompletionMessage(
                        role: .user,
                        content: "What is the meaning of life?"
                    )
                ],
                stream_options: StreamOptions(include_usage: true)
            ) {
                // publish at main event loop
                DispatchQueue.main.async {
                    // parse the result content in structured form
                    // and stream back to the display
                    if let finalUsage = res.usage {
                        self.displayText += "\n" + (finalUsage.extra?.asTextLabel() ?? "")
                    } else {
                        self.displayText += res.choices[0].delta.content!.asText()
                    }
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
