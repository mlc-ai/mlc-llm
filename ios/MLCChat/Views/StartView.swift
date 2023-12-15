//
//  DownloadView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/11/23.
//

import SwiftUI

struct StartView: View {
    @EnvironmentObject private var appState: AppState
    @State private var isAdding: Bool = false
    @State private var isRemoving: Bool = false
    @State private var inputModelUrl: String = ""

    var body: some View {
        NavigationStack {
            List{
                Section(header: Text("Models")) {
                    ForEach(appState.models) { modelState in
                        ModelView(isRemoving: $isRemoving)
                            .environmentObject(modelState)
                            .environmentObject(appState.chatState)
                    }
                    if !isRemoving {
                        Button("Edit model") {
                            isRemoving = true
                        }
                        .buttonStyle(.borderless)
                    } else {
                        Button("Cancel edit model") {
                            isRemoving = false
                        }
                        .buttonStyle(.borderless)
                    }
                }
                if isAdding {
                    Section(header: Text("Click below to import sample model variants, these variants may contain same weights as builtin ones")) {
                        ForEach(appState.exampleModels) { exampleModel in
                            Button(exampleModel.localID) {
                                appState.requestAddModel(url: exampleModel.modelURL, localID: exampleModel.localID)
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                    Section(header: Text("Add model by URL, sample URL: \"https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int4/\"")) {
                        TextField("Input model url here", text: $inputModelUrl, axis: .vertical)
                        Button("Clear URL") {
                            inputModelUrl = ""
                        }
                        .buttonStyle(.borderless)
                        Button("Add model") {
                            appState.requestAddModel(url: inputModelUrl, localID: nil)
                            isAdding = false
                            inputModelUrl = ""
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            .navigationTitle("MLC Chat")
            .alert("Error", isPresented: $appState.alertDisplayed) {
                Button("OK") { }
            } message: {
                Text(appState.alertMessage)
            }
        }
    }
}
