//
//  DownloadView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/11/23.
//

import SwiftUI

struct StartView: View {
    @EnvironmentObject var state: StartState
    @State private var isAdding: Bool = false
    @State private var isRemoving: Bool = false
    @State private var inputModelUrl: String = ""

    var body: some View {
        NavigationStack {
            List{
                Section(header: Text("Models")){
                    ForEach(state.models) { modelState in
                        ModelView(isRemoving: $isRemoving).environmentObject(modelState).environmentObject(state.chatState)
                    }
                    if !isRemoving {
                        Button("Edit model") {
                            isRemoving = true
                        }.buttonStyle(.borderless)
                    } else {
                        Button("Cancel edit model") {
                            isRemoving = false
                        }.buttonStyle(.borderless)
                    }
                    if !isAdding {
                        Button("Add model variant") {
                            isAdding = true
                        }.buttonStyle(.borderless)
                    } else {
                        Button("Cancel add model variant") {
                            isAdding = false
                            inputModelUrl = ""
                        }.buttonStyle(.borderless)
                    }
                }
                if isAdding {
                    Section(header: Text(
                        "Click below to import sample model variants, " +
                        "these variants may contain same weights as builtin ones"
                    )) {
                        ForEach(state.exampleModelUrls) { record in
                            Button(record.local_id) {
                                state.requestAddModel(url: record.model_url, localId: record.local_id)
                            }.buttonStyle(.borderless)
                        }
                    }
                    Section(header: Text("Add model by URL, sample URL: \"https://huggingface.co/mlc-ai/demo-vicuna-v1-7b-int4/\"")) {
                        TextField("Input model url here", text: $inputModelUrl, axis: .vertical)
                        Button("Clear URL") {
                            inputModelUrl = ""
                        }.buttonStyle(.borderless)
                        Button("Add model") {
                            state.requestAddModel(url: inputModelUrl, localId: nil)
                            isAdding = false
                            inputModelUrl = ""
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            .navigationTitle("MLC Chat")
            .alert("Error", isPresented: $state.alertDisplayed, actions: {
                Button("OK") {}
            }, message: {
                Text(state.alertMessage)
            })

        }
    }
}
