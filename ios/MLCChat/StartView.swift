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
    @State private var inputModelUrl: String = ""

    var body: some View {
        NavigationStack {
            List{
                Section(header: Text("Models")){
                    ForEach(state.models) { modelState in
                        ModelView().environmentObject(modelState)
                    }
                    if !isAdding {
                        Button("Add model variant") {
                            isAdding = true
                        }.buttonStyle(.borderless)
                    }
                }
                if isAdding {
                    Section(header: Text("MLC Chat Model URL")) {
                        TextField("Input model url here", text: $inputModelUrl, axis: .vertical)
                    }
                    Section(header: Text(
                        "Click below to input example URLs, " +
                        "these URLs may contain same weights as builtin ones"
                    )) {
                        ForEach(state.exampleModelUrls) { record in
                            Button(record.local_id) {
                                inputModelUrl = record.model_url
                            }.buttonStyle(.borderless)
                        }
                        Button("Clear URL") {
                            inputModelUrl = ""
                        }
                    }
                }
            }
            .navigationTitle("MLC Chat")
            .toolbar{
                if isAdding {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Cancel") {
                            isAdding = false
                            inputModelUrl = ""
                        }
                        .opacity(0.9)
                        .padding()
                    }
                    if !inputModelUrl.isEmpty {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Add") {
                                state.addModel(modelRemoteBaseUrl: inputModelUrl)
                                isAdding = false
                                inputModelUrl = ""
                            }
                            .opacity(0.9)
                            .bold()
                            .padding()
                        }
                    }
                }
            }.alert("Error", isPresented: $state.alertDisplayed, actions: {
                Button("OK") {}
            }, message: {
                Text(state.alertMessage)
            })

        }
    }
}


struct StartView_Previews: PreviewProvider {
    static var previews: some View {
        StartView().environmentObject(StartState())
    }
}
