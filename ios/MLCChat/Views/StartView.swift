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
