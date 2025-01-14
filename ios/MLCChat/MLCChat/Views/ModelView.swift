//
//  ModelView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/14/23.
//

import SwiftUI

struct ModelView: View {
    @EnvironmentObject private var modelState: ModelState
    @EnvironmentObject private var chatState: ChatState
    @Binding var isRemoving: Bool

    @State private var isShowingDeletionConfirmation: Bool = false

    var body: some View {
        VStack(alignment: .leading) {
            if (modelState.modelDownloadState == .finished) {
                NavigationLink(destination:
                                ChatView()
                    .environmentObject(chatState)
                    .onAppear {
                        modelState.startChat(chatState: chatState)
                    }
                ) {
                    HStack {
                        Text(modelState.modelConfig.modelID!)
                        Spacer()
                        if chatState.isCurrentModel(modelID: modelState.modelConfig.modelID!) {
                            Image(systemName: "checkmark").foregroundColor(.blue)
                        }
                    }
                }
                .buttonStyle(.borderless)
            } else {
                Text(modelState.modelConfig.modelID!).opacity(0.5)
            }
            HStack{
                if modelState.modelDownloadState != .finished || isRemoving {
                    ProgressView(value: Double(modelState.progress) / Double(modelState.total))
                        .progressViewStyle(.linear)
                }

                if (modelState.modelDownloadState == .paused) {
                    Button {
                        modelState.handleStart()
                    } label: {
                        Image(systemName: "icloud.and.arrow.down")
                    }
                    .buttonStyle(.borderless)
                } else if (modelState.modelDownloadState == .downloading) {
                    Button {
                        modelState.handlePause()
                    } label: {
                        Image(systemName: "stop.circle")
                    }
                    .buttonStyle(.borderless)
                } else if (modelState.modelDownloadState == .failed) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.red)
                }

                if isRemoving {
                    Button(role: .destructive) {
                        isShowingDeletionConfirmation = true
                    } label: {
                        Image(systemName: "trash")
                    }
                    .confirmationDialog("Delete Model", isPresented: $isShowingDeletionConfirmation) {
                        Button("Delete Model", role: .destructive) {
                            modelState.handleDelete()
                        }
                        .disabled(
                            modelState.modelDownloadState != .downloading &&
                            modelState.modelDownloadState != .paused &&
                            modelState.modelDownloadState != .finished &&
                            modelState.modelDownloadState != .failed)
                        Button("Clear Data") {
                            modelState.handleClear()
                        }
                        .disabled(
                            modelState.modelDownloadState != .downloading &&
                            modelState.modelDownloadState != .paused &&
                            modelState.modelDownloadState != .finished)
                        Button("Cancel", role: .cancel) {
                            isShowingDeletionConfirmation = false
                        }
                    } message: {
                        Text("Delete model will delete the all files with model config, and delete the entry in list. \n Clear model will keep the model config only, and keep the entry in list for future re-downloading.")
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
    }
}
