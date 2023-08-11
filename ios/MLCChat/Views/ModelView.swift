//
//  ModelView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/14/23.
//

import SwiftUI

struct ModelView: View {
    @EnvironmentObject var modelState: ModelState
    @EnvironmentObject var chatState: ChatState
    @Binding var isRemoving: Bool
    @State var isSelected: Bool = false
    @State var alertDelete: Bool = false
    
    var body: some View {
        VStack(alignment: .leading) {
            if (modelState.modelInitState == .finished) {
                NavigationLink(
                    destination:
                        ChatView()
                        .environmentObject(chatState)
                        .onAppear {
                            modelState.startChat(chatState: chatState)
                        }
                ) {
                    HStack {
                        Text(modelState.modelConfig.localID)
                        Spacer()
                        if chatState.isCurrentModel(localId: modelState.modelConfig.localID) {
                            Image(systemName: "checkmark").foregroundColor(.blue)
                        }
                    }
                }.buttonStyle(.borderless)
            } else {
                Text(modelState.modelConfig.localID).opacity(0.5)
            }
            HStack{
                if modelState.modelInitState != .finished || isRemoving {
                    ProgressView(value: Double(modelState.progress) / Double(modelState.total))
                        .progressViewStyle(.linear)
                }
                if (modelState.modelInitState == .paused) {
                    Button() {
                        modelState.handleStart()
                    } label: {
                        Image(systemName: "icloud.and.arrow.down")
                    }
                    .buttonStyle(.borderless)
                } else if (modelState.modelInitState == .downloading) {
                    Button() {
                        modelState.handlePause()
                    } label: {
                        Image(systemName: "stop.circle")
                    }
                    .buttonStyle(.borderless)
                } else if (modelState.modelInitState == .failed) {
                    Image(systemName: "exclamationmark.triangle").foregroundColor(.red)
                }
                
                if isRemoving {
                    Button(role: .destructive) {
                        alertDelete = true
                    } label: {
                        Image(systemName: "trash")
                    }.alert("Delete Model", isPresented: $alertDelete) {
                        Button("Delete Model", role: .destructive) {
                            modelState.handleDelete()
                        }
                        .disabled(
                            modelState.modelInitState != .downloading &&
                                  modelState.modelInitState != .paused &&
                                  modelState.modelInitState != .finished &&
                                  modelState.modelInitState != .failed
                        )
                        Button("Clear Data") {
                            modelState.handleClear()
                        }
                        .disabled(
                            modelState.modelInitState != .downloading &&
                                  modelState.modelInitState != .paused &&
                                  modelState.modelInitState != .finished)
                        Button("Cancel", role: .cancel) {
                            alertDelete = false
                        }
                    } message: {
                        Text("Delete model will delete the all files with model config, and delete the entry in list. \n Clear model will keep the model config only, and keep the entry in list for future re-downloading.")
                    }.buttonStyle(.borderless)
                }
            }
        }
    }
}
