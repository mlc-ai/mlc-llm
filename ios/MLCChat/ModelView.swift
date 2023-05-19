//
//  ModelView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/14/23.
//

import SwiftUI

struct ModelView: View {
    @EnvironmentObject var state: ModelState
    
    var body: some View {
        VStack {
            HStack {
                if (state.modelInitState == .Finished) {
                    NavigationLink(
                        destination:
                            ChatView()
                            .environmentObject(state.chatState)
                            .onAppear {
                                state.reloadChatStateWithThisModel()
                            }
                    ) {
                        Text(state.modelConfig.local_id)
                    }
                } else {
                    Text(state.modelConfig.local_id)
                    .opacity(0.5)
                    Spacer()
                    if (state.modelInitState == .Paused) {
                        Button() {
                            state.handleStart()
                        } label: {
                            Image(systemName: "icloud.and.arrow.down")
                        }
                        .buttonStyle(.borderless)
                    }
                    if (state.modelInitState == .Downloading) {
                        Button() {
                            state.handlePause()
                        } label: {
                            Image(systemName: "stop.circle")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            if (!(state.modelInitState == .Finished || (state.modelInitState == .Paused && state.progress == 0))) {
                ProgressView(value: Double( state.progress) / Double(state.total))
                    .progressViewStyle(.linear)
                
            }
        }
    }
}
