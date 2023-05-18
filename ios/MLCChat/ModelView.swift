//
//  ModelView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/14/23.
//

import SwiftUI

struct ModelView: View {
    @EnvironmentObject var modelState: ModelState
    
    var body: some View {
        VStack {
            HStack {
                Text(modelState.modelConfig.local_id)
                Spacer()
                if (modelState.modelInitState == .Stopped) {
                    Button() {
                        modelState.start()
                    } label: {
                        Image(systemName: "icloud.and.arrow.down")
                    }
                    .buttonStyle(.borderless)
                }
                
                if (modelState.modelInitState == .Downloading) {
                    Button() {
                        modelState.stop()
                    } label: {
                        Image(systemName: "stop.circle")
                    }
                    .buttonStyle(.borderless)
                }
                if (modelState.modelInitState == .Finished) {
                    
                    Button (role: .destructive) {
                        
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                }
            }
            if (!(modelState.modelInitState == .Finished || (modelState.modelInitState == .Stopped && modelState.progress == 0))) {
                ProgressView(value: Double( modelState.progress) / Double(modelState.total))
                    .progressViewStyle(.linear)
                
            }
        }
    }
}
