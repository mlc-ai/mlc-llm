//
//  DownloadView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/11/23.
//

import SwiftUI

struct StartView: View {
    @EnvironmentObject var state: StartState
    
    var body: some View {
        NavigationStack {
            List{
                Section(header: Text("Models")){
                    ForEach(state.models) { modelState in
                        ModelView().environmentObject(modelState)
                    }
                }
            }.navigationTitle("MLC Chat")
        }
    }
}


struct StartView_Previews: PreviewProvider {
    static var previews: some View {
        StartView().environmentObject(StartState())
    }
}
