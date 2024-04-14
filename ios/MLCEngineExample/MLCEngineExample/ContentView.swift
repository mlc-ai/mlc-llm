// This is a minimum example App to interact with MLC Engine
//
// for a complete example, take a look at the MLCChat

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    // simply display text on the app
    var body: some View {
        HStack {
            Text(appState.displayText)
            Spacer()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
