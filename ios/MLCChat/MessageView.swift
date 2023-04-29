//
//  MessageView.swift


import SwiftUI

struct MessageView: View {
    var role: MessageRole;
    var message: String
    
    var body: some View {
        let textColor = role == MessageRole.user ? Color.white : Color(UIColor.label)
        let background = role == MessageRole.user ? Color.blue : Color(
            UIColor.secondarySystemBackground
        )
        HStack {
            if (role == MessageRole.user) {
                Spacer()
            }
            Text(message)
                .padding(10)
                .foregroundColor(textColor)
                .background(background)
                .cornerRadius(10)
                .textSelection(.enabled)
            if (role != MessageRole.user) {
                Spacer()
            }
        }
        .padding()
        .listRowSeparator(.hidden)
    }
}

struct MessageView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            VStack (spacing: 0){
                ScrollView {
                    MessageView(role: MessageRole.user, message: "Message 1")
                    MessageView(role: MessageRole.bot, message: "Message 2")
                    MessageView(role: MessageRole.user, message: "Message 3")
                }
            }
        }
    }
}
