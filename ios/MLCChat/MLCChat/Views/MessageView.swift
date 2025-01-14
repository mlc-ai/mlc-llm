//
//  MessageView.swift
//  MLCChat
//

import SwiftUI
import MarkdownUI

struct MessageView: View {
    let role: MessageRole;
    let message: String
    let isMarkdownSupported: Bool

    @State private var showMarkdown: Bool

    init(role: MessageRole, message: String, isMarkdownSupported: Bool = true) {
        self.role = role
        self.message = message
        self.isMarkdownSupported = isMarkdownSupported
        _showMarkdown = State(initialValue: isMarkdownSupported)
    }
    var body: some View {
        let textColor = role.isUser ? Color.white : Color(UIColor.label)
        let background = role.isUser ? Color.blue : Color(UIColor.secondarySystemBackground)

        HStack {
            if role.isUser {
                Spacer()
                Text(message)
                    .padding(10)
                    .foregroundColor(textColor)
                    .background(background)
                    .cornerRadius(10)
                    .textSelection(.enabled)
            }
            if !role.isUser {
                VStack(alignment: .leading) {
                    // Toggle switch to show/hide Markdown
                    if(isMarkdownSupported){
                        Toggle(isOn: $showMarkdown) {
                            Text("Show as Markdown")
                                .font(.footnote)
                                .foregroundColor(.blue)
                        }
                        .padding(.bottom, 10)
                    }

                    // Conditionally display Text or Markdown
                    if showMarkdown {
                        Markdown {
                            message
                        }
                        .markdownTheme(.gitHub)
                        .padding(10)
                        .foregroundColor(textColor)
                        .background(background)
                        .cornerRadius(10)
                        .textSelection(.enabled)
                    } else {
                        Text(message)
                            .padding(10)
                            .foregroundColor(textColor)
                            .background(background)
                            .cornerRadius(10)
                            .textSelection(.enabled)
                    }
                }
                Spacer()
            }
        }
        .padding()
        .listRowSeparator(.hidden)
    }
}

struct ImageView: View {
    let image: UIImage

    var body: some View {
        let background = Color.blue
        HStack {
            Spacer()
            Image(uiImage: image)
                .resizable()
                .frame(width: 150, height: 150)
                .padding(15)
                .background(background)
                .cornerRadius(20)
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
                    MessageView(role: MessageRole.assistant, message: "Message 2")
                    MessageView(role: MessageRole.user, message: "Message 3")
                }
            }
        }
    }
}
