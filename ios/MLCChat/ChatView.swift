//
//  ChatView.swift

import SwiftUI


struct ChatView: View {
    @State var inputMessage: String = ""
    @FocusState private var inputIsFocused: Bool;
    @EnvironmentObject var chatState: ChatState
    @Environment(\.dismiss) private var dismiss
    
    init() {}
    
    var body: some View {
        VStack {
            Text(chatState.infoText)
                .multilineTextAlignment(.center)
                .opacity(0.5)
                .listRowSeparator(.hidden)
            ScrollViewReader { scrollView in
                ScrollView {
                    VStack {
                        ForEach(chatState.messages.reversed()) {
                            message in
                            MessageView(role: message.role, message: message.message).rotationEffect(.radians(.pi))
                                .scaleEffect(x: -1, y: 1, anchor: .center)
                        }
                    }.id("MessageHistory")
                    
                }.onChange(of: chatState.messages) { _ in
                    withAnimation{
                        scrollView.scrollTo("MessageHistory", anchor: .top)
                    }
                }.rotationEffect(.radians(.pi))
                    .scaleEffect(x: -1, y: 1, anchor: .center)
            }
            
            HStack {
                TextField("Inputs...", text: $inputMessage, axis: .vertical)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(minHeight: CGFloat(30))
                    .focused($inputIsFocused)
                Button("Send") {
                    self.inputIsFocused = false
                    chatState.requestGenerate(prompt: inputMessage)
                    inputMessage = ""
                }.bold().disabled(!(chatState.chattable() && inputMessage != ""))
            }.frame(minHeight: CGFloat(70)).padding()
        }
        .navigationBarTitle("MLC Chat: " + chatState.displayName, displayMode: .inline)
        .navigationBarBackButtonHidden()
        .toolbar{
            ToolbarItem(placement: .navigationBarLeading) {
                Button() {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.backward")
                }
                .buttonStyle(.borderless).disabled(!chatState.interruptable())
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Reset") {
                    chatState.requestResetChat()
                }
                .padding()
                .disabled(!chatState.resettable())
            }
        }
    }
}
