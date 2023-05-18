//
//  ChatView.swift

import SwiftUI


struct ChatView: View {
    @State var inputMessage: String = ""
    @FocusState private var inputIsFocused: Bool;
    @EnvironmentObject var state: ChatState
    @Namespace var bottomID;
    @Namespace var infoID;

    init() {
        // UIScrollView.appearance().bounces = false
    }

    var body: some View {
        NavigationView {
            VStack {
                Text(state.infoText)
                    .multilineTextAlignment(.center)
                    .opacity(0.5)
                    .listRowSeparator(.hidden)
                ScrollViewReader { proxy in
                    ScrollView {
                        // Hack:rotate and reverse the inner view
                        // then rotate inverse the scroll
                        // so the result auto-scrolls
                        //
                        // This works more smoothly than scrollTo
                        // when there are a lot of text
                        if (state.unfinishedRespondMessage != "") {
                            MessageView(
                                role: state.unfinishedRespondRole,
                                message: state.unfinishedRespondMessage
                            ).rotationEffect(.radians(.pi))
                            .scaleEffect(x: -1, y: 1, anchor: .center)
                        }
                        ForEach(state.messages.reversed()) { msg in
                            MessageView(role: msg.role, message: msg.message)
                                .rotationEffect(.radians(.pi))
                                .scaleEffect(x: -1, y: 1, anchor: .center)
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
                        generateMessage()
                    }.bold().opacity(state.inProgress ? 0.5 : 1)
                }.frame(minHeight: CGFloat(70)).padding()
            }
            .navigationBarTitle("MLC Chat: " + state.modelName, displayMode: .inline)
            .toolbar{
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Reset") {
                        resetChat()
                    }
                    .opacity(0.9)
                    .padding()
                }
            }
        }
    }

    func generateMessage()  {
        if (inputMessage == "") {
            return
        }
        if (!state.inProgress) {
            state.generate(prompt: inputMessage)
            inputMessage = ""
        }
    }

    func resetChat()  {
        state.resetChat()
    }
}

struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        ChatView()
            .environmentObject(ChatState())
    }
}
