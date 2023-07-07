//
//  ChatView.swift

import SwiftUI


struct ChatView: View {
    @State var inputMessage: String = ""
    @FocusState private var inputIsFocused: Bool;
    @EnvironmentObject var chatState: ChatState
    @Environment(\.dismiss) private var dismiss
    // vision-related properties
    @State private var showActionSheet: Bool = false
    @State private var showImagePicker: Bool = false
    @State private var imageConfirmed: Bool = false
    @State private var imageSourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var image: UIImage?
    
    var body: some View {
        VStack {
            Text(chatState.infoText)
                .multilineTextAlignment(.center)
                .opacity(0.5)
                .listRowSeparator(.hidden)
            ScrollViewReader { scrollView in
                ScrollView {
                    VStack {
                        let messageLen = chatState.messages.count
                        let isolateSystemMessage = messageLen > 0 && chatState.messages[0].role == MessageRole.bot
                        let startIdx = isolateSystemMessage ? 1 : 0
                        // display conversations
                        ForEach(chatState.messages[startIdx...].reversed()) {
                            message in
                            MessageView(role: message.role, message: message.message).rotationEffect(.radians(.pi))
                                .scaleEffect(x: -1, y: 1, anchor: .center)
                        }
                        // display image
                        if image != nil && imageConfirmed {
                            ImageView(image: image!).rotationEffect(.radians(.pi))
                                .scaleEffect(x: -1, y: 1, anchor: .center)
                        }
                        // display the system message
                        if isolateSystemMessage {
                            MessageView(role: chatState.messages[0].role, message: chatState.messages[0].message)
                                .rotationEffect(.radians(.pi))
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
            
            if chatState.useVision && !imageConfirmed {
                if image == nil {
                    Button("Upload picture to chat") {
                        self.showActionSheet = true
                    }.actionSheet(isPresented: $showActionSheet) {
                        ActionSheet(title: Text("Choose from"), buttons: [
                            .default(Text("Photo Library")) {
                                self.showImagePicker = true
                                self.imageSourceType = .photoLibrary
                            },
                            .default(Text("Camera")) {
                                self.showImagePicker = true
                                self.imageSourceType = .camera
                            },
                            .cancel()
                      ])
                    }.sheet(isPresented: $showImagePicker) {
                        ImagePicker(image: self.$image, showImagePicker: self.$showImagePicker, imageSourceType: self.imageSourceType)
                    }.disabled(!chatState.uploadable())
                } else {
                    VStack {
                        Image(uiImage: image!)
                            .resizable()
                            .frame(width: 300, height: 300)
                        HStack {
                            Button("Undo") {
                                self.image = nil
                            }.padding()
                            Button("Submit") {
                                self.imageConfirmed = true
                                self.chatState.requestProcessImage(image: image!)
                            }.padding()
                        }
                    }
                }
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
                    self.image = nil
                    self.imageConfirmed = false
                    chatState.requestResetChat()
                }
                .padding()
                .disabled(!chatState.resettable())
            }
        }
    }
}
