//
//  ChatView.swift
//  MLCChat
//

import SwiftUI
import GameController

struct ChatView: View {
    @EnvironmentObject private var chatState: ChatState

    @State private var inputMessage: String = ""
    @FocusState private var inputIsFocused: Bool
    @Environment(\.dismiss) private var dismiss
    @Namespace private var messagesBottomID

    // vision-related properties
    @State private var showActionSheet: Bool = false
    @State private var showImagePicker: Bool = false
    @State private var imageConfirmed: Bool = false
    @State private var imageSourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var image: UIImage?
    
    var body: some View {
        VStack {
            modelInfoView
            messagesView
            uploadImageView
            messageInputView
        }
        .navigationBarTitle("MLC Chat: \(chatState.displayName)", displayMode: .inline)
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.backward")
                }
                .buttonStyle(.borderless)
                .disabled(!chatState.isInterruptible)
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Reset") {
                    image = nil
                    imageConfirmed = false
                    chatState.requestResetChat()
                }
                .padding()
                .disabled(!chatState.isResettable)
            }
        }
    }
}

private extension ChatView {
    var modelInfoView: some View {
        Text(chatState.infoText)
            .multilineTextAlignment(.center)
            .opacity(0.5)
            .listRowSeparator(.hidden)
    }

    var messagesView: some View {
        ScrollViewReader { scrollViewProxy in
            ScrollView {
                VStack {
                    let messageCount = chatState.messages.count
                    let hasSystemMessage = messageCount > 0 && chatState.messages[0].role == MessageRole.bot
                    let startIndex = hasSystemMessage ? 1 : 0

                    // display the system message
                    if hasSystemMessage {
                        MessageView(role: chatState.messages[0].role, message: chatState.messages[0].message)
                    }

                    // display image
                    if let image, imageConfirmed {
                        ImageView(image: image)
                    }

                    // display conversations
                    ForEach(chatState.messages[startIndex...], id: \.id) { message in
                        MessageView(role: message.role, message: message.message)
                    }
                    HStack { EmptyView() }
                        .id(messagesBottomID)
                }
            }
            .onChange(of: chatState.messages) { _ in
                withAnimation {
                    scrollViewProxy.scrollTo(messagesBottomID, anchor: .bottom)
                }
            }
        }
    }

    @ViewBuilder
    var uploadImageView: some View {
        if chatState.useVision && !imageConfirmed {
            if image == nil {
                Button("Upload picture to chat") {
                    showActionSheet = true
                }
                .actionSheet(isPresented: $showActionSheet) {
                    ActionSheet(title: Text("Choose from"), buttons: [
                        .default(Text("Photo Library")) {
                            showImagePicker = true
                            imageSourceType = .photoLibrary
                        },
                        .default(Text("Camera")) {
                            showImagePicker = true
                            imageSourceType = .camera
                        },
                        .cancel()
                    ])
                }
                .sheet(isPresented: $showImagePicker) {
                    ImagePicker(image: $image,
                                showImagePicker: $showImagePicker,
                                imageSourceType: imageSourceType)
                }
                .disabled(!chatState.isUploadable)
            } else {
                VStack {
                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .frame(width: 300, height: 300)

                        HStack {
                            Button("Undo") {
                                self.image = nil
                            }
                            .padding()

                            Button("Submit") {
                                imageConfirmed = true
                                chatState.requestProcessImage(image: image)
                            }
                            .padding()
                        }
                    }
                }
            }
        }
    }

    var messageInputView: some View {
        HStack {
            TextField("Inputs...", text: $inputMessage, axis: .vertical)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .frame(minHeight: CGFloat(30))
                .focused($inputIsFocused)
                .onSubmit {
                    let isKeyboardConnected = GCKeyboard.coalesced != nil
                    if isKeyboardConnected {
                        send()
                    }
                }
            Button("Send") {
                send()
            }
            .bold()
            .disabled(!(chatState.isChattable && inputMessage != ""))
        }
        .frame(minHeight: CGFloat(70))
        .padding()
    }

    func send() {
        inputIsFocused = false
        chatState.requestGenerate(prompt: inputMessage)
        inputMessage = ""
    }
}
