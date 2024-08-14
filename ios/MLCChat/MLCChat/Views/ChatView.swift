//
//  ChatView.swift
//  MLCChat
//

import SwiftUI
import GameController

struct ChatView: View {
    @EnvironmentObject private var chatState: ChatState
    @Environment(\.scenePhase) var scenePhase
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
        .onChange(of: scenePhase) { oldPhase, newPhase in
            if newPhase == .background {
                self.chatState.requestSwitchToBackground()
            }
        }
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
                    let messageCount = chatState.displayMessages.count
                    let hasSystemMessage = messageCount > 0 && chatState.displayMessages[0].role == MessageRole.assistant
                    let startIndex = hasSystemMessage ? 1 : 0

                    // display the system message
                    if hasSystemMessage {
                        MessageView(role: chatState.displayMessages[0].role, message: chatState.displayMessages[0].message, isMarkdownSupported: false)
                    }

                    // display image
                    if let image, imageConfirmed {
                        ImageView(image: image)
                    }

                    // display conversations
                    ForEach(chatState.displayMessages[startIndex...], id: \.id) { message in
                        MessageView(role: message.role, message: message.message)
                    }
                    HStack { EmptyView() }
                        .id(messagesBottomID)
                }
            }
            .onChange(of: chatState.displayMessages) { _ in
                withAnimation {
                    scrollViewProxy.scrollTo(messagesBottomID, anchor: .bottom)
                }
            }
        }
    }

    @ViewBuilder
    var uploadImageView: some View {
        if chatState.legacyUseImage && !imageConfirmed {
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
