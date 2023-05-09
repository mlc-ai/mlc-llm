package ai.mlc.mlcchat;

import androidx.annotation.NonNull;

enum MessageRole {
    BOT, USER
}

public class MessageData {

    private final MessageRole messageRole;
    private final String text;

    public MessageData(MessageRole r, String t) {
        messageRole = r;
        text = t;
    }

    public boolean isBot() {
        return messageRole == MessageRole.BOT;
    }

    public String getText() {
        return text;
    }
}
