package ai.mlc.mlcchat;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

public class MessageAdapter extends BaseAdapter {
    private List<MessageData> messages;
    private Context context;

    public MessageAdapter(Context c) {
        super();
        context = c;
        messages = new ArrayList<MessageData>();
    }

    public void appendMessage(MessageData m) {
        messages.add(m);
        notifyDataSetChanged();
    }

    public void updateMessage(MessageData m) {
        messages.set(messages.size() - 1, m);
        notifyDataSetChanged();
    }

    public boolean isBotLast() {
        return messages.get(messages.size() - 1).isBot();
    }

    public int size() {
        return messages.size();
    }

    public void reset() {
        messages.clear();
        notifyDataSetChanged();
    }

    @Override
    public int getCount() {
        return messages.size();
    }

    @Override
    public Object getItem(int position) {
        return messages.get(position);
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        MessageViewHolder holder = new MessageViewHolder();
        LayoutInflater inflater = LayoutInflater.from(context);
        MessageData chatMessage = messages.get(position);
        if (chatMessage.isBot()) {
            convertView = inflater.inflate(R.layout.bot_message, null);
            holder.textView = convertView.findViewById(R.id.bot_message);
            holder.textView.setText(chatMessage.getText());
            convertView.setTag(holder);
        } else {
            convertView = inflater.inflate(R.layout.user_message, null);
            holder.textView = convertView.findViewById(R.id.user_message);
            holder.textView.setText(chatMessage.getText());
            convertView.setTag(holder);
        }
        return convertView;
    }

    static class MessageViewHolder {
        public TextView textView;
    }
}
