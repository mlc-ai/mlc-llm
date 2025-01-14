import request from 'needle';

( async () => {
const color = {
    PURPLE : '\x1b[95m',
    CYAN : '\x1b[96m',
    DARKCYAN : '\x1b[36m',
    BLUE : '\x1b[94m',
    GREEN : '\x1b[92m',
    YELLOW : '\x1b[93m',
    RED : '\x1b[91m',
    BOLD : '\x1b[1m',
    UNDERLINE : '\x1b[4m',
    END : '\x1b[0m'
};

let payload = {
    model : 'vicuna-v1-7b',
    messages: [{"role": "user", "content": "Write a haiku"}],
    stream: false
};

const print = ( str ) => {
    process.stdout.write(str);
};

const newline = () => {
    print('\n');
}

newline();
print(color.BOLD + "Without streaming:" + color.END);
newline();

let r = await request("post", "http://127.0.0.1:8000/v1/chat/completions", payload, {json: true});

print(color.GREEN + r.body.choices[0].message.content + color.END);
print('\n');
// Reset the chat
r = await request("post", "http://127.0.0.1:8000/v1/chat/completions", payload, {json: true});
print(color.BOLD + "Reset chat" + color.END);
newline();

// Get a response using a prompt with streaming

payload = {
    "model": "vicuna-v1-7b",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
}

print( color.BOLD + "With streaming:" + color.END);
newline();
r =  request.post( "http://127.0.0.1:8000/v1/chat/completions", payload, {json: true})
.on('readable', function() {
    let jsData = '';
    let data = '';
    while (data = this.read()) {
       const chunk = data.toString().substring(6);
       if (chunk.trim() === "[DONE]")  break;
       jsData = JSON.parse(chunk);
       print(color.GREEN + jsData.choices[0].delta.content + color.END);
    }
})
.on('done', async function () {
    newline();
    let txtresp = await request("get", "http://127.0.0.1:8000/stats");
    print(color.BOLD + "Runtime stats:" + color.END + txtresp.body);

})

})()
