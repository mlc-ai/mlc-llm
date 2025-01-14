import { Configuration, OpenAIApi }  from "openai";
import dotenv from "dotenv";
dotenv.config();

( async () =>  {

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
    basePath : process.env.OPENAI_API_BASE
})
const openai = new OpenAIApi(configuration);
let model = "vicuna-v1-7b"

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

const print = ( str ) => {
    process.stdout.write(str);
};

const newline = () => {
    print('\n');
}

// Chat completion example without streaming
newline();
print(color.BOLD + "OpenAI chat completion example without streaming:" + color.END);
newline();

let completion = await openai.createChatCompletion({
  model: model,
  messages: [{"role": "user", "content": "Write a poem about OpenAI"}]
});


print(color.GREEN + completion.data.choices[0].message.content + color.END)
newline();  newline();


// Chat completion example with streaming
// (raw implementation since npm module does not support it yet - it will have support in upcoming 4.x)

print(color.BOLD + "OpenAI chat completion example with streaming:" + color.END);
newline();
completion = await openai.createChatCompletion({
    model: model,
    messages: [{"role": "user", "content": "Write a poem about OpenAI"}],
    stream: true,
}, {responseType: 'stream'});

completion.data.on('data', async (data) => {
        const parsed = JSON.parse(data.toString().substring(6));
        print(color.GREEN + parsed.choices[0].delta.content + color.END);
});

completion.data.on('close', async ()  => {
    newline(); newline();

    // Completion example
    print(color.BOLD + "OpenAI completion example:" + color.END)
    newline();
    let res = await openai.createCompletion({ prompt: "Write a poem about OpenAI", model: model});
    print(color.GREEN + res.data.choices[0].text + color.END);
    newline();  newline();

    });
})()
