import { OpenAI } from "langchain/llms/openai";
import { BufferWindowMemory } from "langchain/memory";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import {TextLoader } from "langchain/document_loaders/fs/text";
import { loadQAStuffChain } from "langchain/chains";

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

function print(str: string) {
    process.stdout.write(str);
}

const newline = () => {
    print('\n');
}

  const chat = new OpenAI( {
      openAIApiKey: "empty",
      temperature: 0
    },   {
        basePath: 'http://127.0.0.1:8000/v1'
    });

// Conversational LLMChain example
  const memory = new BufferWindowMemory({ memoryKey: "history", k: 1 });

  const template = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {human_input}
    AI:`;


  const prompt = PromptTemplate.fromTemplate(template);
  let chain = new LLMChain({ llm: chat, prompt, memory });

  let input = "Write a poem about Pittsburgh.";
  print(color.BOLD + input + "..." + color.END);
  newline();
  let res = await chain.call({ human_input:  input });
  newline();
  print(color.GREEN + res.text + color.END);
  newline();
  input = "What does it mean?";
  print(color.BOLD + input + "..." + color.END);
  newline();
  res = await chain.call({ human_input: input });
  newline();
  print(color.GREEN + res.text + color.END);
  newline();

// Question and answer stuff chain example with text loader
const loader = new TextLoader('../resources/linux.txt');
const documents = await loader.load();
const schain =  loadQAStuffChain(chat);
const query = "When was Linux released?";
newline(); newline();
print(color.BOLD + "Query: " + color.END + color.BLUE + query + color.END);
newline();
const result = await schain.call({ input_documents: documents,  question: query});
print(color.BOLD + "Response: " + color.END +  color.GREEN + result.text  + color.END);
