# Node/Javascript/Typescript Access Examples for mlc_llm REST APIs

Please make sure you are running v18.17.x of node (and npm v9.6.7)  --  v20.x currently has some compatibility problems with typescript used in the langchain example.

First install dependencies.

`npm i`

Copy `dotenv.exmaple` to `.env`.

To run JS chat completion (both streaming and non-streaming) example:

`node sample_client.js`

To run OpenAI (chat completion streaming and non-streaming, and legacy completion) example:

`node sample_openai.js`

To run LangchainJS Typescript example:

`npm run example`
