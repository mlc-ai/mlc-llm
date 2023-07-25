# Node/Javascript Access Examples for MLC_CHAT REST APIs

First install dependencies.

`npm i`

Copy `dotenv.exmaple` to `.env`.

To run JS chat completion (both streaming and non-streaming) example: 

`node sample_client.js`

To run OpenAI (chat completion streaming and non-streaming, and legacy completion) example:

`node sample_openai.js`

Note: Langchain example will require an upstream change to expose the OPENAI_API_BASE path, currently not included.
