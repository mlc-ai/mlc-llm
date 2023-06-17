Use MLC-Chat with Javascript APIs
=================================

.. contents:: Table of Contents
   :local:
   :depth: 2

MLC-Chat also provide its Javascript bindings (`WebLLM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_) to allow user to use MLC-Chat in your web application.

Install WebLLM NPM Package
--------------------------

.. code:: bash

  npm i @mlc-ai/web-llm

🚧 API References
-----------------

Please refer to the source code of the ChatModule at `this link <https://github.com/mlc-ai/web-llm/blob/main/src/chat_module.ts>` to examine the function interface.

Use WebLLM API in your own program
----------------------------------

Below is a simple example to use WebLLM API in your own Typescript program.

.. code:: typescript

  import * as webllm from "@mlc-ai/web-llm";

  function setLabel(id: string, text: string) {
    const label = document.getElementById(id);
    if (label == null) {
      throw Error("Cannot find label " + id);
    }
    label.innerText = text;
  }

  async function main() {
    const chat = new webllm.ChatModule();

    chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
      setLabel("init-label", report.text);
    });

    await chat.reload("vicuna-v1-7b-q4f32_0");

    const generateProgressCallback = (_step: number, message: string) => {
      setLabel("generate-label", message);
    };

    const prompt0 = "What is the capital of Canada?";
    setLabel("prompt-label", prompt0);
    const reply0 = await chat.generate(prompt0, generateProgressCallback);
    console.log(reply0);

    const prompt1 = "Can you write a poem about it?";
    setLabel("prompt-label", prompt1);
    const reply1 = await chat.generate(prompt1, generateProgressCallback);
    console.log(reply1);

    console.log(await chat.runtimeStatsText());
  }

  main();
