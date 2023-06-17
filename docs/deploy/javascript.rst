WebLLM and Javascript API
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2

WebLLM is a MLC chat webruntime (`WebLLM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_)
that allows you to build chat applications directly in browser.

Try out Prebuilt Webpage
------------------------

To get started, you can try out `WebLLM prebuilt webpage <https://mlc.ai/webllm>`__.

A WebGPU-compatible browser and a local GPU are needed to run WebLLM.
You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__
to verify the functionality of WebGPU on your browser.


Use WebLLM NPM Package
----------------------

WebLLM is available as a npm package.
Below is a simple example to use WebLLM API in your own Typescript program.
You can follow the instructions in  `get-started <https://github.com/mlc-ai/web-llm/tree/main/examples/get-started>`__
to run the example.

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

Build a Chat App
----------------
You can find a complete a complete chat app example in `simple-chat <https://github.com/mlc-ai/web-llm/tree/main/examples/simple-chat>`__.