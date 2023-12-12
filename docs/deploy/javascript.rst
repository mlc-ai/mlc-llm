.. _webllm-runtime:

WebLLM and Javascript API
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2

`WebLLM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ is an MLC chat web runtime
that allows you to build chat applications directly in the browser, leveraging
`WebGPU <https://www.w3.org/TR/webgpu/>`_ and providing users a natural layer of abstraction.

Try out the Prebuilt Webpage
----------------------------

To get started, you can try out `WebLLM prebuilt webpage <https://webllm.mlc.ai/#chat-demo>`__.

A WebGPU-compatible browser and a local GPU are needed to run WebLLM.
You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__
to verify the functionality of WebGPU on your browser.


Use WebLLM NPM Package
----------------------

WebLLM is available as an `npm package <https://www.npmjs.com/package/@mlc-ai/web-llm>`_.
The source code is available in `the WebLLM repo <https://github.com/mlc-ai/web-llm>`_,
where you can make your own modifications and build from source.

Note that the `WebLLM prebuilt webpage <https://webllm.mlc.ai/#chat-demo>`__ above
is powered by the WebLLM npm package, specifically with the code in
the `simple-chat <https://github.com/mlc-ai/web-llm/tree/main/examples/simple-chat>`__ example.

Below is a simple example to use WebLLM API in your own TypeScript program.
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

    await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

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

Build your Own Model/WASM
-------------------------
The above examples all utilize MLC's prebuilt model libraries (i.e. WASMs). For
models that are included in the list of prebuilt, you can first follow :ref:`install-web-build`,
and then follow :ref:`compile-models-via-MLC`.
