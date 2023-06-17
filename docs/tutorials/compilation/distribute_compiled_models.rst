Distribute Compiled Models
==========================

When you want to run the model compiled by yourself on mobile devices and/or web browser, you need to distribute the model you compiled to the Internet (for example, as a repository in Hugging Face), so that the applications released by MLC LLM can download your model from the Internet location.

This section introduces how to distribute the model you compiled.

.. note::
    Before proceeding, you should first have the model compiled manually.
    At this moment, the iOS/Android/web app released by MLC LLM only support **specific model architectures with specific quantization modes**. Particularly,

    - the :ref:`released iOS/iPadOS app <get_started>` supports models structured by LLaMA-7B and quantized by ``q3f16_0``, and models structured by GPT-NeoX-3B and quantized by ``q4f16_0``.
    - the :ref:`released Android app <get_started>` supports models structured by LLaMA-7B/GPT-NeoX-3B and quantized by ``q4f16_0``.
    - the `Web LLM demo page <https://mlc.ai/web-llm/>`_ supports models structured by LLaMA-7B and quantized by ``q4f32_0``, and models structured by GPT-NeoX-3B and quantized by both ``q4f16_0`` and ``q4f32_0``.

    If you have not compiled the model with supported quantization mode(s), please refer to :doc:`the model compile tutorial </tutorials/compilation/compile_models>` to compile the model with supported quantization modes and then come back to this page. You also need to build the iOS/Android app manually when the quantization mode you used is not supported by the iOS/Android app prebuilt by MLC LLM.

Assume you have compiled the model ``vicuna-v1-7b`` with quantization mode ``q4f32_0``.

1. To begin with, you can optionally :doc:`update the MLCChat configuration JSON </get_started/mlc_chat_config>` file ``dist/vicuna-v1-7b-q4f32_0/params/mlc-chat-config.json``. You can also use the default configuration, in which case no action is needed.
2. Then you need to distribute the directory ``dist/vicuna-v1-7b-q4f32_0/params`` (including all its contents) to an Internet location that is publicly accessible. For example, we have distributed a few prebuilt model weight directory (`example 1 <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0/tree/main>`_, `example 2 <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0/tree/main>`_) to Hugging Face.
3. If you are *building WebLLM* on your own, you also need to distribute the compiled model libraries (``.wasm`` file) to Internet. For example, we have distributed the prebuilt `vicuna-v1-7b-q4f32_0 Wasm file <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/vicuna-v1-7b-q4f32_0-webgpu.wasm>`_ and `RedPajama-INCITE-Chat-3B-v1-q4f16_0 Wasm file <https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-webgpu.wasm>`_ to GitHub.

This concludes the model distribution, and you can proceed to other sections to run models on your devices.
