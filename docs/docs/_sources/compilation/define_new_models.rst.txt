Define New Model Architectures
==============================

This page guides you how to add a new model architecture in MLC.

This notebook (runnable in Colab) should contain all necessary information to add a model in
MLC LLM:
https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_add_new_model_architecture_in_tvm_nn_module.ipynb

In the notebook, we leverage ``tvm.nn.module`` to define a model in MLC LLM. We also use ``JIT``
(just-in-time compilation) to debug the implementation.

You can also refer to the PRs below on specific examples of adding a model architecture in MLC LLM:

- `GPTNeoX PR <https://github.com/mlc-ai/mlc-llm/pull/1408>`_
- `GPT-2 PR <https://github.com/mlc-ai/mlc-llm/pull/1314>`_
- `Mistral PR <https://github.com/mlc-ai/mlc-llm/pull/1230>`_

.. note::

    When adding a model variant that has
    its architecture already supported in mlc-llm , you **only need to convert weights**
    (e.g. adding ``CodeLlama`` when MLC supports ``llama-2``; adding ``OpenHermes Mistral``
    when MLC supports ``mistral``). On the other hand, a new model architecture
    (or inference logic) requires more work (following the tutorial above).
