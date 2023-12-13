🚧 Define New Model Architectures
=================================

This page guides you how to add a new model architecture in MLC.

Tutorial coming up soon. As of now, you can refer to
the `GPTNeoX PR <https://github.com/mlc-ai/mlc-llm/pull/1408>`_,
the `GPT-2 PR <https://github.com/mlc-ai/mlc-llm/pull/1314>`_,
or the `Mistral PR <https://github.com/mlc-ai/mlc-llm/pull/1230>`_.

Here is a tutorial for adding a model architecture in the previous workflow (similar concept but
different workflow):
https://github.com/mlc-ai/notebooks/blob/main/tutorial/How_to_add_model_architeture_in_MLC_LLM.ipynb

.. note:: 

    As mentioned in :ref:`Model Prebuilts`, when adding a model variant that has
    its architecture already supported in mlc-llm , you **only need to convert weights** 
    (e.g. adding ``CodeLlama`` when MLC supports ``llama-2``; adding ``OpenHermes Mistral``
    when MLC supports ``mistral``). On the other hand, a new model architecture
    (or inference logic) requires more work.