.. _package-libraries-and-weights:

Package Libraries and Weights
=============================

When we want to build LLM applications with MLC LLM (e.g., iOS/Android apps),
usually we need to build static model libraries and app binding libraries,
and sometimes bundle model weights into the app.
MLC LLM provides a tool for fast model library and weight packaging: ``mlc_llm package``.

This page briefly introduces how to use ``mlc_llm package`` for packaging.
Tutorials :ref:`deploy-ios` and :ref:`deploy-android` contain detailed examples and instructions
on using this packaging tool for iOS and Android deployment.

-----

Introduction
------------

To use ``mlc_llm package``, we must clone the source code of `MLC LLM <https://github.com/mlc-ai/mlc-llm>`_
and `install the MLC LLM and TVM Unity package <https://llm.mlc.ai/docs/install/mlc_llm.html#option-1-prebuilt-package>`_.
Depending on the app we build, there might be some other dependencies, which are described in
corresponding :ref:`iOS <deploy-ios>` and :ref:`Android <deploy-android>` tutorials.

After cloning, the basic usage of ``mlc_llm package`` is as the following.

.. code:: bash

    export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm
    cd /path/to/app  # The app root directory which contains "mlc-package-config.json".
                     # E.g., "ios/MLCChat" or "android/MLCChat"
    mlc_llm package

**The package command reads from the JSON file** ``mlc-package-config.json`` **under the current directory.**
The output of this command is a directory ``dist/``,
which contains the packaged model libraries (under ``dist/lib/``) and weights (under ``dist/bundle/``).
This directory contains all necessary data for the app build.
Depending on the app we build, the internal structure of ``dist/lib/`` may be different.

.. code::

   dist
   ├── lib
   │   └── ...
   └── bundle
       └── ...

The input ``mlc-package-config.json`` file specifies

* the device (e.g., iPhone or Android) to package model libraries and weights for,
* the list of models to package.

Below is an example ``mlc-package-config.json`` file:

.. code:: json

    {
        "device": "iphone",
        "model_list": [
            {
                "model": "HF://mlc-ai/Mistral-7B-Instruct-v0.2-q3f16_1-MLC",
                "model_id": "Mistral-7B-Instruct-v0.2-q3f16_1",
                "estimated_vram_bytes": 3316000000,
                "bundle_weight": true,
                "overrides": {
                    "context_window_size": 512
                }
            },
            {
                "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
                "model_id": "gemma-2b-q4f16_1",
                "estimated_vram_bytes": 3000000000,
                "overrides": {
                    "prefill_chunk_size": 128
                }
            }
        ]
    }

This example ``mlc-package-config.json`` specifies "iphone" as the target device.
In the ``model_list``,

* ``model`` points to the Hugging Face repository which contains the pre-converted model weights. Apps will download model weights from the Hugging Face URL.
* ``model_id`` is a unique model identifier.
* ``estimated_vram_bytes`` is an estimation of the vRAM the model takes at runtime.
* ``"bundle_weight": true`` means the model weights of the model will be bundled into the app when building.
* ``overrides`` specifies some model config parameter overrides.


Below is a more detailed specification of the ``mlc-package-config.json`` file.
Each entry in ``"model_list"`` of the JSON file has the following fields:

``model``
   (Required) The path to the MLC-converted model to be built into the app.

   Usually it is a Hugging Face URL (e.g., ``"model": "HF://mlc-ai/phi-2-q4f16_1-MLC"```) that contains the pre-converted model weights.
   For iOS, it can also be a path to a local model directory which contains converted model weights (e.g., ``"model": "../dist/gemma-2b-q4f16_1"``).
   Please check out :ref:`convert-weights-via-MLC` if you want to build local model into the app.

``model_id``
  (Required) A unique local identifier to identify the model.
  It can be an arbitrary one.

``estimated_vram_bytes``
   (Required) Estimated requirements of vRAM to run the model.

``bundle_weight``
   (Optional) A boolean flag indicating whether to bundle model weights into the app.
   If this field is set to true, the ``mlc_llm package`` command will copy the model weights
   to ``dist/bundle/$model_id``.

``overrides``
   (Optional) A dictionary to override the default model context window size (to limit the KV cache size) and prefill chunk size (to limit the model temporary execution memory).
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "overrides": {
                     "context_window_size": 512,
                     "prefill_chunk_size": 128
                  }
            }
         ]
      }

``model_lib``
   (Optional) A string specifying the system library prefix to use for the model.
   Usually this is used when you want to build multiple model variants with the same architecture into the app.
   **This field does not affect any app functionality.**
   The ``"model_lib_path_for_prepare_libs"`` introduced below is also related.
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "model_lib": "gpt_neox_q4f16_1"
            }
         ]
      }


Besides ``model_list`` in ``MLCChat/mlc-package-config.json``,
you can also **optionally** specify a dictionary of ``"model_lib_path_for_prepare_libs"``,
**if you want to use model libraries that are manually compiled**.
The keys of this dictionary should be the ``model_lib`` that specified in model list,
and the values of this dictionary are the paths (absolute, or relative) to the manually compiled model libraries.
The model libraries specified in ``"model_lib_path_for_prepare_libs"`` will be built into the app when running ``mlc_llm package``.
Example:

.. code:: json

   {
      "device": "iphone",
      "model_list": [
         {
               "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
               "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
               "estimated_vram_bytes": 2960000000,
               "model_lib": "gpt_neox_q4f16_1"
         }
      ],
      "model_lib_path_for_prepare_libs": {
         "gpt_neox_q4f16_1": "../../dist/lib/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar"
      }
   }

Compilation Cache
-----------------
``mlc_llm package`` leverage a local JIT cache to avoid repetitive compilation of the same input.
It also leverages a local cache to download weights from remote. These caches
are shared across the entire project. Sometimes it is helpful to force rebuild when
we have a new compiler update or when something goes wrong with the cached library.
You can do so by setting the environment variable ``MLC_JIT_POLICY=REDO``

.. code:: bash

   MLC_JIT_POLICY=REDO mlc_llm package

Arguments of ``mlc_llm package``
--------------------------------

Command ``mlc_llm package`` can optionally take the arguments below:

``--package-config``
    A path to ``mlc-package-config.json`` which contains the device and model specification.
    By default, it is the ``mlc-package-config.json`` under the current directory.

``--mlc-llm-source-dir``
    The path to MLC LLM source code (cloned from https://github.com/mlc-ai/mlc-llm).
    By default, it is the ``$MLC_LLM_SOURCE_DIR`` environment variable.
    If neither ``$MLC_LLM_SOURCE_DIR`` or ``--mlc-llm-source-dir`` is specified, error will be reported.

``--output`` / ``-o``
    The output directory of ``mlc_llm package`` command.
    By default, it is ``dist/`` under the current directory.


Summary and What to Do Next
---------------------------

In this page, we introduced the ``mlc_llm package`` command for fast model library and weight packaging.

* It takes input file ``mlc-package-config.json`` which contains the device and model specification for packaging.
* It outputs directory ``dist/``, which contains packaged libraries under ``dist/lib/`` and model weights under ``dist/bundle/``.

Next, please feel free to check out the :ref:`iOS <deploy-ios>` and :ref:`Android <deploy-android>` tutorials for detailed examples of using ``mlc_llm package``.
