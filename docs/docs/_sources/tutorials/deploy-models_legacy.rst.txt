.. _How to Deploy Models:

How to Deploy Models
====================

In this tutorial, we will guide you on how to **deploy** models built by MLC LLM to different backends and devices.
We support four options of device categories for deployment: laptop/desktop, web browser, iPhone/iPad, and Android phones.

This page contains the following sections.
We first introduce how to prepare the (pre)built model libraries and weights, and one section for each backend device will then follow.

.. contents:: Table of Contents
    :depth: 1
    :local:


.. _clone_repo:

Clone the MLC LLM Repository
----------------------------

If you haven't already cloned the MLC-LLM repository, now is the perfect time to do so.

.. code:: shell

    git clone git@github.com:mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

.. _knowing-local-id:

Get to Know Model's Local ID
----------------------------

In MLC LLM, we use **local ids** to denote the models we build.

A model's local id is in the format ``MODELNAME-QUANTMODE`` (for example, ``vicuna-v1-7b-q3f16_0``, ``RedPajama-INCITE-Chat-3B-v1-q4f16_0``, etc.).
You can find the local id by checking the directory names under ``dist`` (if you built model on your own) or ``dist/prebuilt`` if you use prebuilt models:

.. code:: shell

    # Check id for models manually built.
    ~/mlc-llm > ls dist
    RedPajama-INCITE-Chat-3B-v1-q4f16_0     models                vicuna-v1-7b-q3f16_0
    RedPajama-INCITE-Chat-3B-v1-q4f32_0     prebuilt              vicuna-v1-7b-q4f32_0

    # Check id for prebuilt models.
    # Note: Local ids start with the model name after `mlc-chat-`.
    ~ > ls dist/prebuilt
    mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0    mlc-chat-vicuna-v1-7b-q3f16_0

We will use local ids for model deployment.


.. _prepare-weight-library:

Prepare Model Weight and Library
--------------------------------

In order to load the model correctly in deployment, we need to put the model libraries and weights to the right location before deployment.
You can select from the panel below for the preparation steps you will need.

.. tabs::

    .. tab:: Desktop/laptop

        .. tabs::

            .. tab:: Use prebuilt model

                MLC LLM provides a list of prebuilt models (check our :doc:`/model-prebuilts` for the list).
                If you want to use them, run the commands below under the root directory of MLC LLM to download the libraries and weights to the target location.

                .. code:: shell

                    # Make sure you have installed git-lfs.
                    mkdir -p dist/prebuilt
                    cd dist/prebuilt
                    git lfs install
                    git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git lib

                    # Choose one or both commands from below according to the model(s) you want to deploy.
                    git clone https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0
                    # and/or
                    git clone https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0

            .. tab:: Use model you built

                You are all good. There is no further action to prepare the model libraries and weights.

    .. tab:: iPhone/iPad/Android phones

        .. tabs::

            .. tab:: Use prebuilt model

                Please direct to the :ref:`iOS/iPadOS app download <iPhone-download-app>` and/or the :ref:`Android app download <Android-download-app>`.
                This allows you to use the prebuilt models in the application.

                .. note::
                    To build the iOS/iPadOS/Android app from source, you need first build the model manually.
                    Please refer to :ref:`the model build tutorial <How to Compile Models>`, build the model, and then come back to this page.

            .. tab:: Use model you built

                If you want to deploy the model you built on mobile devices via the released iOS/Android app, please go through the section about :ref:`uploading your model to the Internet <upload-model>`.

                If you want to build the iOS/Android app on your own, the upload action is optional.
                The "build application from source" sections for iPhone/iPad and Android provide more detailed instructions.

    .. tab:: Web browser

        .. tabs::

            .. tab:: Use prebuilt model

                .. .. tabs::

                ..     .. tab:: Local deployment

                ..         Please clone Web LLM

                TBA.

            .. tab:: Use model you built

                TBA.

.. _upload-model:

Upload the Model You Built to Internet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    If you only want to deploy the model to your desktop/laptop, you can skip this section.

When you want to deploy the model built by yourself on mobile devices and/or web browser, but do not want to build the iOS/Android/web application on your own, you need to upload the model you built to the Internet (for example, as a repository of Hugging Face), so that the applications released by MLC LLM can download your model from the Internet location.

This section introduces how to prepare and upload the model you built.

.. note::
    Before proceeding, you should first have the model built manually.
    At this moment, the iOS/Android/web app released by MLC LLM only support **specific model architectures with specific quantization modes**. Particularly,

    - the :ref:`released iOS/iPadOS app <iPhone-download-app>` supports models structured by LLaMA-7B and quantized by ``q3f16_0``, and models structured by GPT-NeoX-3B and quantized by ``q4f16_0``.
    - the :ref:`released Android app <Android-download-app>` supports models structured by LLaMA-7B and quantized by ``q4f16_0``.
    - the `Web LLM demo page <https://mlc.ai/web-llm/>`_ supports models structured by LLaMA-7B and quantized by ``q4f32_0``, and models structured by GPT-NeoX-3B and quantized by both ``q4f16_0`` and ``q4f32_0``.

    If you have not built the model with supported quantization mode(s), please refer to :ref:`the model build tutorial <How to Compile Models>`, build the model with supported quantization modes, and then come back to this page.

Assume you have built the model with :ref:`local id <knowing-local-id>` ``MODELNAME-QUANTMODE``, the directory you need to upload finally will be ``dist/MODELNAME-QUANTMODE/params``.
Before uploading, we need to update ``dist/MODELNAME-QUANTMODE/params/mlc-chat-config.json``.

Opening that file, the ``model_lib`` field specifies the model library name we use when deploying this model. Please update this field according to the panel selection below.

.. tabs::

    .. tab:: iPhone/iPad

        .. tabs::

            .. tab:: Model arch: LLaMA-7B

                The model is expected to be quantized by ``q3f16_0``:

                .. code::

                    {
                        "model_lib": "vicuna-v1-7b-q3f16_0",
                        ...
                    }

            .. tab:: GPT-NeoX-3B

                The model is expected to be quantized by ``q4f16_0``:

                .. code::

                    {
                        "model_lib": "RedPajama-INCITE-Chat-3B-v1-q4f16_0",
                        ...
                    }

            .. tab:: RWKV

                The model is expected to be quantized by ``q8f16_0``:

                .. code::

                    {
                        "model_lib": "rwkv-raven-1b5-q8f16_0",
                        ...
                    }

    .. tab:: Android

        .. tabs::

            .. tab:: Model arch: LLaMA-7B

                The model is expected to be quantized by ``q4f16_0``:

                .. code::

                    {
                        "model_lib": "vicuna-v1-7b-q4f16_0",
                        ...
                    }

    .. tab:: Web

        .. tabs::

            .. tab:: Model arch: LLaMA-7B

                The model is expected to be quantized by ``q4f32_0``:

                .. code::

                    {
                        "model_lib": "vicuna-v1-7b-q4f32_0",
                        ...
                    }

            .. tab:: GPT-NeoX-3B

                If the model is quantized by ``q4f16_0``:

                .. code::

                    {
                        "model_lib": "RedPajama-INCITE-Chat-3B-v1-q4f16_0",
                        ...
                    }

                Or if the model is quantized by ``q4f32_0``:

                .. code::

                    {
                        "model_lib": "RedPajama-INCITE-Chat-3B-v1-q4f32_0",
                        ...
                    }

.. note::
    The necessity of updating ``model_lib`` is due to the app build restriction. For app development, developers are required to pack all libraries into the app at the time of app build. So the iOS/Android app released by MLC LLM and the Web LLM demo page now only contain the libraries for model architecture LLaMA-7B and GPT-NeoX-3B.

For other fields in ``mlc-chat-config.json``, you can choose to update those configurable parameters (e.g., ``temperature``, ``top_p``, etc.) if you want, to control the behavior of the model text generation. You can refer to documentation for these configurable parameters in `Hugging Face <https://huggingface.co/docs/api-inference/detailed_parameters>`_.

After updating ``mlc-chat-config.json``, you need to upload the directory ``dist/MODELNAME-QUANTMODE/params`` (including all its contents) to an Internet location that is publicly accessible. For example, we have uploaded a few prebuilt model weight directory (`example 1 <https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q3f16_0/tree/main>`_, `example 2 <https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0/tree/main>`_) to Hugging Face.

This concludes the model upload, and you can proceed to the sections below to deploy or run models on your devices.

.. _deploy-on-laptop-desktop:

Deploy Models on Your Laptop/Desktop
------------------------------------

This section goes through the process of deploying prebuilt model or the model you built on your laptop or desktop.
MLC LLM provides a Command-Line Interface (CLI) application to deploy and help interact with the model.

Please follow the instructions in :ref:`install-mlc-chat-cli` section to install the CLI application, then you can :ref:`deploy and interact <CLI-run-model>` with the model on your machine through CLI.

.. _CLI-run-model:

Run the Models Through CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the model, we need to know the :ref:`local id <knowing-local-id>` of the model we want to deploy.
After confirming the local id, we can run the model in CLI by

.. code:: shell

    mlc_chat_cli --local-id LOCAL_ID
    # example:
    mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_0
    mlc_chat_cli --local-id vicuna-v1-7b-q3f16_0

.. image:: https://mlc.ai/blog/img/redpajama/cli.gif

.. _deploy-on-ios:

Deploy Models on Your iPhone/iPad
---------------------------------

This section introduces how to deploy model you built or prebuilt by us on your iPhone/iPad devices.
The iOS/iPadOS application supports chatting with prebuilt Vicuna or RedPajama models, and also supports using the model you manually built.

MLC LLM has released an iOS/iPadOS application which you can directly :ref:`download and use <iPhone-download-app>`.
You can also :ref:`build the application <iPhone-build-Xcode-app>` on your own.

Please check the section ":ref:`Run the model you built <iPhone-deploy-custom-model>`" to run the model you built on iPhone/iPad.

.. note::
    The app needs about 2GB of memory to run RedPajama-v1-3B quantized by ``q4f16_0``, and needs about 4GB of memory to run Vicuna-v1-7B quantized by ``q3f16_0``.
    Due to memory limitation, iPhone models with 4GB RAM may not be able to launch Vicuna-v1-7B successfully.

.. _iPhone-download-app:

Option 1. Download Released App
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `TestFlight page <https://testflight.apple.com/join/57zd7oxa>`_ to install the application for iPhone/iPad.
The link is valid for the first 9000 users.

.. _iPhone-build-Xcode-app:

Option 2. Build Application from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start building the iOS/iPadOS application, you need first build the model manually.
If you have not built the model, please refer to :ref:`the model build tutorial <How to Compile Models>`, build the model, and then come back to this page.
The following part of this subsection assumes you have built the model successfully and will not cover the model build part.

.. _update-ios-app-config:

Update iOS App Config
^^^^^^^^^^^^^^^^^^^^^

First, open ``ios/MLCChat/app-config.json`` and update the file according to the model(s) you build.

In this file, the ``model_libs`` list contains the local ids of the models we want to deploy on iPhone/iPad.

.. note::
    Due to the requirement of iOS app development, we need to pack all the model libraries into the application when we build the app.
    The ``model_libs`` field specifies the model libraries we pack into the application.

``model_list`` contains the weight repo URLs and the local ids of the models whose weights are **not** packed into the application when building the app. In the app, the models in ``model_list`` are downloaded *on users' demands*. Please check section ":ref:`Upload the model you built to Internet <upload-model>`" for how to upload the model you built to Internet and get the model URL.

``add_model_samples`` is providing a sample demo for downloading customized model weight with model URL inside the application. It can be left unchanged in the JSON.

For example, the existing ``app-config.json`` (`here <https://github.com/mlc-ai/mlc-llm/blob/main/ios/MLCChat/app-config.json>`__) means

- we pack the libraries of ``vicuna-v1-7b-q3f16_0`` and ``RedPajama-INCITE-Chat-3B-v1-q4f16_0`` into the app, so that the app supports ``q3f16_0`` quantization of Vicuna-v1-7B model and ``q4f16_0`` quantization of RedPajama-v1-3B model.
- The weight of ``vicuna-v1-7b-q3f16_0`` model is not packed into the model by default. It can be downloaded from the Hugging Face repo URL in the app on users' demands.

.. note::
    1. By packing the library of ``vicuna-v1-7b-q3f16_0`` and ``RedPajama-INCITE-Chat-3B-v1-q4f16_0`` into the app, it means we support the any LLaMA-7B structured model quantized by ``q3f16_0`` and any GPT-NeoX-3B model quantized by ``q4f16_0``.
    2. For models we want to deploy to iPhone/iPad, if we do not put it in ``model_list``, we will need to pack the model weights to the application when building the app. This is done in the following steps. In the example app config, model ``RedPajama-INCITE-Chat-3B-v1-q4f16_0`` is not put in ``model_list``. It means we will pack the weight of ``RedPajama-INCITE-Chat-3B-v1-q4f16_0`` directly into the app later.

.. _run-library-weight-preparation-scripts:

Run Library/Weight Preparation Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After updating ``app-config.json``, run ``prepare_libs.sh`` under ``ios/`` to generate required libraries.

.. code:: shell

    cd ios
    ./prepare_libs.sh

Now we specify the id of the models whose weights **we pack into the app when building**.
Open ``ios/prepare_params.sh`` and update ``builtin_list`` with such models' local ids.
For example, the existing ``prepare_params.sh`` (`here <https://github.com/mlc-ai/mlc-llm/blob/main/ios/prepare_params.sh#L8-L11>`__) means we only pack the weight of ``RedPajama-INCITE-Chat-3B-v1-q4f16_0`` into the application when building.

Then run ``prepare_params.sh`` to copy the model weights to the pre-defined target location.

.. code:: shell

    ./prepare_params.sh

.. _open-xcode-to-build-app:

Open Xcode to Build the App
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final steps go with `Xcode <https://developer.apple.com/xcode/>`_. Download Xcode and open ``ios/MLCChat.xcodeproj`` using Xcode.

.. note::
    You will need an `Apple Developer Account <https://developer.apple.com/programs/>`_ to use Xcode, and you may be prompted to use your own developer team credential and product bundle identifier.

Once you have made the necessary changes, build the iOS app using Xcode. Make sure to select a target device (requires connecting your device to your Mac via wire) or simulator for the build.

After a successful build, you can run the iOS app on your device or simulator to use the LLM model for text generation and processing.


.. _iPhone-deploy-custom-model:

Run the Model You Built
~~~~~~~~~~~~~~~~~~~~~~~

In section ":ref:`Upload the model you built to Internet <upload-model>`", we introduced how to upload the model you built to Internet for applications to download.

If you want to run the model you built, simply follow the steps below.

.. tabs::

    .. tab:: Step 1

        Open "MLCChat" app, click "Add model variant".

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-1.png
            :align: center
            :width: 30%

    .. tab:: Step 2

        Paste the repository URL of the model built on your own, and click Add.

        You can refer to the link in the image as an example.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-2.png
            :align: center
            :width: 30%

    .. tab:: Step 3

        After adding the model, you can download your model from the URL by clicking the download button.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-3.png
            :align: center
            :width: 30%

    .. tab:: Step 4

        When the download is finished, click into the model and enjoy.

        .. image:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/iPhone-custom-4.png
            :align: center
            :width: 30%


.. _deploy-model-on-android:

Deploy Model on Your Android Phone
----------------------------------

.. _Android-download-app:

Download Released App
~~~~~~~~~~~~~~~~~~~~~

TBA.

.. _deploy-model-web-browser:

Deploy Models on Your Web Browser
---------------------------------

TBA.
