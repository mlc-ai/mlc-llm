.. MLC-LLM documentation master file, created by
   sphinx-quickstart on Mon May 15 14:00:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MLC-LLM!
===================================

`Join our Discord Server! <https://discord.gg/9Xpy2HGBuD>`_

🚧 This doc is under heavy construction.

MLC LLM is the universal deployment solution that allows LLMs to run locally with native hardware acceleration on consumer devices.

.. _navigation:

.. |mlc-cli-demo| image:: https://mlc.ai/blog/img/redpajama/cli.gif
.. |web-llm-demo| image:: https://github.com/mlc-ai/web-llm/raw/main/site/img/fig/demo.gif
.. |ios-demo| image:: https://mlc.ai/mlc-llm/gif/ios-demo.gif
.. |android-demo| image:: https://mlc.ai/mlc-llm/gif/android-demo.gif

Navigate by Topics
------------------

MLC LLM offers a set of pre-compiled models (:ref:`off-the-shelf-models`), as well as Python scripts that enable developers to define and compile models using either existing architectures or create new ones with customized weights.

.. tabs ::

   .. tab :: Run Compiled Model

      An MLC-compiled model is composed of two elements: weights and a library of CPU/GPU compute kernels. For easy integration and redistribution, MLC further offers platform-specific lightweight runtimes, with a user-friendly interface to interact with the compiled models.

      .. tabs ::

         .. tab :: Command Line

            ``mlc_chat_cli`` is the CLI app provided to load the model weights and compute kernels.

            - Demo: |mlc-cli-demo|
            - Install or build the CLI app: :ref:`install-mlc-chat-cli`.
            - Run compiled models via the CLI app: :ref:`CLI-run-model`.

            .. - Use MLC-compiled models in your own C++ project: TBA

         .. tab :: Web Browser

            MLC compiles a model to WebGPU and WebAssembly, which can be executed by MLC LLM JavaScript runtime.

            - Demo: |web-llm-demo|
            - Set up WebLLM: `Web-LLM project <https://mlc.ai/web-llm/>`__.
            - Use MLC-compiled models in your own JavaScript project: `Web-LLM NPM Package <https://www.npmjs.com/package/@mlc-ai/web-llm>`__.

         .. tab :: iOS

            A model can be compiled to static system libraries and further linked to an iOS app. An example iOS app is provided with a clear structure that iOS developers could refer to ship LLMs in iOS.

            - Demo: |ios-demo|
            - Set up iOS: `iOS <https://mlc.ai/mlc-llm/#iphone>`__.

            .. - Use MLC-compiled models in your own iOS app: TBD

         .. tab :: Android

            A model can be compiled to static system libraries and further linked to an Android app. An example Android app is provided with a clear structure that Android developers could refer to ship LLMs in Android.

            - Demo: |android-demo|
            - Set up Android: `Android <https://mlc.ai/mlc-llm/#android>`__

            .. - Use MLC-compiled models in your own Android app: TBD

   .. tab :: Compile Models

      MLC LLM is a Python package that uses TVM Unity to compile LLMs for universal deployment.

      .. image:: https://mlc.ai/blog/img/redpajama/customization.svg

      - Install TVM Unity: :ref:`Installation Guidelines <tvm-unity-install-prebuilt-package>`.
      - Compile models: :doc:`tutorials/compile-models`.
      - Contribute new models: :ref:`contribute-new-models`.

      .. - MLC LLM Compilation: TBD
      .. - Configuring build environments: TBD

   .. tab :: Define Model Architectures

      - Install TVM Unity: :ref:`Installation Guidelines <tvm-unity-install-prebuilt-package>`.
      - Define new model architectures: :doc:`tutorials/bring-your-own-models`.
      - Machine Learning Compilation Basics: `Machine Learning Compilation <https://mlc.ai/>`__
      - Contribute new models: :ref:`contribute-new-models`.


.. toctree::
   :maxdepth: 1
   :caption: All tutorials

   install/index.rst
   install/software-dependencies.rst
   tutorials/deploy-models.rst
   tutorials/compile-models.rst
   tutorials/bring-your-own-models.rst
   tutorials/customize.rst

.. toctree::
   :maxdepth: 1
   :caption: Contribute to MLC-LLM

   contribute/community.rst

.. toctree::
   :maxdepth: 1
   :caption: Model Prebuilts

   model-prebuilts.rst

.. toctree::
   :maxdepth: 1
   :caption: Misc

   misc/faq.rst
