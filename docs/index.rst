.. MLC-LLM documentation master file, created by
   sphinx-quickstart on Mon May 15 14:00:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MLC-LLM's documentation!
===================================

MLC LLM is a **universal solution** that allows **any language models** to be **deployed natively** on a diverse set of hardware backends and native applications, plus a **productive framework** for everyone to further optimize model performance for their own use cases.

Our mission is to **enable everyone to develop, optimize and deploy AI models natively on everyone's devices**.

Everything runs locally with no server support and accelerated with local GPUs on your phone and laptops.

`Join our Discord Server! <https://discord.gg/9Xpy2HGBuD>`_

.. _navigation:

Navigation
----------

Before you start, please select your use case so that we can narrow down the search scope.

.. tabs ::

   .. tab :: I want to run models on my device.
 

      Please select your platform:

      .. tabs ::

         .. tab :: Android

            Please check our instructions on `MLC-LLM Android app <https://mlc.ai/mlc-llm/#android>`__.

         .. tab :: iOS

            Please check our instructions on `MLC-LLM IOS app <https://mlc.ai/mlc-llm/#iphone>`__
         
         .. tab :: WebGPU

            Please check `Web-LLM project <https://mlc.ai/web-llm/>`__.
         
         .. tab :: PC

            MLC-LLM already provided a set of prebuilt models which you can deploy directly, check the :doc:`model-zoo` for the list of supported models.

            .. tabs ::
               
               .. tab :: Use prebuilt models.

                  Please check :doc:`tutorials/deploy-models` for instructions on preparing models and deploying models with MLC-LLM CLI.

                  * If you are a Mac OS user, and you will run a model compiled with ``metal`` backend, congratulations! You don't need to install any external drivers/packages but ``metal`` is natively supported by Mac OS.
                  * If you will run a model compiled with ``CUDA`` backend, please install CUDA accordingly to our :ref:`CUDA installation guide <software-dependencies-cuda>`.
                  * If you will run a model compiled with ``Vulkan`` backend, please install Vulkan Driver accordingly to our :ref:`Vulkan Driver Installation Guide <software-dependencies-vulkan-driver>`.

               .. tab :: Build your own models.

                  There are two different cases where you need to build your own models:

                  * Use your own moden weights, or use different quantization data type/running data type.
                  * Please check :doc:`tutorials/compile-models` for details on how to prepare build models for existing architectures.
                  * Use a brand new model architecture which is not supported by MLC-LLM yet.
                  * Please check :doc:`tutorials/bring-your-own-models` for details on how to add new model architectures to the MLC-LLM family.
                
                  In either cases, you are ecouraged to contribute to the MLC-LLM, see :ref:`contribute-new-models` on guidelines for contributing new models.

   .. tab :: I need to customize MLC-LLM.

      There are lots of interesting ways to further improve and customize MLC-LLM.

      * The performance of MLC-LLM can be improved in a lot of ways, including (but not limited to) fusing multi-head attention with FlashAttention algorithm, or using more advanced quantization algorithms.
      * We can also add new backends/language binding with the existing infrastructure.
      
      Before you start, please check our :doc:`tutorials/customize` to see how can you customize MLC-LLM for your own purpose.

      You are encouraged to contribute to the MLC-LLM if your found your customization intersting.

      .. tabs ::
     
         .. tab :: I need to customize TVM-Unity

            In this case, user need to change TVM-Unity codebase. Please check :ref:`tvm-unity-build-from-source` on how to install TVM-Unity from source.

            * If user want to compile models with ``CUDA`` backend, please install CUDA according to our :ref:`CUDA installation guide <software-dependencies-cuda>`.
            * If user want to compile models with ``Vulkan`` backend, please install Vulkan-SDK according to our :ref:`Vulkan SDK Installation Guide <software-dependencies-vulkan-sdk>`.
            * If user want to compile models with ``OpenCL`` backend, please install OpenCL-SDK according to our :ref:`OpenCL SDK Installation Guide <software-dependencies-opencl-sdk>`.

         .. tab :: Use original TVM-Unity

            In this case, please install prebuilt TVM-Unity package according to our :ref:`Installation Guidelines <tvm-unity-install-prebuilt-package>`.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

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
   :caption: Model Zoo

   model-zoo.rst

.. toctree::
   :maxdepth: 1
   :caption: Misc

   misc/faq.rst

