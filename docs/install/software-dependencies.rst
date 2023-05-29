.. _Software Dependencies:

Software Dependencies
=====================

.. contents:: Table of Contents
    :depth: 2
    :local:

While we have included most of the dependencies in our pre-built wheels/scripts, there are still some platform-dependent packages that you will need to install on your own. In most cases, you won't need all the packages listed on this page. If you're unsure about which packages are required for your specific use case, please check the :ref:`navigation panel <navigation>` first.

.. _software-dependencies-cuda:

CUDA
----

CUDA is required to compile and run models with ``cuda`` backend.

.. _cuda-installation:

Installation
^^^^^^^^^^^^

If you have a NVIDIA GPU and you want to use models compiled with CUDA
backend, you should install CUDA, which can be downloaded from
`here <https://developer.nvidia.com/cuda-downloads>`__.

.. _cuda-validate-installation:

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed CUDA runtime and NVIDIA driver, you can simply run ``nvidia-smi`` in command line and see
if you can get the GPU information.

.. _software-dependencies-vulkan-driver:

Vulkan Driver
-------------

.. _vulkan-driver-installation:

Installation
^^^^^^^^^^^^

To run pre-trained models (e.g. pulled from MLC-AI's Hugging Face repository) compiled with ``vulkan`` backend, you are expected to install Vulkan driver on your machine.

Please check `this
page <https://www.vulkan.org/tools#vulkan-gpu-resources>`__ and find the
Vulkan driver according to your GPU vendor.

.. _valkan-driver-validate-installation:

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify whether Vulkan installation is successful or not, you are encouraged to install ``vulkaninfo``, below are the instructions to install ``vulkaninfo`` on different platforms:

.. tabs ::
   
   .. code-tab :: bash Ubuntu/Debian

      sudo apt-get update
      sudo apt-get install vulkan-tools

   .. code-tab :: bash Fedora

      sudo dnf install vulkan-tools
   
   .. code-tab :: bash Arch Linux

      sudo pacman -S vulkan-tools
      # Arch Linux has maintained an awesome wiki page for Vulkan which you can refer to for troubleshooting: https://wiki.archlinux.org/title/Vulkan
   
   .. code-tab :: bash Other Distributions

      # Please install vulkan-sdk for your platform
      # https://vulkan.lunarg.com/sdk/home


After installation, you can run ``vulkaninfo`` in command line and see if you can get the GPU information.

.. note::
   If you found trouble running models with vulkan backend, please attach the ``vulkaninfo`` output in your issue report so that we can identify whether it's vulkan driver installation issue.

.. _software-dependencies-vulkan-sdk:

Vulkan-SDK
----------

Vulkan-SDK is required for compiling models to Vulkan backend. However, our `pre-built mlc-ai-nightly wheels <https://mlc.ai/package>`__ already packaged Vulkan-SDK, so there is no need to install it separately.

Installing Vulkan-SDK is only necessary when user build TVM-Unity from source by following our :ref:`tvm-unity-build-from-source` manual.

Please check the Vulkan-SDK installation guide according to your platform:

.. tabs ::

   .. tab :: Windows

      `Getting Started with the Windows Tarball Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/windows/getting_started.html>`__
   
   .. tab :: Linux

      For Ubuntu user, please check 
      `Getting Started with the Ubuntu Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started_ubuntu.html>`__

      For other Linux distributions, please check
      `Getting Started with the Linux Tarball Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html>`__
   
   .. tab :: Mac

      `Getting Started with the macOS Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html>`__

Please refer to installation and setup page for next steps to build TVM-Unity from source.

.. _software-dependencies-opencl-sdk:

OpenCL-SDK
----------

OpenCL-SDK is only required when you want to build your own models for OpenCL backend. Please refer to `OpenCL's Github Repository <https://github.com/KhronosGroup/OpenCL-SDK>`__ for installation guide of OpenCL-SDK.
