GPU Drivers and SDKs
====================

.. contents:: Table of Contents
    :depth: 2

MLC LLM is a universal deployment solution that allows efficient CPU/GPU code generation without AutoTVM-based performance tuning. This section focuses on generic GPU environment setup and troubleshooting.

CUDA
----

CUDA is required to compile and run models with CUDA backend.

Installation
^^^^^^^^^^^^

If you have a NVIDIA GPU and you want to use models compiled with CUDA
backend, you should install CUDA, which can be downloaded from
`here <https://developer.nvidia.com/cuda-downloads>`__.

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed CUDA runtime and NVIDIA driver, run ``nvidia-smi`` in command line and see if you can get the GPU information.

Vulkan Driver
-------------

Installation
^^^^^^^^^^^^

To run pre-trained models (e.g. pulled from MLC-AI's Hugging Face repository) compiled with Vulkan backend, you are expected to install Vulkan driver on your machine.

Please check `this
page <https://www.vulkan.org/tools#vulkan-gpu-resources>`__ and find the
Vulkan driver according to your GPU vendor.

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify whether Vulkan installation is successful or not, you are encouraged to install ``vulkaninfo``, below are the instructions to install ``vulkaninfo`` on different platforms:

.. tabs ::
   
   .. code-tab :: bash Ubuntu/Debian

      sudo apt-get update
      sudo apt-get install vulkan-tools

   .. code-tab :: bash Windows

      # It comes with your GPU driver

   .. code-tab :: bash Fedora

      sudo dnf install vulkan-tools
   
   .. code-tab :: bash Arch Linux

      sudo pacman -S vulkan-tools
      # Arch Linux has maintained an awesome wiki page for Vulkan which you can refer to for troubleshooting: https://wiki.archlinux.org/title/Vulkan
   
   .. code-tab :: bash Other Distributions

      # Please install Vulkan SDK for your platform
      # https://vulkan.lunarg.com/sdk/home


After installation, you can run ``vulkaninfo`` in command line and see if you can get the GPU information.

.. note::
   WSL support for Windows is work-in-progress at the moment. Please do not use WSL on Windows to run Vulkan.

Vulkan SDK
----------

Vulkan SDK is required for compiling models to Vulkan backend. To build TVM Unity compiler from source, you will need to install Vulkan SDK as a dependency, but our `pre-built wheels <https://mlc.ai/package>`__ already ships with Vulkan SDK.

Check Vulkan SDK installation guide according to your platform:

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

OpenCL SDK
----------

OpenCL SDK is only required when you want to build your own models for OpenCL backend. Please refer to `OpenCL's Github Repository <https://github.com/KhronosGroup/OpenCL-SDK>`__ for installation guide of OpenCL-SDK.
