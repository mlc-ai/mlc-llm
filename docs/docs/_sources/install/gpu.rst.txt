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

ROCm
----

ROCm is required to compile and run models with ROCm backend.

Installation
^^^^^^^^^^^^

Right now MLC LLM only supports ROCm 6.1/6.2.
If you have AMD GPU and you want to use models compiled with ROCm
backend, you should install ROCm from `here <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/install/quick-start.html>`__.

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed ROCm, run ``rocm-smi`` in command line.
If you see the list of AMD devices printed out in a table, it means the ROCm is correctly installed.

.. _vulkan_driver:

Vulkan Driver
-------------

Installation
^^^^^^^^^^^^

To run pre-trained models (e.g. pulled from MLC-AI's Hugging Face repository) compiled with Vulkan backend, you are expected to install Vulkan driver on your machine.

Please check `this
page <https://www.vulkan.org/tools#vulkan-gpu-resources>`__ and find the
Vulkan driver according to your GPU vendor.

AMD Radeon and Radeon PRO
#########################

For AMD Radeon and Radeon PRO users, please download AMD's drivers from official website (`Linux <https://www.amd.com/en/support/linux-drivers>`__ / `Windows <https://www.amd.com/en/support>`__).
For Linux users, after you installed the ``amdgpu-install`` package, you can follow the instructions in its `documentation <https://amdgpu-install.readthedocs.io/en/latest/install-script.html>`__ to install
the driver. We recommend you installing ROCr OpenCL and PRO Vulkan (proprietary) for best performance, which can be done by running the following command:

.. code:: bash

   amdgpu-install --usecase=graphics,opencl --opencl=rocr --vulkan=pro --no-32

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

Vulkan SDK is required for compiling models to Vulkan backend. To build TVM Unity compiler from source, you will need to install Vulkan SDK as a dependency, but our :doc:`pre-built wheels <../install/mlc_llm>` already ships with Vulkan SDK.

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

Orange Pi 5 (RK3588 based SBC)
------------------------------

OpenCL SDK and Mali GPU driver is required to compile and run models for OpenCL backend.

Installation
^^^^^^^^^^^^

* Download and install the Ubuntu 22.04 for your board from `here <https://github.com/Joshua-Riek/ubuntu-rockchip/releases/tag/v1.22>`__

* Download and install ``libmali-g610.so``

.. code-block:: bash

   cd /usr/lib && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/lib/aarch64-linux-gnu/libmali-valhall-g610-g6p0-x11-wayland-gbm.so

* Check if file ``mali_csffw.bin`` exist under path ``/lib/firmware``, if not download it with command:

.. code-block:: bash

   cd /lib/firmware && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/firmware/g610/mali_csffw.bin

* Download OpenCL ICD loader and manually add libmali to ICD

.. code-block:: bash

   sudo apt update
   sudo apt install mesa-opencl-icd
   sudo mkdir -p /etc/OpenCL/vendors
   echo "/usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so" | sudo tee /etc/OpenCL/vendors/mali.icd

* Download and install ``libOpenCL``

.. code-block:: bash

   sudo apt install ocl-icd-opencl-dev

* Download and install dependencies for Mali OpenCL

.. code-block:: bash

   sudo apt install libxcb-dri2-0 libxcb-dri3-0 libwayland-client0 libwayland-server0 libx11-xcb1

* Download and install clinfo to check if OpenCL successfully installed

.. code-block:: bash

   sudo apt install clinfo

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed OpenCL runtime and Mali GPU driver, run ``clinfo`` in command line and see if you can get the GPU information.
You are expect to see the following information:

.. code-block:: bash

   $ clinfo
   arm_release_ver: g13p0-01eac0, rk_so_ver: 3
   Number of platforms                               2
      Platform Name                                   ARM Platform
      Platform Vendor                                 ARM
      Platform Version                                OpenCL 2.1 v1.g6p0-01eac0.2819f9d4dbe0b5a2f89c835d8484f9cd
      Platform Profile                                FULL_PROFILE
      ...
