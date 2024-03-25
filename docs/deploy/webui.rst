WebUI Frontent
==============

.. contents:: Table of Contents
   :local:
   :depth: 2

We provide support for running MLC-Chat models on https://github.com/oobabooga/text-generation-webui.

Setup MLC fork of text-generation-webui
---------------------------------------

Clone the MLC fork of text-generation-webui repo from https://github.com/mlc-ai/text-generation-webui .
Create a new conda environment and install the required dependencies

.. code:: bash
    
   git clone https://github.com/mlc-ai/text-generation-webui
   cd text-generation-webui
   conda create -n textgen python=3.11
   conda activate textgen
   pip install -r <requirements file according to your architecture>
   pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly mlc-chat-nightly


Setup models
------------

Copy the desired model weights and model library into the ``text-generation-webui/models/<model_name>`` directory. For more details, refer to the :doc:`the Model Prebuilts </prebuilt_models>` page or the :doc:`the Model Compilation </compilation/compile_models>` page.


Run WebUI
---------

Run the text-generation-webui server using the following command

.. code:: bash
   
   python server.py


Navigate to the ``Model`` tab. Under the ``Model`` dropdown menu, select your MLC-Chat model name (which you had copied into the ``text-generation-webui/models/<model_name>`` directory). Under the ``Model loader`` dropdown menu, select ``MLCChat``. Click on ``Load``. The model should now be loaded and ready to use.

Navigate to the ``Parameters`` tab to set custom parameters such as `temperature`, `top_p`, `max_new_tokens`, etc.

Finally, navigate to the ``Chat`` tab to start chatting with the model.

