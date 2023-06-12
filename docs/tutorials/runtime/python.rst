🚧 Run Models with Python
=========================

Currently we provide REST API and Gradio API to run models with Python.

REST API
--------

To run models with the REST API, there is a dependency to build from source in order to use the
`REST
API <https://www.ibm.com/topics/rest-apis#:~:text=the%20next%20step-,What%20is%20a%20REST%20API%3F,representational%20state%20transfer%20architectural%20style.>`__.

1. Follow the documentation :ref:`mlcchat_build_from_source` to build
   the CLI from source.
2. Launch the server at http://127.0.0.1:8000/.

.. code:: bash

    cd mlc-llm/python
    python -m mlc_chat.rest

3. Go to http://127.0.0.1:8000/docs to look at the list of supported
   endpoints, or run the sample client script to see how to send
   queries.

.. code:: bash
    
    python -m mlc_chat.sample_client

Gradio API
----------

To launch the Gradio API, in the current folder, run the following
example command. The ``--share`` argument is for optionally creating a
publicly shareable link for the interface.

.. code:: bash

   PYTHONPATH=python python3 -m mlc_chat.gradio --artifact-path /path/to/your/models --device-name cuda --device-id 0 --share
