# Build Instructions

## REST API

There is currently a dependency to build from source in order to use the REST API. 

1. Follow the instructions [here](https://github.com/mlc-ai/mlc-llm/tree/main/cpp) to build the CLI from source.
2. Launch the server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
    ```shell
    cd mlc-llm
    python python/mlc_chat/server.py
    ```
3. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to look at the list of supported endpoints, or run the sample frontend script to see how to send queries.
    ```
    python python/mlc_chat/sample_frontend.py
    ```

## Gradio App

To be completed.