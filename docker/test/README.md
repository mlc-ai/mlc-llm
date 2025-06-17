## Tests for mlc_llm `serve` 

The simple test programs for REST API serving (including function calling / tools pattern) when using mlc_llm `serve` with any of the 100s of supported models.

|Test name|Description|
|------------|---------------|
|`sample_client_for-testing.py`|Calls completion REST API once without streaming, and then again with streaming, and displays the output.  Make sure you modify the `payload` LLM name field to match the actually LLM you are testing.|
|`functionall.py`|Actual function calling example utilizing OpenAI compatible API  _tools_ field.  Make sure you modify the `payload` LLM name field to match the actually LLM you are testing.  This example will only work with models fine-tuned for function calling, including many Mixtral/Mistral derivatives.|
