import requests

# To launch the server, run
# $ uvicorn server:app --reload

"""
curl -X POST http://127.0.0.1:8000/models/init -H "Content-Type: application/json" \
-d '{"model": "vicuna-v1-7b", "mlc_path": "/home/sudeepag/mlc-llm", "artifact_path": "/home/sudeepag/mlc-llm/dist", "device_name": "vulkan"}'
"""

"""
curl http://127.0.0.1:8000/models/init
"""

"""
curl -X POST http://127.0.0.1:8000/chat/completions -H "Content-Type: application/json" \
-d '{"model": "vicuna-v1-7b", "prompt": "Hello!"}'
"""