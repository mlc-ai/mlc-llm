#!/bin/bash
set -euxo pipefail

ln -s $(python -c "import mlc_chat; print(mlc_chat.__path__[0])"/*.so python/mlc_chat/
python -c "import mlc_chat; print(mlc_chat)"
ls python/mlc_chat/
cd docs && make html && cd ..
