#!/bin/bash
set -euxo pipefail

ln -s $(python -c "import site; print(site.getsitepackages()[0])")/mlc_chat/*.so python/mlc_chat/
python -c "import mlc_chat; print(mlc_chat)"
ls python/mlc_chat/
cd docs && make html && cd ..
