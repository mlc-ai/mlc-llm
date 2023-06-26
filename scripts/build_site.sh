#!/bin/bash
set -euxo pipefail

if [ ! -d "build"] then
    mkdir build
    ln -s $(python -c "import mlc_chat; print(mlc_chat.__path__[0])")/*.so build/
fi

cd docs && make html && cd ..
