#!/bin/bash
set -euxo pipefail

ln -s $(python -c "import site; print(site.getsitepackages()[0])")/mlc_chat/*.so python/mlc_chat/
cd docs && make html && cd ..

cd site && jekyll b && cd ..

rm -rf site/_site/docs
cp -r docs/_build/html site/_site/docs
