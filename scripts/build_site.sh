#!/bin/bash
set -euxo pipefail

export PYTHONPATH=$PWD/python
cd docs && make html && cd ..

cd site && jekyll b && cd ..

rm -rf site/_site/docs
cp -r docs/_build/html site/_site/docs
