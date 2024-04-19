#!/bin/bash
# NOTE: this script is triggered by github action automatically
# when megred into main

set -euxo pipefail

scripts/build_mlc_for_docs.sh
scripts/build_site.sh

git fetch
git checkout -B gh-pages origin/gh-pages
rm -rf docs .gitignore
mkdir -p docs
cp -rf site/_site/* docs
touch docs/.nojekyll

DATE=`date`
git add docs && git commit -am "Build at ${DATE}"
git push origin gh-pages
git checkout main && git submodule update
echo "Finish deployment at ${DATE}"
