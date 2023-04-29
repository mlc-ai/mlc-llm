#!/bin/bash
set -euxo pipefail

cd site && jekyll b && cd ..

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
