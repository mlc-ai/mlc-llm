# This script does two things
# It calls prepare_libs.sh in $MLC_LLM_HOME/ios/ to setup the iOS package and build binaries
# It then calls mlc_llm package to setup the weight and library bundle
# Feel free to copy this file and mlc-package-config.json to your project

MLC_LLM_HOME="${MLC_LLM_HOME:-../..}"
cd ${MLC_LLM_HOME}/ios && ./prepare_libs.sh $@ && cd -
rm -rf dist/lib && mkdir -p dist/lib
cp ${MLC_LLM_HOME}/ios/build/lib/* dist/lib/
python -m mlc_llm package mlc-package-config.json --device iphone -o dist
