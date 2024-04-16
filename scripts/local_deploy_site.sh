#!/bin/bash
# NOTE: use this script to check local site

set -euxo pipefail

scripts/build_site.sh

cd site && jekyll serve  --skip-initial-build --host localhost --baseurl / --port 8888
