#!/bin/bash
set -euxo pipefail

cd site && jekyll serve --host localhost --baseurl /mlc-llm --port 8888
