// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

run_cpu = "bash ci/bash.sh mlcaidev/ci-cpu:a7b7a72"
run_cuda = "bash ci/bash.sh mlcaidev/ci-cu121:a7b7a72"
pkg_cuda = "bash ci/bash.sh mlcaidev/package-cu121:1da76a2"

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

def init_git(submodule = false) {
  cleanWs()
  checkout scm
  if (submodule) {
    retry(5) {
      timeout(time: 10, unit: 'MINUTES') {
        sh(script: 'git submodule update --init --recursive -f', label: 'Update git submodules')
      }
    }
  }
}

stage('Lint') {
  parallel(
    'isort': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-lint-isort')) {
          init_git()
          sh(script: "ls", label: 'debug')
          sh(script: "${run_cpu} conda env export --name ci-lint", label: 'Checkout version')
          sh(script: "${run_cpu} conda run -n ci-lint ci/task/isort.sh", label: 'Lint')
        }
      }
    },
    'black': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-lint-black')) {
          init_git()
          sh(script: "ls", label: 'debug')
          sh(script: "${run_cpu} conda env export --name ci-lint", label: 'Checkout version')
          sh(script: "${run_cpu} conda run -n ci-lint ci/task/black.sh", label: 'Lint')
        }
      }
    },
    'mypy': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-lint-mypy')) {
          init_git()
          sh(script: "ls", label: 'debug')
          sh(script: "${run_cpu} conda env export --name ci-lint", label: 'Checkout version')
          sh(script: "${run_cpu} conda run -n ci-lint ci/task/mypy.sh", label: 'Lint')
        }
      }
    },
    'pylint': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-lint-pylint')) {
          init_git()
          sh(script: "ls", label: 'debug')
          sh(script: "${run_cpu} conda env export --name ci-lint", label: 'Checkout version')
          sh(script: "${run_cpu} conda run -n ci-lint ci/task/pylint.sh", label: 'Lint')
        }
      }
    },
    'clang-format': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-lint-clang-format')) {
          init_git()
          sh(script: "ls", label: 'debug')
          sh(script: "${run_cpu} conda env export --name ci-lint", label: 'Checkout version')
          sh(script: "${run_cpu} conda run -n ci-lint ci/task/clang-format.sh", label: 'Lint')
        }
      }
    },
  )
}

stage('Build') {
  parallel(
    'CUDA': {
      node('CPU-SMALL') {
        ws(per_exec_ws('mlc-llm-build')) {
          init_git(true)
          sh(script: "ls", label: 'debug')
          sh(script: 'git clone --recursive https://github.com/mlc-ai/package', label: 'Clone mlc-ai/package')
          sh(script: "${pkg_cuda} ./ci/task/build_lib.sh", label: 'Build MLC LLM runtime')
          sh(script: "${pkg_cuda} ./ci/task/build_wheel.sh", label: 'Build MLC LLM wheel')
          sh(script: "${pkg_cuda} ./ci/task/build_clean.sh", label: 'Cleanups after build')
        }
      }
    }
  )
}
