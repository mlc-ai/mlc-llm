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

image = 'mlcaidev/ci-cpu:caab922'
docker_run = "bash ci/bash.sh ${image}"

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

def init_git(submodule = false) {
  checkout scm
  if (submodule) {
    retry(5) {
      timeout(time: 2, unit: 'MINUTES') {
        sh(script: 'git submodule update --init --recursive -f', label: 'Update git submodules')
      }
    }
  }
}

def lint_common(cmd) {
  // No need to checkout submodule for lint
  node('CPU-SMALL') {
    ws(per_exec_ws('mlc-llm-lint')) {
      init_git()
      sh(script: "ls", label: 'debug')
      sh(script: "${docker_run} conda env export --name ci-lint", label: 'Checkout version')
      sh(script: "${docker_run} ${cmd}", label: 'Lint')
    }
  }
}

stage('Lint') {
  parallel(
    'isort': {
      lint_common('bash ci/task/isort.sh')
    },
    'black': {
      lint_common('bash ci/task/black.sh')
    },
    'mypy': {
      lint_common('bash ci/task/mypy.sh')
    },
    'pylint': {
      lint_common('bash ci/task/pylint.sh')
    },
    'clang-format': {
      lint_common('bash ci/task/clang-format.sh')
    },
  )
}