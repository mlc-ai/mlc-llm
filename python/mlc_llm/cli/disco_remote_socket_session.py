# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Internal remote disco socket session."""
import sys

from tvm import runtime as _  # pylint: disable=unused-import
from tvm.ffi import get_global_func

from .. import base  # pylint: disable=unused-import, no-name-in-module

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <server_host> <server_port> <num_workers>")
        sys.exit(1)

    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    num_workers = int(sys.argv[3])
    func = get_global_func("runtime.disco.RemoteSocketSession")
    func(server_host, server_port, num_workers)
