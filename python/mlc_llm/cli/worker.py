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
# pylint: disable=invalid-name
"""Internal DiscoWorker for Disco ProcessSession."""
import os
import sys

from tvm import runtime as _  # pylint: disable=unused-import
from tvm._ffi import get_global_func

from .. import base  # pylint: disable=unused-import, no-name-in-module


def main():
    """Main worker function"""
    if len(sys.argv) != 5:
        print("Usage: <worker_id> <num_workers> <read_fd> <write_fd>")
        return

    worker_id = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        reader = msvcrt.open_osfhandle(int(sys.argv[3]), os.O_BINARY)
        writer = msvcrt.open_osfhandle(int(sys.argv[4]), os.O_BINARY)
    else:
        reader = int(sys.argv[3])
        writer = int(sys.argv[4])

    worker_func = get_global_func("runtime.disco.WorkerProcess")
    worker_func(worker_id, num_workers, reader, writer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
