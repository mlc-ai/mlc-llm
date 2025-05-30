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
from tvm.ffi import get_global_func

from .. import base  # pylint: disable=unused-import, no-name-in-module

# register the calibration functions
from ..interface import calibrate  # pylint: disable=unused-import


def main():
    """Main worker function"""
    if len(sys.argv) != 6:
        print("Usage: <worker_id> <num_workers> <num_groups> <read_fd> <write_fd>")
        return

    worker_id = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_groups = int(sys.argv[3])
    read_fd = int(sys.argv[4])
    write_fd = int(sys.argv[5])
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        reader = msvcrt.open_osfhandle(read_fd, os.O_BINARY)
        writer = msvcrt.open_osfhandle(write_fd, os.O_BINARY)
    else:
        reader = read_fd
        writer = write_fd

    worker_func = get_global_func("runtime.disco.WorkerProcess")
    worker_func(worker_id, num_workers, num_groups, reader, writer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
