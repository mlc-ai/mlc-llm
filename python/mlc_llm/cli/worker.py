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

# register the calibration functions
from ..interface import calibrate  # pylint: disable=unused-import


def main():
    """Main worker function"""

    if len(sys.argv) == 5 or len(sys.argv) == 6:
        *args, read_fd, write_fd = map(int, sys.argv[1:])
    else:
        print(
            f"Expected exactly either 4 or 5 arguments, "
            f"but received {len(sys.argv)-1} arguments.: {sys.argv}"
        )
        # The <num_groups> argument was added in
        # https://github.com/apache/tvm/pull/17180.  This script
        # currently checks the number of arguments present, to
        # determine whether `num_groups` was provided.  This allows
        # the worker.py script provided by MLC-LLM to be compatible
        # with either pre-17180 or post-17180 arguments.
        #
        # After the TVM version used by MLC-LLM includes #17180, the
        # usage can be updated to always require `len(sys.argv)==6`.
        print("Usage (without num groups): <worker_id> <num_workers> <read_fd> <write_fd>")
        print(
            "Usage (with num groups): <worker_id> <num_workers> <num_groups> <read_fd> <write_fd>"
        )
        return

    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        reader = msvcrt.open_osfhandle(read_fd, os.O_BINARY)
        writer = msvcrt.open_osfhandle(write_fd, os.O_BINARY)
    else:
        reader = read_fd
        writer = write_fd

    worker_func = get_global_func("runtime.disco.WorkerProcess")
    worker_func(*args, reader, writer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
