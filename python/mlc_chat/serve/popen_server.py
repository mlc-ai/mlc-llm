"""The MLC LLM server launched in a subprocess."""
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests


class PopenServer:  # pylint: disable=too-many-instance-attributes
    """The wrapper of MLC LLM server, which runs the server in
    a background subprocess."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: str = "auto",
        *,
        model_lib_path: str = "",
        use_threaded_engine: bool = True,
        max_batch_size: int = 80,
        max_total_sequence_length: int = 16800,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Please check out `server.py` for the server arguments."""
        self.model = model
        self.device = device
        self.model_lib_path = model_lib_path
        self.use_threaded_engine = use_threaded_engine
        self.max_batch_size = max_batch_size
        self.max_total_sequence_length = max_total_sequence_length
        self.host = host
        self.port = port
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """Launch the server in a popen subprocess.
        Wait until the server becomes ready before return.
        """
        cmd = [sys.executable]
        cmd += ["-m", "mlc_chat.serve.server"]
        cmd += ["--model", self.model]
        cmd += ["--device", self.device]
        cmd += ["--model-lib-path", self.model_lib_path]
        cmd += ["--max-batch-size", str(self.max_batch_size)]
        cmd += ["--max-total-seq-length", str(self.max_total_sequence_length)]
        if self.use_threaded_engine:
            cmd += ["--use-threaded-engine"]

        cmd += ["--host", self.host]
        cmd += ["--port", str(self.port)]
        process_path = str(Path(__file__).resolve().parents[3])
        self._proc = subprocess.Popen(cmd, cwd=process_path)  # pylint: disable=consider-using-with
        # NOTE: DO NOT USE `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
        # in subprocess.Popen here. PIPE has a fixed-size buffer with may block
        # and hang forever.

        # Try to query the server until it is ready.
        openai_v1_models_url = "http://127.0.0.1:8000/v1/models"
        query_result = None
        timeout = 60
        attempts = 0
        while query_result is None and attempts < timeout:
            try:
                query_result = requests.get(openai_v1_models_url, timeout=60)
            except:  # pylint: disable=bare-except
                attempts += 1
                time.sleep(1)

        # Check if the subprocess terminates unexpectedly or
        # the queries reach the timeout.
        process_return_code = self._proc.poll()
        if process_return_code is not None:
            raise RuntimeError(
                "The server fails to launch. "
                f'Please check if "{self.model}" is a valid model compiled by MLC LLM.'
            )
        if attempts == timeout:
            self.terminate()
            raise RuntimeError(f"The server fails to launch in {timeout} seconds.")

    def terminate(self) -> None:
        """Terminate the server subprocess."""
        if self._proc is None:
            return

        # Kill the process.
        try:
            self._proc.kill()
        except OSError:
            pass

        # Join the process to avoid zombies.
        try:
            self._proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
        self._proc = None
