"""The MLC LLM server launched in a subprocess."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

import psutil
import requests
from tvm.runtime import Device


class PopenServer:  # pylint: disable=too-many-instance-attributes
    """The wrapper of MLC LLM server, which runs the server in
    a background subprocess."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        additional_models: Optional[List[str]] = None,
        max_batch_size: Optional[int] = None,
        max_total_sequence_length: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        speculative_mode: Literal["disable", "small_draft", "eagle"] = "disable",
        spec_draft_length: int = 4,
        enable_tracing: bool = False,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Please check out `python/mlc_llm/cli/serve.py` for the server arguments."""
        self.model = model
        self.model_lib = model_lib
        self.device = device
        self.mode = mode
        self.additional_models = additional_models
        self.max_batch_size = max_batch_size
        self.max_total_sequence_length = max_total_sequence_length
        self.prefill_chunk_size = prefill_chunk_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.speculative_mode = speculative_mode
        self.spec_draft_length = spec_draft_length
        self.enable_tracing = enable_tracing
        self.host = host
        self.port = port
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:  # pylint: disable=too-many-branches
        """Launch the server in a popen subprocess.
        Wait until the server becomes ready before return.
        """
        cmd = [sys.executable]
        cmd += ["-m", "mlc_llm", "serve", self.model]
        if self.model_lib is not None:
            cmd += ["--model-lib", self.model_lib]
        cmd += ["--device", self.device]
        if self.mode is not None:
            cmd += ["--mode", self.mode]
        if self.additional_models is not None:
            cmd += ["--additional-models", *self.additional_models]
        if self.max_batch_size is not None:
            cmd += ["--max-batch-size", str(self.max_batch_size)]
        if self.max_total_sequence_length is not None:
            cmd += ["--max-total-seq-length", str(self.max_total_sequence_length)]
        if self.prefill_chunk_size is not None:
            cmd += ["--prefill-chunk-size", str(self.prefill_chunk_size)]
        if self.speculative_mode != "disable":
            cmd += [
                "--speculative-mode",
                self.speculative_mode,
                "--spec-draft-length",
                str(self.spec_draft_length),
            ]
        if self.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.enable_tracing:
            cmd += ["--enable-tracing"]

        cmd += ["--host", self.host]
        cmd += ["--port", str(self.port)]
        process_path = str(Path(__file__).resolve().parents[4])
        self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd, cwd=process_path, env=os.environ
        )
        # NOTE: DO NOT USE `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
        # in subprocess.Popen here. PIPE has a fixed-size buffer with may block
        # and hang forever.

        # Try to query the server until it is ready.
        openai_v1_models_url = f"http://{self.host}:{str(self.port)}/v1/models"
        query_result = None
        timeout = 60
        attempts = 0.0
        while query_result is None and attempts < timeout:
            try:
                query_result = requests.get(openai_v1_models_url, timeout=60)
                if query_result.status_code != 200:
                    query_result = None
                    attempts += 0.1
                    time.sleep(0.1)
            except:  # pylint: disable=bare-except
                attempts += 0.1
                time.sleep(0.1)

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

        # Kill all the child processes.
        def kill_child_processes():
            try:
                parent = psutil.Process(self._proc.pid)
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                return

            for process in children:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass

        kill_child_processes()

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

    def __enter__(self):
        """Start the server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminate the server."""
        self.terminate()
