"""The MLC LLM server launched in a subprocess."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Union

import psutil
import requests
from tvm.runtime import Device

from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.engine_base import _check_engine_config


class PopenServer:  # pylint: disable=too-many-instance-attributes
    """The wrapper of MLC LLM server, which runs the server in
    a background subprocess.

    This server can be used for debugging purposes.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        engine_config: Optional[EngineConfig] = None,
        enable_debug: bool = True,
        enable_tracing: bool = False,
        host: str = "127.0.0.1",
        port: int = 8082,
    ) -> None:
        """Please check out `python/mlc_llm/cli/serve.py` for the server arguments."""
        # - Check the fields fields of `engine_config`.
        if engine_config is None:
            engine_config = EngineConfig()
        _check_engine_config(model, model_lib, mode, engine_config)

        self.model = model
        self.model_lib = model_lib
        self.device = device
        self.mode = mode
        self.enable_debug = enable_debug
        self.engine_config = engine_config
        self.enable_tracing = enable_tracing
        self.enable_debug = enable_debug
        self.host = host
        self.port = port
        self._proc: Optional[subprocess.Popen] = None

        self.base_url = ""
        self.openai_v1_base_url = ""

    def start(  # pylint: disable=too-many-branches,too-many-statements
        self, extra_env=None
    ) -> None:
        """Launch the server in a popen subprocess.
        Wait until the server becomes ready before return.
        """
        extra_env = extra_env or {}
        cmd = [sys.executable]
        cmd += ["-m", "mlc_llm", "serve", self.model]
        if self.model_lib is not None:
            cmd += ["--model-lib", self.model_lib]
        cmd += ["--device", self.device]

        if self.enable_debug:
            cmd += ["--enable-debug"]

        if self.mode is not None:
            cmd += ["--mode", self.mode]

        if len(self.engine_config.additional_models) > 0:
            args_additional_model = []
            for additional_model in self.engine_config.additional_models:
                if isinstance(additional_model, str):
                    args_additional_model.append(additional_model)
                else:
                    args_additional_model.append(additional_model[0] + "," + additional_model[1])
            cmd += ["--additional-models", *args_additional_model]
        cmd += ["--speculative-mode", self.engine_config.speculative_mode]
        cmd += ["--prefix-cache-mode", self.engine_config.prefix_cache_mode]

        args_overrides = []
        if self.engine_config.max_num_sequence is not None:
            args_overrides.append(f"max_num_sequence={self.engine_config.max_num_sequence}")
        if self.engine_config.max_total_sequence_length is not None:
            args_overrides.append(
                f"max_total_seq_length={self.engine_config.max_total_sequence_length}"
            )
        if self.engine_config.prefill_chunk_size is not None:
            args_overrides.append(f"prefill_chunk_size={self.engine_config.prefill_chunk_size}")
        if self.engine_config.max_history_size is not None:
            args_overrides.append(f"max_history_size={self.engine_config.max_history_size}")
        if self.engine_config.gpu_memory_utilization is not None:
            args_overrides.append(
                f"gpu_memory_utilization={self.engine_config.gpu_memory_utilization}"
            )
        if self.engine_config.spec_draft_length is not None:
            args_overrides.append(f"spec_draft_length={self.engine_config.spec_draft_length}")
        if self.engine_config.prefix_cache_max_num_recycling_seqs is not None:
            args_overrides.append(
                "prefix_cache_max_num_recycling_seqs="
                + str(self.engine_config.prefix_cache_max_num_recycling_seqs)
            )
        if len(args_overrides) > 0:
            cmd += ["--overrides", ";".join(args_overrides)]

        if self.enable_tracing:
            cmd += ["--enable-tracing"]
        if self.enable_debug:
            cmd += ["--enable-debug"]

        cmd += ["--host", self.host]
        cmd += ["--port", str(self.port)]
        process_path = str(Path(__file__).resolve().parents[4])
        final_env = os.environ.copy()
        for key, value in extra_env.items():
            final_env[key] = value
        self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd, cwd=process_path, env=final_env
        )
        # NOTE: DO NOT USE `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
        # in subprocess.Popen here. PIPE has a fixed-size buffer with may block
        # and hang forever.

        # Try to query the server until it is ready.
        self.base_url = f"http://{self.host}:{str(self.port)}"
        self.openai_v1_base_url = f"http://{self.host}:{str(self.port)}/v1"
        openai_v1_models_url = f"{self.base_url}/v1/models"

        query_result = None
        timeout = 120
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
