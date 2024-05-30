"""MLC LLM bench replay request"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

from mlc_llm.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)


class RequestReplayer:
    """
    Replay generated events based on historical timestamps. The replaying requests start
    from a new start time while preserving the intervals between requests.

    Parameters
    ----------
    log_path : str
        The path to the event log CSV or JSONL file containing the events to replay.

    host : Optional[str]
        The host address for the API. Default is "127.0.0.1".

    port : Optional[int]
        The port number for the API. Default is 8008.

    stream : Optional[bool]
        Indicates whether the streaming should be enabled. Default is True.

    timeout : Optional[int]
        The timeout in seconds for each request. Default is 180.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        log_path: str,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 8008,
        stream: Optional[bool] = True,
        timeout: Optional[int] = 180,
    ) -> None:
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.stream = stream
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("MLC_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('MLC_API_KEY')}"
        self.request_params = self.get_request_params(log_path)

    def get_request_params(self, log_path: str) -> List[Dict]:
        """
        Loads and preprocesses the event log from either a CSV or JSONL file to prepare payloads and
        request parameters for replay.

        Parameters
        ----------
        log_path : str
            The path to the event log CSV or JSONL file containing the events to replay.

        Returns
        -------
        res: List[Dict]
            A list of preprocessed event data dictionaries for replay.
        """
        if log_path.endswith(".csv"):
            return self._load_csv(log_path)
        if log_path.endswith(".jsonl"):
            return self._load_jsonl(log_path)
        raise ValueError("Unsupported file format. Please use .csv or .jsonl.")

    def _load_csv(self, filepath: str) -> List[Dict]:
        """
        Loads parameters from a CSV file and returns a list of event data dictionaries.

        Parameters
        ----------
        filepath : str
            The path to the CSV file.

        Returns
        -------
        res: List[Dict]
            A list of event data dictionaries from the CSV file, sorted by timestamp.
        """
        df = pd.read_csv(filepath)
        column_names = df.columns.values
        assert (
            ("Date" in column_names)
            and ("@request" in column_names)
            and ("Message" in column_names)
        )
        df["timestamp"] = pd.to_datetime(df["Date"])
        df.sort_values("timestamp", inplace=True)
        # Get the request params from the loaded CSV
        params = []
        for _, row in df.iterrows():
            request = row["@request"]
            payload = json.loads(str(request))
            params.append(
                {
                    "timestamp": row["timestamp"],
                    "payload": payload,
                }
            )
        return params

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """
        Loads parameters from a JSONL file and returns a list of event data dictionaries.

        Parameters
        ----------
        filepath : str
            The path to the JSONL file.

        Returns
        -------
        res: List[Dict]
            A list of event data dictionaries from the JSONL file, sorted by timestamp.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
            for item in data:
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
        data.sort(key=lambda x: x["timestamp"])
        return data

    async def send_request(self, session, params: Dict):
        """
        Sends an asynchronous HTTP POST request using an aiohttp session.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The active aiohttp client session.

        params : dict
            The parameters for the request, including URL, headers, and payload, etc.

        Returns
        -------
        res
            The JSON response from the server or None if an error occurs.
        """
        try:
            url = params.get("address", self.url)
            headers = params.get("headers", self.headers)
            payload = params["payload"]
            async with session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            ) as response:
                return await response.json()
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Error in send request %s", err)

    async def run(self):
        """
        Replays the stored requests.

        Utilizes the asyncio event loop to schedule each request according to its timestamp,
        adjusting for the current time.
        """
        if not self.request_params:
            return
        async with aiohttp.ClientSession() as session:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            first_timestamp = self.request_params[0]["timestamp"]

            for params in self.request_params:
                original_delay = (params["timestamp"] - first_timestamp).total_seconds()
                loop.call_at(
                    start_time + original_delay,
                    lambda p=params: asyncio.create_task(self.send_request(session, p)),
                )

            last_delay = (self.request_params[-1]["timestamp"] - first_timestamp).total_seconds()
            await asyncio.sleep(last_delay + 1)
