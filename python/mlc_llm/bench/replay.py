"""MLC LLM bench replay request"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd


class OpenAIRequestSender:
    """
    Handles the sending of asynchronous HTTP requests.

    Parameters
    ----------
    host : Optional[str]
        The host address for the API, by default "127.0.0.1".

    port : Optional[int]
        The port number for the API, by default 8008.

    stream : Optional[bool]
        Indicates whether streaming should be enabled. Default is True.

    timeout : Optional[float]
        The timeout in seconds for each request, by default 180.
    """

    def __init__(
        self,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 8008,
        stream: Optional[bool] = True,
        timeout: Optional[float] = 180,
    ):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.stream = stream
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("MLC_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('MLC_API_KEY')}"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.session.close()

    async def __call__(self, params):
        """
        Sends an asynchronous HTTP POST request using the class's aiohttp session.

        Parameters
        ----------
        params : dict
            The parameters for the request, including url, headers, and payload.

        Returns
        -------
        response : dict
            The JSON response from the server or None if an error occurs.
        """
        try:
            url = params.get("url", self.url)
            headers = params.get("headers", self.headers)
            payload = params.get(
                "payload",
                {key: value for key, value in params.items() if key != "timestamp"},
            )
            if self.session:
                async with self.session.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                ) as response:
                    return await response.json()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, json=payload, timeout=self.timeout
                    ) as response:
                        return await response.json()
        except Exception as err:  # pylint: disable=broad-except
            print(f"Error in send request: {err}")
            return


def load_replay_log(log_path: str) -> List[Dict]:
    """
    Load replay log from file

    Parameters
    ----------
    log_path : str
        The path to the event log CSV or JSONL file containing the events to replay.

    Returns
    -------
    res: List[Dict]
        A list of preprocessed event data for replay.
    """
    if log_path.endswith(".csv"):
        df = pd.read_csv(log_path)
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
    if log_path.endswith(".jsonl"):
        with open(log_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
            for row in data:
                row["timestamp"] = datetime.fromisoformat(str(row["timestamp"]))
        return data
    raise ValueError("Unsupported file format. Please use .csv or .jsonl.")


async def replay(
    replay_log: List[Dict],
    callback,
    *,
    base_timestamp: Optional[float] = None,
    start_timestamp: Optional[float] = None,
    max_schedule_gap: Optional[float] = 0.1,
    wait_until_last_task_done: bool = True,
):  # pylint: disable=too-many-arguments
    """
    Replay generated events based on historical timestamps. The replaying requests start
    from a new start time while preserving the ordering of requests.

    Parameters
    ----------
    replay_log : List[Dict]
        A list of event data, each containing a 'timestamp' and replay parameters.

    callback : coroutine function
        The async function to be called for each log item.

    base_timestamp : Optional[float]
        The timestamp of the first log entry, used as a reference point for scheduling.
        Defaults to the timestamp of the first item in `replay_log`.

    start_timestamp : Optional[float]
        The time when the replay starts.

    max_schedule_gap : Optional[float]
        The maximum allowed delay between the scheduled time in seconds. Defaults to 0.1 seconds.

    Raises
    ------
    TypeError
        If the callback is not a coroutine or an awaitable function.
    """
    if not replay_log:
        return
    loop = asyncio.get_running_loop()
    if base_timestamp is None:
        base_timestamp = replay_log[0]["timestamp"].timestamp()
    if start_timestamp is None:
        start_timestamp = loop.time() + max_schedule_gap

    for item in replay_log:
        cur_time = loop.time()
        launch_time = item["timestamp"].timestamp() - base_timestamp + start_timestamp
        if launch_time - cur_time > max_schedule_gap:
            await asyncio.sleep(launch_time - cur_time - max_schedule_gap)
        loop.call_at(
            launch_time,
            lambda: asyncio.create_task(callback(item)),  # pylint: disable=cell-var-from-loop
        )

    if wait_until_last_task_done:
        # Wait for all tasks to be scheduled
        await asyncio.sleep(launch_time - loop.time() + max_schedule_gap)
        await asyncio.gather(*asyncio.all_tasks(loop) - {asyncio.current_task()})
