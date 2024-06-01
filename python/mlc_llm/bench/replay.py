"""MLC LLM bench replay request"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


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

    wait_until_last_task_done : bool
        Whether to wait until the last task is done. Defaults to True.

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
