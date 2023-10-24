"""Help function for detecting the model configuration file `config.json`"""
import logging
from pathlib import Path

from .style import green

logger = logging.getLogger(__name__)

FOUND = green("Found")


def detect_config(config_path: Path) -> Path:
    """Detect and return the path that points to config.json. If config_path is a directory,
    it looks for config.json below it.

    Parameters
    ---------
    config_path : pathlib.Path
        The path to config.json or the directory containing config.json.

    Returns
    -------
    config_json_path : pathlib.Path
        The path points to config.json.
    """
    if not config_path.exists():
        raise ValueError(f"{config_path} does not exist.")

    if config_path.is_dir():
        # search config.json under config_path
        config_json_path = config_path / "config.json"
        if not config_json_path.exists():
            raise ValueError(f"Fail to find config.json under {config_path}.")
    else:
        config_json_path = config_path

    logger.info("%s model configuration: %s", FOUND, config_json_path)
    return config_json_path
