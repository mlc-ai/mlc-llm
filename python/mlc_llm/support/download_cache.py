"""Common utilities for downloading files from HuggingFace or other URLs online."""

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import requests  # pylint: disable=import-error

from . import logging, tqdm
from .constants import (
    MLC_DOWNLOAD_CACHE_POLICY,
    MLC_LLM_HOME,
    MLC_LLM_READONLY_WEIGHT_CACHE,
    MLC_TEMP_DIR,
)
from .style import bold

logger = logging.getLogger(__name__)


def log_download_cache_policy():
    """log current download policy"""
    logger.info(
        "%s = %s. Can be one of: ON, OFF, REDO, READONLY",
        bold("MLC_DOWNLOAD_CACHE_POLICY"),
        MLC_DOWNLOAD_CACHE_POLICY,
    )


def _ensure_directory_not_exist(path: Path, force_redo: bool) -> None:
    if path.exists():
        if force_redo:
            logger.info("Deleting existing directory: %s", path)
            shutil.rmtree(path)
        else:
            raise ValueError(f"Directory already exists: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def git_clone(url: str, destination: Path, ignore_lfs: bool) -> None:
    """Clone a git repository into a directory."""
    repo_name = ".tmp"
    command = ["git", "clone", url, repo_name]
    _ensure_directory_not_exist(destination, force_redo=False)
    try:
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as tmp_dir:
            logger.info("[Git] Cloning %s to %s", bold(url), destination)
            subprocess.run(
                command,
                env=env,
                cwd=tmp_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            git_dir = os.path.join(tmp_dir, repo_name)
            if not ignore_lfs:
                git_lfs_pull(Path(git_dir))
            shutil.move(git_dir, str(destination))
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Git clone failed with return code {error.returncode}: {error.stderr}. "
            f"The command was: {command}"
        ) from error


def git_lfs_pull(repo_dir: Path, ignore_extensions: Optional[List[str]] = None) -> None:
    """Pull files with Git LFS."""
    filenames = (
        subprocess.check_output(
            ["git", "-C", str(repo_dir), "lfs", "ls-files", "-n"],
            stderr=subprocess.STDOUT,
        )
        .decode("utf-8")
        .splitlines()
    )
    if ignore_extensions is not None:
        filenames = [
            filename
            for filename in filenames
            if not any(filename.endswith(extension) for extension in ignore_extensions)
        ]
    logger.info("[Git LFS] Downloading %d files with Git LFS: %s", len(filenames), filenames)
    with tqdm.redirect():
        for file in tqdm.tqdm(filenames):
            logger.info("[Git LFS] Downloading %s", file)
            subprocess.check_output(
                ["git", "-C", str(repo_dir), "lfs", "pull", "--include", file],
                stderr=subprocess.STDOUT,
            )


def download_file(
    url: str,
    destination: Path,
    md5sum: Optional[str],
) -> Tuple[str, Path]:
    """Download a file from a URL to a destination file."""
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()  # type: ignore
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):  # type: ignore
                file.write(chunk)
    if md5sum is not None:
        hash_md5 = hashlib.md5()
        with destination.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                hash_md5.update(chunk)
        file_md5 = hash_md5.hexdigest()
        if file_md5 != md5sum:
            raise ValueError(
                f"MD5 checksum mismatch for downloaded file: {destination}. "
                f"Expected {md5sum}, got {file_md5}"
            )
    return url, destination


def download_and_cache_mlc_weights(  # pylint: disable=too-many-locals
    model_url: str,
    num_processes: int = 4,
    force_redo: Optional[bool] = None,
) -> Path:
    """Download weights for a model from the HuggingFace Git LFS repo."""
    log_download_cache_policy()
    if MLC_DOWNLOAD_CACHE_POLICY == "OFF":
        raise RuntimeError(f"Cannot download {model_url} as MLC_DOWNLOAD_CACHE_POLICY=OFF")

    prefixes, mlc_prefix = ["HF://", "https://huggingface.co/"], ""
    mlc_prefix = next(p for p in prefixes if model_url.startswith(p))
    assert mlc_prefix

    git_url_template = "https://huggingface.co/{user}/{repo}"
    bin_url_template = "https://huggingface.co/{user}/{repo}/resolve/main/{record_name}"

    if model_url.count("/") != 1 + mlc_prefix.count("/") or not model_url.startswith(mlc_prefix):
        raise ValueError(f"Invalid model URL: {model_url}")
    user, repo = model_url[len(mlc_prefix) :].split("/")
    domain = "hf"

    readonly_cache_dirs = []
    for base in MLC_LLM_READONLY_WEIGHT_CACHE:
        cache_dir = base / domain / user / repo
        readonly_cache_dirs.append(str(cache_dir))
        if (cache_dir / "mlc-chat-config.json").is_file():
            logger.info("Use cached weight: %s", bold(str(cache_dir)))
            return cache_dir

    if force_redo is None:
        force_redo = MLC_DOWNLOAD_CACHE_POLICY == "REDO"

    git_dir = MLC_LLM_HOME / "model_weights" / domain / user / repo
    readonly_cache_dirs.append(str(git_dir))

    try:
        _ensure_directory_not_exist(git_dir, force_redo=force_redo)
    except ValueError:
        logger.info("Weights already downloaded: %s", bold(str(git_dir)))
        return git_dir

    if MLC_DOWNLOAD_CACHE_POLICY == "READONLY":
        raise RuntimeError(
            f"Cannot find cache for {model_url}, "
            "cannot proceed to download as MLC_DOWNLOAD_CACHE_POLICY=READONLY, "
            "please check settings MLC_LLM_READONLY_WEIGHT_CACHE, "
            f"local path candidates: {readonly_cache_dirs}"
        )

    with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as tmp_dir_prefix:
        tmp_dir = Path(tmp_dir_prefix) / "tmp"
        git_url = git_url_template.format(user=user, repo=repo)
        git_clone(git_url, tmp_dir, ignore_lfs=True)
        git_lfs_pull(tmp_dir, ignore_extensions=[".bin"])
        shutil.rmtree(tmp_dir / ".git", ignore_errors=True)
        with (tmp_dir / "ndarray-cache.json").open(encoding="utf-8") as in_file:
            param_metadata = json.load(in_file)["records"]
        with cf.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for record in param_metadata:
                record_name = record["dataPath"]
                file_url = bin_url_template.format(user=user, repo=repo, record_name=record_name)
                file_dest = tmp_dir / record_name
                file_md5 = record.get("md5sum", None)
                futures.append(executor.submit(download_file, file_url, file_dest, file_md5))
            with tqdm.redirect():
                for future in tqdm.tqdm(cf.as_completed(futures), total=len(futures)):
                    file_url, file_dest = future.result()
                    logger.info("Downloaded %s to %s", file_url, file_dest)
        logger.info("Moving %s to %s", tmp_dir, bold(str(git_dir)))
        shutil.move(str(tmp_dir), str(git_dir))
    return git_dir


def get_or_download_model(model: str) -> Path:
    """Use user-provided argument ``model`` to get model_path

    We define "valid" as having an ``mlc-chat-config.json`` right under the folder.

    Parameters
    ----------
    model : str
        User's input; may a path or url

    Returns
    ------
    model_path : Path
        A "valid" path to model folder, with
        ``(model_path / "mlc-chat-config.json").is_file`` being True

    Note
    ----
    This function may perform additional download and caching

    Raises
    ------
    FileNotFoundError: if we cannot find a valid `model_path`.
    """
    if model.startswith("HF://"):
        logger.info("Downloading model from HuggingFace: %s", model)
        model_path = download_and_cache_mlc_weights(model)
    else:
        model_path = Path(model)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Cannot find model {model}, directory does not exist")
    mlc_config_path = model_path / "mlc-chat-config.json"
    if mlc_config_path.is_file():
        return model_path
    raise FileNotFoundError(f"Cannot find {str(mlc_config_path)} in the model directory provided")
