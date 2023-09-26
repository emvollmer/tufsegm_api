"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However, you may
need to modify them for your needs.
"""
import logging
from pathlib import Path
import subprocess
import sys
from subprocess import TimeoutExpired
from typing import Union

from tufsegm_api.api import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def ls_dirs(path: Path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    dirscan = (x.name for x in path.iterdir() if x.is_dir())
    return sorted(dirscan)


def ls_files(path: Path, pattern: str):
    """Utility to return a list of files available in `path` folder.

    Arguments:
        path -- Directory path to scan.
        pattern -- File pattern to filter found files. See glob.glob() python.

    Returns:
        A list of strings for files found according to the pattern.
    """
    logger.debug("Scanning for %s files at: %s", pattern, path)
    dirscan = (x.name for x in path.glob(pattern))
    return sorted(dirscan)


def ls_remote_dirs(suffix: str, exclude: Union[None, str] = None, timeout=600):
    """Utility to return a list of remote (e.g. NextCloud) directories
    containing files with a specific suffix.
        - `ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude='perun_results')`
        - `ls_remote_dirs(suffix='.zip')`

    Arguments:
        suffix -- File suffix to filter found files.
        exclude -- String to exclude specific files with from the list of directories.
        timeout -- Timeout in seconds for the reading command.

    Returns:
        A tuple with stdout and stderr from the command.
    """
    frompath = config.REMOTE_PATH
    with subprocess.Popen(
        args=["rclone", "lsf", f"{frompath}", "-R", "--absolute"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True,  # Return strings rather than bytes
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)

            files = outs.splitlines()
            if exclude is not None:
                dirscan = [frompath + str(Path(f).parent).rstrip("/") for f in files 
                           if f.endswith(suffix)
                           if not exclude in f]
            else:
                dirscan = [frompath + str(Path(f).parent).rstrip("/") for f in files 
                           if f.endswith(suffix)]
            return list(set(dirscan))   # removes duplicates
        except TimeoutExpired:
            logger.error(f"Timeout when reading remote directory '{frompath}'.")
            process.kill()
            outs, errs = process.communicate()
            return []
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Error reading remote directory '{frompath}'\n{exc}")
            process.kill()
            outs, errs = process.communicate()
            return []


def copy_remote(frompath, topath, timeout=600):
    """Copies remote (e.g. NextCloud) folder in your local deployment or
    vice versa for example:
        - `copy_remote('rshare:data/images', '/srv/myapp/data/images')`

    Arguments:
        frompath -- Source folder to be copied.
        topath -- Destination folder.
        timeout -- Timeout in seconds for the copy command.

    Returns:
        A tuple with stdout and stderr from the command.
    """
    with subprocess.Popen(
        args=["rclone", "copy", f"{frompath}", f"{topath}"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True,  # Return strings rather than bytes
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)
        except TimeoutExpired:
            print("Timeout when copying from/to remote directory.")
            process.kill()
            outs, errs = process.communicate()
        except Exception as exc:  # pylint: disable=broad-except
            print("Error copying from/to remote directory\n", exc)
            process.kill()
            outs, errs = process.communicate()
    return outs, errs


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
