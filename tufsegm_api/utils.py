"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
import os
from pathlib import Path
import subprocess
from subprocess import TimeoutExpired
import time
import threading
from typing import Union
import zipfile

import tufsegm_api.config as cfg
import tufsegm_api.api.config as api_cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


class DiskSpaceExceeded(Exception):
    """Raised when disk space is exceeded."""
    pass


def unzip(zip_file: Union[Path, str]):
    """
    Unzipping files while staying below the deployment space limit.

    Args:
        zip_file: pathlib.Path or str .zip file to extract

    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during unzipping.
    """
    # get limit by comparing to remaining available space on node
    limit_gb = check_node_disk_limit(cfg.DATA_LIMIT_GB)
    limit_bytes = limit_gb * (1024 ** 3)    # convert to bytes

    # get the current amount of bytes stored in the data directory
    stored_bytes = get_disk_usage()

    print(f"Data folder currently contains {round(stored_bytes / (1024 ** 3), 2)} GB.\n"
          f"Now unpacking '{zip_file}'...")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if stored_bytes + file_info.file_size >= limit_bytes:
                raise DiskSpaceExceeded(f"Unzipping will exceed the maximum allowed disk space "
                                        f"of {limit_gb} GB for '{api_cfg.DATA_PATH}' folder.")
        # unzip the file to its current directory
        zip_ref.extractall(zip_file.parent)


def setup(data_path: Path, test_size: int, save_for_view: bool = False):
    """
    General setup function for annotation processing and dataset split.
    Generates segmentation masks from annotation data and splits the data into
    test and training sets.

    Args:
        data_path: pathlib.Path to the data directory.
        test_size: int. Size of the test set in splitting of train test.
    
    Raises:
        FileNotFoundError: If the necessary files don't exist after setup.
    """
    # get absolute limit by comparing to remaining available space on node
    limit_gb = check_node_disk_limit(cfg.DATA_LIMIT_GB)

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space, 
                                          args=(limit_gb,), daemon=True)
        monitor_thread.start()

        setup_cmd = ["/bin/bash",
                     str(Path(cfg.SUBMODULE_PATH, 'scripts', 'setup', 'setup.sh')),
                     "-j", str(Path(data_path, 'annotations')),
                     "-i", str(Path(data_path, 'images')),
                     "--test-size", str(test_size),
                     cfg.VERBOSITY]
        if save_for_view:
            setup_cmd.insert(-1,"--save-for-view")

        run_bash_subprocess(setup_cmd)

    except DiskSpaceExceeded as e:
        logger.error(f"Disk space limit exceeded: {str(e)}")

    if not all(e in os.listdir(str(data_path)) for e in ["masks", "train.txt", "test.txt"]):
        raise FileNotFoundError(f"Data path '{data_path}' does not contain required entries after setup!")

    print(f"Setup complete. Data path now contains {round(get_disk_usage() / (1024 ** 3), 2)} GB.")


def monitor_disk_space(limit_gb: int = cfg.LIMIT_GB):
    """
    Thread function to monitor disk space and check the current usage doesn't exceed 
    the defined limit.

    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during threading.
    """
    limit_bytes = limit_gb * (1024 ** 3)  # convert to bytes
    while True:
        time.sleep(10)

        stored_bytes = get_disk_usage()

        if stored_bytes >= limit_bytes:
            raise DiskSpaceExceeded(f"Exceeded maximum allowed disk space of {limit_gb} GB "
                                    f"for '{api_cfg.DATA_PATH}' folder.")


def check_node_disk_limit(limit_gb: int = cfg.LIMIT_GB):
    """
    Check overall data limit on node and redefine limit if necessary.

    Args:
        limit_gb: user defined disk space limit (in GB)
    
    Returns:
        available GB on node
    """
    try:
        # get available space on entire node
        available_gb = int(subprocess.getoutput("df -h | grep 'overlay' | awk '{print $4}'").split("G")[0])
    except ValueError as e:
        logger.info(f"ValueError: Node disk space not readable. Using provided limit of {limit_gb} GB.")
        available_gb = limit_gb

    if available_gb >= limit_gb:
        return limit_gb
    else:
        logger.warning(f"Available disk space on node ({available_gb} GB) is less than the user "
                       f"defined limit ({limit_gb} GB). Limit will be reduced to {available_gb} GB.")
        return available_gb


def get_disk_usage(folder: Path = api_cfg.DATA_PATH):
    """Get the current amount of bytes stored in the provided folder.
    """
    return sum(f.stat().st_size for f in folder.glob('**/*') if f.is_file())


def run_bash_subprocess(cmd: list, timeout: int = 600):
    """
    Run bash script call via subprocess command
    while printing all outputs to the terminal

    Args:
        cmd -- list of command line arguments for subprocess call
        timeout -- int. Timeout in seconds for the subprocess command
    """
    print(f"Running subprocess command with arguments: '{cmd}'")
    logger.debug(f"Running subprocess command with arguments: '{cmd}'")
    with subprocess.Popen(
        args=cmd,
        stderr=subprocess.PIPE,
        text=True
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)  # required for multiple module run via bash
            if process.returncode != 0:
               print(f"Subprocess exited with a non-zero return code")
        except TimeoutExpired:
            print(f"Timeout during execution of bash script '{cmd}'.")
            process.kill()
        except Exception as exc:
            print(f"Error during execution of bash script '{cmd}'", exc)
            process.kill()
