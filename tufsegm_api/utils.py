"""Package to create datasets, pipelines and other utilities.

This module is used to define all the functions needed to
operate the methods defined at `__init__.py`.
"""
import logging
from math import floor
import os
from pathlib import Path
import signal
import shutil
import subprocess
from subprocess import TimeoutExpired
from tensorflow.config.experimental import list_physical_devices
import time
import threading
from typing import Union
import zipfile

import tufsegm_api.config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


class DiskSpaceExceeded(Exception):
    """Raised when disk space is exceeded."""
    pass


class SubprocessError(Exception):
    """Raised when disk space is exceeded."""
    pass


def configure_api_logging(logger, log_level: int):
    """Define basic logging configuration

    :param logger: logger
    :param log_level: User defined input
    """
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(log_level)


def copy_remote(frompath, topath):
    """Copies remote (e.g. NextCloud) folder/file in your local deployment or
    vice versa for example:
        - `copy_remote('/storage/data/images', '/srv/myapp/data/images')`
    Ensures deployment node space isn't being exceeded during copying.

    Args:
        frompath (Path): The path to the file to be copied
        topath (Path): The path to the destination folder directory

    Raises:
        OSError: If the source isn't a directory
        FileNotFoundError: If the source file doesn't exist
    """
    frompath: Path = Path(frompath)
    topath: Path = Path(topath)

    log_disk_usage("Begin copying from NextCloud")
    limit_gb = check_node_disk_limit()  # get absolute limit by comparing with available node space

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space, 
                                          args=(limit_gb,), daemon=True)
        monitor_thread.start()

        logger.info(f"Copying from '{frompath}' to '{topath}'...")
        try:
            if frompath.is_dir():
                shutil.copytree(frompath, topath, dirs_exist_ok=True)
            else:
                shutil.copy(frompath, topath)
        except OSError as e:
            logger.error(f"Directory not copied because {frompath} "
                         f"is not a directory. Error: %s" % e)
        except FileNotFoundError as e:
            logger.error(f"Error in copying from {frompath} to {topath}. "
                         f"Error: %s" % e)
    
    except DiskSpaceExceeded as e:
        logger.error(f"Disk space limit exceeded: {str(e)}")
    
    log_disk_usage("Rclone process complete")


def unzip(zip_file: Union[Path, str]):
    """
    Unzipping files while staying below the deployment space limit.

    Args:
        zip_file: pathlib.Path or str .zip file to extract

    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during unzipping.
    """
    log_disk_usage("Beginning unzipping")
    # get limit by comparing to remaining available space on node
    limit_gb = check_node_disk_limit(cfg.DATA_LIMIT_GB)
    limit_bytes = limit_gb * (1024 ** 3)    # convert to bytes

    # get the current amount of bytes stored in the data directory
    stored_bytes = get_disk_usage(cfg.DATA_PATH)

    logger.info(f"Data folder currently contains {round(stored_bytes / (1024 ** 3), 2)} GB.\n"
                f"Now unpacking '{zip_file}'...")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if stored_bytes + file_info.file_size >= limit_bytes:
                raise DiskSpaceExceeded(f"Unzipping will exceed the maximum allowed disk space "
                                        f"of {limit_gb} GB for '{cfg.DATA_PATH}' folder.")
        # unzip the file to its current directory
        zip_ref.extractall(zip_file.parent)
    
    log_disk_usage("Unzipping complete")


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
        DiskSpaceExceeded: If available disk space was exceeded during setup.
    """
    log_disk_usage("Beginning setup")
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
        raise DiskSpaceExceeded(f"Setting up exceeds the maximum allowed disk space "
                                f"of {limit_gb} GB for '{cfg.DATA_PATH}' folder.")

    if not set(os.listdir(data_path)) >= {"masks", "train.txt", "test.txt"}:
        raise FileNotFoundError(f"Data path '{data_path}' does not contain required entries after setup!")

    log_disk_usage("Setup complete")


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
                                    f"for '{cfg.BASE_PATH}' folder.")


def check_node_disk_limit(limit_gb: int = cfg.LIMIT_GB):
    """
    Check overall data limit on node and redefine limit if necessary.

    Args:
        limit_gb: user defined disk space limit (in GB)
    
    Returns:
        available GB on node
    """
    #todo: incorrect logic here (see function below on how to start correcting!)
    try:
        # get available space on entire node
        available_gb = float(subprocess.getoutput("df -h | grep 'overlay' | awk '{print $4}'").split("G")[0])
    except ValueError as e:
        logger.info(f"ValueError: Node disk space not readable. Using provided limit of {limit_gb} GB.")
        available_gb = limit_gb

    if available_gb >= limit_gb:
        return limit_gb
    else:
        logger.warning(f"Available disk space on node ({available_gb} GB) is less than the user "
                       f"defined limit ({limit_gb} GB). Limit will be reduced to {floor(available_gb)} GB.")
        return floor(available_gb)


# def check_node_disk_limit(limit_gb: int = cfg.LIMIT_GB):
#     """
#     Check overall data limit on node and redefine limit if necessary.

#     Args:
#         limit_gb: user defined disk space limit (in GB)
    
#     Returns:
#         available GB on node
#     """
#     try:
#         # get available space on entire node
#         available_gb = float(subprocess.getoutput("df -h | grep 'overlay' | awk '{print $4}'").split("G")[0])
#     except ValueError as e:
#         logger.info(f"ValueError: Node disk space not readable. Using provided limit of {limit_gb} GB.")
#         available_gb = limit_gb

#     # get already used space from project (rounded up)
#     used_gb = ceil(get_disk_usage() / (1024 ** 3))  #todo: make this work with different limit_gb inputs or ignore!!
#     leftover_gb = limit_gb - used_gb

#     if available_gb >= leftover_gb:
#         return limit_gb
#     else:
#         # calculate new limit (which includes used space)
#         reduced_limit_gb = available_gb + used_gb
#         logger.warning(f"Available disk space on node ({available_gb} GB) is less than the user "
#                        f"defined limit ({limit_gb} GB). Limit for entire project redefined as "
#                        f" {reduced_limit_gb} GB (which includes the already used {used_gb} GB).")
#         return reduced_limit_gb


def get_disk_usage(folder: Path = cfg.BASE_PATH):
    """Get the current amount of bytes stored in the provided folder.
    """
    return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())


def log_disk_usage(process_message: str):
    """Log used disk space to the terminal with a process_message describing what has occurred.
    """
    logger.debug(f"{process_message}: Repository currently takes up {round(get_disk_usage() / (1024 ** 3), 2)} GB.")


def run_bash_subprocess(cmd: list, timeout: int = 600):
    """
    Run bash script call via subprocess command
    while printing all outputs to the terminal

    Args:
        cmd -- list of command line arguments for subprocess call
        timeout -- int. Timeout in seconds for the subprocess command
    """
    logger.debug(f"Running subprocess command with arguments: '{cmd}'")

    # check available physical devices (GPU or CPU)
    if not list_physical_devices('GPU'):
        timeout = timeout * 3
        logger.warning(f"No GPU devices detected, running on CPU. "
                       f"Extending timeout to {timeout} seconds.")

    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        return_code = process.wait(timeout=timeout)

        # check the return code to terminate in case the bash script was forcefully exited
        if return_code == 0:
            logger.info("Bash script executed successfully.")
        else:
            process.terminate()
            raise SubprocessError(
                f"Error during execution of bash script '{cmd[1]}'. "
                f"Terminated with return code {return_code}.")

    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout during execution of bash script '{cmd[1]}'.")
        process.terminate()
        raise SubprocessError(f"Timeout during execution of bash script '{cmd[1]}'.")
