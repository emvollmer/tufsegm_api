"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
from pathlib import Path
import subprocess
import threading
from typing import Union
import zipfile

import tufsegm_api.config as cfg
import tufsegm_api.api.config as api_cfg

from ThermUrbanFeatSegm.scripts.setup.generate_segm_masks import main as generate_segm_masks_func
from ThermUrbanFeatSegm.scripts.setup.train_test_split import main as train_test_split_func

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


def setup(data_path: Path, test_size: int):
    """
    General setup function for annotation processing and dataset split.
    Generates segmentation masks from annotation data and splits the data into
    test and training sets.

    Args:
        data_path: pathlib.Path to the data directory.
        test_size: int. Size of the test set in splitting of train test.
    
    Raises:
        DiskSpaceExceeded: If available disk space was exceeded during setup.
    """
    # get absolute limit by comparing to remaining available space on node
    limit_gb = check_node_disk_limit(cfg.DATA_LIMIT_GB)

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space, 
                                          args=(limit_gb,), daemon=True)
        monitor_thread.start()

        print("Generating segmentation masks from annotation data...")
        generate_segm_masks_func(
            img_dir=Path(data_path, "images"),
            json_dir=Path(data_path, "annotations"),
            save_for_view=False,
            log_level=cfg.LOG_LEVEL
        )

    except DiskSpaceExceeded as e:
        logger.error(f"Disk space limit exceeded: {str(e)}")

    print("Splitting data into training and testing sets...")
    train_test_split_func(
        source_dir=None,
        destination_dir=None,
        test_size=test_size,
        log_level=cfg.LOG_LEVEL
    )
    logger.info(f"Setup complete. Data path now contains {get_disk_usage() / (1024 ** 3)} GB.")


def monitor_disk_space(limit_gb: int = cfg.LIMIT_GB):
    """
    Thread function to monitor disk space and check the current usage doesn't exceed 
    the defined limit.
    """
    limit_bytes = limit_gb * (1024 ** 3)     # convert to bytes
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