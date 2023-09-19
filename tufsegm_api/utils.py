"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to white all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
from pathlib import Path
from typing import Union

import tufsegm_api.config as cfg
import api.config as api_cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


# make data
# = HAVE TO MODIFY FOR YOUR NEEDS =
def mkdata(input_filepath, output_filepath):
    """ Main/public function to run data processing to turn raw data
        from (data/raw) into cleaned data ready to be analyzed.
    """

    logger.info('Making final data set from raw data')

    # EXAMPLE for finding various files
    project_dir = Path(__file__).resolve().parents[2]


# create model
# = HAVE TO MODIFY FOR YOUR NEEDS =
def create_model(**kwargs):
    """Main/public method to create AI model
    """
    # define model parameters
    
    # build model based on the deep learning framework
    
    return model


def unzip(zip_file: Union[Path, str], limit_gb: int):
    """
    Unzipping files while staying below the deployment space limit.

    Args:
        zip_file: pathlib.Path or str .zip file to extract
        limit_gb: disk space limit (in GB) that shouldn't be exceeded during unpacking

    Returns:
        limit_exceeded (Bool): Turns true if no more data is allowed to be extracted

    """
    limit_exceeded = False
    # convert limit_gb to bytes
    limit_bytes = limit_gb * 1024 * 1024 * 1024

    # get the current amount of bytes stored in the data directory
    stored_bytes = sum(f.stat().st_size for f in Path(api_cfg.DATA_PATH).glob('**/*')
                       if f.is_file())

    print(f"Data folder currently contains {stored_bytes / (1024 * 3)} GB.\n"
          f"Now unpacking {zip_file}...")
    zip_command = ["unzip", str(zip_file)]
    # Capture the standard output and standard error
    process = subprocess.Popen(
        zip_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    while True:
        line = process.stdout.readline()
        if not line:
            break

        # Update the extracted size with the size of the current file
        stored_bytes += len(line)

        # Check if the extracted size exceeds the limit
        if stored_bytes >= limit_bytes:
            print(f"Exceeded maximum allowed size of {limit_gb} GB for '{api_cfg.DATA_PATH}' folder.")
            limit_exceeded = True
            process.terminate()
            break

    process.wait()

    # Check if the process was successful
    assert process.returncode == 0, f"Error during unpacking of file {zip_file}!"
    return limit_exceeded