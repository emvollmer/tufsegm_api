"""Module to define CONSTANTS used across the AI-model package.

This module is used to define CONSTANTS used across the AI-model package.
Do not misuse this module to define variables that are not CONSTANTS or
that are exclusive to the `api` package. You can use the `config.py`
inside `api` to define exclusive CONSTANTS related to your interface.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
import logging
import os
from pathlib import Path
import sys

MODULE_NAME = 'TUFSeg'

# DEEPaaS can load more than one installed models. Therefore, in order to
# avoid conflicts, each default PATH environment variables should lead to
# a different folder. The current practice is to use the path from where the
# model source is located.
BASE_PATH = Path(__file__).resolve(strict=True).parents[1]
if str(BASE_PATH) not in sys.path:
    print(f"BASE_PATH '{BASE_PATH}' not in sys.path. "
          f"Adding to allow for imports from submodule...")
    sys.path.insert(0, str(BASE_PATH))

# Path definition for data folder
DATA_PATH = os.getenv("DATA_PATH", default=Path(BASE_PATH, "data"))
# Path definition for the pre-trained models
MODELS_PATH = os.getenv("MODELS_PATH", default=Path(BASE_PATH, "models"))

MODEL_SUFFIX = ".hdf5"

# Remote (rshare) paths for data and models
REMOTE_PATH = os.getenv("REMOTE_PATH", default="rshare:tufsegm")
REMOTE_DATA_PATH = os.getenv("REMOTE_DATA_PATH", default=Path(REMOTE_PATH, "data"))
REMOTE_MODELS_PATH = os.getenv("REMOTE_MODELS_PATH", default=Path(REMOTE_PATH, "models"))

# Define submodule name and path
SUBMODULE_NAME = 'ThermUrbanFeatSegm'
SUBMODULE_PATH = Path(BASE_PATH, SUBMODULE_NAME)

# configure logging:
# logging level across various modules can be setup via USER_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("USER_LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())

if LOG_LEVEL == 10:
    VERBOSITY = "-vv"
elif LOG_LEVEL == 20:
    VERBOSITY = "-v"
else:
    VERBOSITY = "--quiet"

# Data limits on node to uphold
LIMIT_GB = int(os.getenv("LIMIT_GB", default="20"))
DATA_LIMIT_GB = int(os.getenv("DATA_LIMIT_GB", default="15"))

# Remote MLFlow server
MLFLOW_REMOTE_SERVER="https://mlflow.dev.ai4eosc.eu/"
MLFLOW_EXPERIMENT_NAME=MODULE_NAME + "_training"
