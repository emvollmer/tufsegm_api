"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the DEEPaaS API Interface.
Do not misuse this module to define variables that are not CONSTANTS or
that are not used across the `api` package. You can use the `config.py`
file in your model package to define CONSTANTS related to your model.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
import os
import logging
from importlib import metadata
from pathlib import Path
import sys


# Default AI model
MODEL_NAME = os.getenv("MODEL_NAME", default="tufsegm_api")

# Get AI model metadata
MODEL_METADATA = metadata.metadata(MODEL_NAME)  # .json

# Fix metadata for emails from pyproject parsing
_EMAILS = MODEL_METADATA["Author-email"].split(", ")
_EMAILS = map(lambda s: s[:-1].split(" <"), _EMAILS)
MODEL_METADATA["Author-emails"] = dict(_EMAILS)

# Fix metadata for authors from pyproject parsing
_AUTHORS = MODEL_METADATA.get("Author", "").split(", ")
_AUTHORS = [] if _AUTHORS == [""] else _AUTHORS
_AUTHORS += MODEL_METADATA["Author-emails"].keys()
MODEL_METADATA["Authors"] = sorted(_AUTHORS)

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

# logging level across API modules can be setup via API_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
