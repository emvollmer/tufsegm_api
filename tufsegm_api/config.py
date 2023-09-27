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

import tufsegm_api.api.config as api_cfg

# Define submodule name and path
SUBMODULE_NAME = 'ThermUrbanFeatSegm'
SUBMODULE_PATH = Path(api_cfg.BASE_PATH, SUBMODULE_NAME)

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