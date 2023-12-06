"""Endpoint functions to integrate the submodule TUFSeg with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import getpass
import logging
import os
from pathlib import Path
import numpy as np

from aiohttp.web import HTTPException

import tufsegm_api as aimodel

from tufsegm_api.api import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
            "datasets_local": utils.ls_dirs(config.DATA_PATH),
            "datasets_remote": utils.ls_remote_dirs(suffix=".zip", exclude="additional_data"),
            "models_local": utils.ls_dirs(config.MODELS_PATH),
            "models_remote": utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude='perun_results'),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(accept='application/json', **options):
    """Performs model prediction from given input data and parameters.

    Arguments:
        accept -- Response parser type, default is json.
        **options -- keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    print(f"User provided:\n{options}")
    try:  # Call the AI model predict() method
        if 'model_folder' in options.keys() and 'model_folder_new' in options.keys():
            raise ValueError(
                'Only one model can be selected for performing inference - either from the '
                'offered list or entered folder path.'
            )
        if 'model_folder' not in options.keys() and 'model_folder_new' not in options.keys():
            raise ValueError(
                'No model folder path for inference was selected / provided! Select either '
                'an option from the "model_folder" list or enter a value into the "model_folder_new" field.'
            )

        try:
            # Model folder path is from input field 'model_folder'
            options['model_name'] = options['model_folder']
        except TypeError:
            # Model folder path is from input field 'model_folder_new'
            options['model_name'] = options['model_folder_new']

        logger.info(f"Using model '{options['model_name']}' for predictions")
        print(f"Predicting with the user defined options: ", options)    # logger.debug
        result = aimodel.predict(**options)
        logger.debug(f"Predict result: ", result)
        print(f"Returning content_type for: ", accept)    # logger.info
        return responses.content_types[accept](result, **options)
    except Exception as err:
        logger.error("Error while doing predictions: %s", err, exc_info=True)
        # raise HTTPException(reason=err) from err      # with this uncommented we get a TypeError, but without any raise we get a 200 response
        raise


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**options):
    """Performs model training from given input data and parameters.

    Arguments:
        **options -- keyword arguments from TrainArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        Parsed history/summary of the training process.
    """
    if options['mlflow_username']:
        # for direct API calls via HTTP we need to inject credentials
        MLFLOW_TRACKING_USERNAME = options['mlflow_username']
        print(f"MLFlow model experiment tracking with account\nUsername: {MLFLOW_TRACKING_USERNAME}")
        MLFLOW_TRACKING_PASSWORD =  getpass.getpass()  # inject password by typing manually
        # for MLFLow-way we have to set the following environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        os.environ['LOGNAME'] = MLFLOW_TRACKING_USERNAME

    try:  # Call the AI model train() method
        print(f"Retraining a '{options['model_type']}' model")    # logger.info
        logger.debug(f"Training with the user defined options: {options}")
        result = aimodel.train(**options)
        logger.debug(f"Training result: {result}")
        return result
    except Exception as err:
        logger.error("Error while training: %s", err, exc_info=True)
        raise  # Reraise the exception after log


if __name__ == "__main__":
    metadata = get_metadata()
    print(f"Metadata:\n{metadata}")

    # train_args = {
    #     'mlflow,_username': None
    #     'model_type': 'UNet',
    #     'dataset_path': None,
    #     'save_for_viewing': False,
    #     'test_size': 0.2,
    #     'channels': 4,
    #     'processing': "basic",
    #     'img_size': "320x256", # "640x512",
    #     'epochs': 1,
    #     'batch_size': 4, # 8,
    #     'lr': 0.001,
    #     'seed': 42
    # }
    # train(
    #     **train_args
    # )

    # pred_args = {
    #     'model_folder': 'rshare:tufsegm/models/2023-09-21_11-42-18',    # '2023-09-27_16-19-41'
    #     'input_file': 'KA_01/DJI_0_0001_R.npy',   # None
    #     'display': False,
    #     #'save': True,
    #     #'accept': 'application/json'
    # }
    # predict(
    #     accept='application/json',
    #     **pred_args
    # )