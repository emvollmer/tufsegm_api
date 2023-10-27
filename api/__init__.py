"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging
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
        logger.info("Collecting metadata from: %s", config.MODEL_NAME)
        metadata = {
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get("author-emails"),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "datasets_local": utils.ls_dirs(Path(config.DATA_PATH)),
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
    try:  # Call your AI model predict() method
        print(f"Using model '{options['model_name']}' for predictions")
        try:
            # Input file is local
            options['input_file'] = str(Path(config.DATA_PATH, "images", options['input_file_local']))
            print(f"Predicting on image: ", options['input_file'])
        # TODO: Uncomment when input_file_external TODO is fixed
        # except TypeError:
        #     # Input file is external
        #     options['input_file'] = options['input_file_external'].filename
        #     print(f"Predicting on image: ", options['input_file_external'].original_filename)
        except Exception:
            raise HTTPException(reason=err) from err

        print(f"Predicting with the user defined options: ", options)    # logger.debug
        result = aimodel.predict(**options)
        logger.debug(f"Predict result: ", result)
        print(f"Returning content_type for: ", accept)    # logger.info
        return responses.content_types[accept](result, **options)
    except Exception as err:
        logger.error("Error while doing predictions: %s", err, exc_info=True)
        raise HTTPException(reason=err) from err


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
    try:  # Call your AI model train() method
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

    pred_args = {
        'model_name': 'rshare:tufsegm/models/2023-09-21_11-42-18',    # '2023-09-27_16-19-41'
        'input_file_local': 'KA_01/DJI_0_0001_R.npy',   # None
        #'input_file_remote': None,     # UploadedFile(name='input_file_external', filename='/tmp/tmport1qpph', content_type='application/octet-stream', original_filename='DJI_....npy'),
        'display': False,
        'save': True,
        #'accept': 'application/json'
    }
    predict(
        accept='application/json',
        **pred_args
    )