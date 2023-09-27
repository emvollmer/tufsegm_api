"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging
from pathlib import Path

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
            "datasets_local": utils.ls_dirs(Path(config.DATA_PATH, "raw")),
            "datasets_remote": utils.ls_remote_dirs(suffix=".zip", exclude="additional_data"),
            "models_local": utils.ls_dirs(config.MODELS_PATH),
            "models_remote": utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude='perun_results'),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


def warm():
    """Function to run preparation phase before anything else can start.

    Raises:
        RuntimeError: Unexpected errors aim to stop model loading.
    """
    try:  # Call your AI model warm() method
        logger.info("Warming up the model.api...")
        aimodel.warm()
    except Exception as err:
        raise RuntimeError(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(model_name, input_file, accept='application/json', **options):
    """Performs model prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        input_file -- File with data to perform predictions from model.
        accept -- Response parser type, default is json.
        **options -- keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    try:  # Call your AI model predict() method
        logger.info(f"Using model '{model_name}' for predictions")
        logger.debug("Loading data from input_file: %s", input_file.filename)
        logger.debug("Predict with options: %s", options)
        result = aimodel.predict(input_file.filename, model_name, **options)
        logger.debug("Predict result: %s", result)
        logger.info("Returning content_type for: %s", accept)
        return responses.content_types[accept](result, **options)
    except Exception as err:
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
        logger.info(f"Retraining a '{options['model_type']}' model")
        logger.debug(f"Training with options: {options}")
        result = aimodel.train(**options)
        logger.debug(f"Training result: {result}")
        return result
    except Exception as err:
        raise HTTPException(reason=err) from err


if __name__ == "__main__":
    metadata = get_metadata()
    print(metadata)

    ex_args = {
        'model_type': 'UNet',
        'dataset_path': None,
        'save_for_viewing': False,
        'test_size': 0.2,
        'channels': 4,
        'processing': "basic",
        'img_size': "640x512",
        'epochs': 1,
        'batch_size': 8,
        'lr': 0.0001,
        'seed': 42
    }
    train(
        **ex_args
    )