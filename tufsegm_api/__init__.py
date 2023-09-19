"""Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at tufsegm_api/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""
# TODO: add your imports here
import logging
from pathlib import Path
import tufsegm_api.config as cfg
import api.config as api_cfg
import api.utils as api_utils

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


# TODO: warm (Start Up)
# = HAVE TO MODIFY FOR YOUR NEEDS =
def warm(**kwargs):
    """Main/public method to start up the model
    """
    # if necessary, start the model
    pass


# TODO: predict
# = HAVE TO MODIFY FOR YOUR NEEDS =
def predict(**kwargs):
    """Main/public method to perform prediction
    """
    # if necessary, preprocess data
    
    # choose AI model, load weights
    
    # return results of prediction
    predict_result = {'result': 'not implemented'}
    logger.debug(f"[predict()]: {predict_result}")

    return predict_result

# TODO: train
# = HAVE TO MODIFY FOR YOUR NEEDS =
def train(**kwargs):
    """Main/public method to perform training
    """
    # if no data in local data folder, download it from Nextcloud
    if not all(folder in list(api_cfg.DATA_PATH.iterdir()) for folder in ["images", "annotations"]):
        logger.info(f"Data folder '{api_cfg.DATA_PATH}' empty, "
                    f"downloading data from '{api_cfg.REMOTE_DATA_PATH}'...")
        api_utils.copy_remote(frompath=api_cfg.REMOTE_DATA_PATH,
                              topath=api_cfg.DATA_PATH)

        logger.info("Extracting data from .zip format files...")
        for zip_path in Path(api_cfg.DATA_PATH).glob("**/*.zip"):
            limit_exceeded = unzip(zip_file=zip_path, 
                                   limit_gb=12)
            if limit_exceeded:
                break

    # prepare the dataset, e.g.
    # dtst.mkdata()
    
    # create model, e.g.
    # create_model()
    
    # train model
    # describe training steps

    # return training results
    train_result = {'result': 'not implemented'}
    logger.debug(f"[train()]: {train_result}")
    
    return train_result
