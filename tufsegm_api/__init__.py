"""Package to create dataset, build training and prediction pipelines.

This file defines or imports all the functions needed to operate the
methods defined at tufsegm_api/api.py.
```
"""
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys
import os

import tufsegm_api.config as cfg

from tufsegm_api import utils

from TUFSeg.scripts.segm_models.infer_UNet import main as predict_func

logger = logging.getLogger(__name__)
utils.configure_api_logging(logger, cfg.LOG_LEVEL)


class ResultError(Exception):
    """Raised when disk space is exceeded."""
    pass


def predict(**kwargs):
    """Main/public method to perform prediction
    --- WITHOUT COPYING DATA OR MODELS (WORKING IN NEXTCLOUD IF THAT'S WHERE THE DATA/MODEL IS)
    """
    model_path = Path(kwargs['model_dir'])
    logger.debug(f"Predicting with model: {model_path}")

    # if input_file is a remote file ('/storage/'), work remotely
    if cfg.REMOTE_PATH in kwargs['input_file']:
        input_file_path = Path(kwargs['input_file'])
    else:
        input_file_path = Path(cfg.DATA_PATH, kwargs['input_file'])
    
    logger.debug(f"Predicting on image: {input_file_path}")

    # prediction
    predict_func(
        model_dir=model_path,
        img_path=input_file_path,
        mask_path=None,
        display=kwargs['display'],
        save=True,
        log_level=cfg.LOG_LEVEL
    )
    
    # return results of prediction
    if Path(model_path, 'predictions').is_dir():
        pred_results = [f for f in Path(model_path, 'predictions').rglob("*.npy") 
                        if Path(input_file_path).name == f.name]
        if pred_results:
            predict_result = {'result': f'predicted segmentation results saved to {pred_results[0]}'}
        else:
            predict_result = {'result': f'error occured. No matching prediction results '
                                        f'for file "{kwargs["input_file"]}" in "{model_path}".'}
    else:
        predict_result = {'result': f'error occured. No prediction folder created at "{model_path}".'}
    logger.debug(f"[predict()]: {predict_result}")

    return predict_result


def train(**kwargs):
    """Main/public method to perform training
    """
    data_path = Path(kwargs['dataset_path'] or Path(cfg.DATA_PATH))
    logger.debug(f"Training on data from: {data_path}")
    
    # get data - check files in local data_path, if no setup, check NextCloud
    data_entries = set(os.listdir(data_path))
    required_entries = {"images", "annotations"}

    if not data_entries >= required_entries:

        if set(os.listdir(cfg.REMOTE_DATA_PATH)) >= required_entries:
            logger.info(f"Data folder '{data_path}' does not contain images & annotations, "
                        f"downloading data from '{cfg.REMOTE_DATA_PATH}'...")
            utils.copy_remote(frompath=Path(cfg.REMOTE_DATA_PATH), topath=Path(data_path))

        else:
            raise FileNotFoundError(f"Remote data folder '{cfg.REMOTE_DATA_PATH}' "
                                    f"does not contain required data.")

    # if zipped data in local data folder, unzip it
    zip_paths = list(data_path.rglob("*.zip"))
    if zip_paths:
        logger.info(f"Extracting data from {len(zip_paths)} .zip files...")
        utils.unzip(zip_paths)

    # prepare data if not yet done
    if not data_entries >= {"masks", "train.txt", "test.txt"}:
        utils.setup(
            data_path=data_path, 
            test_size=kwargs['test_size'],
            save_for_view=kwargs['save_for_viewing']
        )

    # train model
    kwargs['cfg_options'] = {'backbone': kwargs['backbone'],
                             'encoded_weights': kwargs['weights'],
                             'epochs': kwargs['epochs'],
                             'batch_size': kwargs['batch_size'],
                             'lr': kwargs['lr'],
                             'seed': kwargs['seed'],
                             'SIZE_W': kwargs['img_size'].split("x")[0],
                             'SIZE_H': kwargs['img_size'].split("x")[1],
                            }
    cfg_options_str = ' '.join([f"{key}={value}" for key, value in kwargs['cfg_options'].items()])
    
    train_cmd = ["/bin/bash", str(Path(cfg.SUBMODULE_PATH, 'scripts', 'segm_models', 'train.sh')),
                 "-src", str(data_path),
                 "-dst", str(cfg.MODELS_PATH),
                 "--channels", str(kwargs['channels']),
                 "--processing", str(kwargs['processing']),
                 "--cfg-options", cfg_options_str,
                 cfg.VERBOSITY]

    creation_time = datetime.now()

    logger.info(f"Training with arguments:\n{train_cmd}")
    utils.run_bash_subprocess(train_cmd)
    logger.info(f"Training and evaluation completed.")

    # log model (if desired) and return training results
    try:
        model_path = sorted(cfg.MODELS_PATH.glob("[!.]*"))[-1]

        # track model with mlflow if user provided information
        if kwargs['mlflow_username']:
            logger.info("Beginning MLFLow experiment logging...")
            utils.mlflow_logging(model_root=Path(model_path))
            logger.info("Completed MLFLow experiment logging.")

        model_time = datetime.strptime(model_path.name, "%Y-%m-%d_%H-%M-%S")
        if creation_time - model_time <= timedelta(minutes=1):
            eval_file = Path(model_path, "eval.json")
            with open(eval_file, "r") as f:
                train_result = json.load(f)
                return train_result
        else:
            raise ResultError(f'Error during training, no model folder similar to '
                              f'{creation_time.strftime("%Y-%m-%d_%H-%M-%S")} exists.')
    except IndexError as e:
        raise ResultError(f'Error during training, no model folders exist at {cfg.MODELS_PATH}. ', e)

    except FileNotFoundError as e:
        raise ResultError(f'Error during training or evaluation, no model scores saved. ', e)


if __name__ == '__main__':
    ex_args = {
        # 'model_type': 'UNet',
        'dataset_path': None,
        'save_for_viewing': False,
        'test_size': 0.2,
        'channels': 4,
        'processing': "basic",
        'img_size': "320x256",
        'epochs': 1,
        'batch_size': 9,
        'lr': 0.001,
        'seed': 42
    }
    train(**ex_args)