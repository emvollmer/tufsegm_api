"""Package to create dataset, build training and prediction pipelines.

This file defines or imports all the functions needed to operate the
methods defined at tufsegm_api/api.py.
```
"""
import logging
from pathlib import Path
import sys
import os

import tufsegm_api.config as cfg
import tufsegm_api.api.config as api_cfg

from tufsegm_api.utils import unzip, setup, run_bash_subprocess
from tufsegm_api.api.utils import copy_remote

from ThermUrbanFeatSegm.scripts.segm_models.infer_UNet import main as predict_func

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


def train(**kwargs):
    """Main/public method to perform training
    """
    data_path = Path(kwargs['dataset_path'] or Path(api_cfg.DATA_PATH, "raw"))
    print(f"Training with the user defined parameters:\n{locals()}")

    # get file and folder names in data_path (non-recursive)
    data_path_entries = os.listdir(str(data_path))

    # if no data in local data folder, download it from Nextcloud
    if not all(e in data_path_entries for e in ["images", "annotations"]):
        print(f"Data folder '{data_path}' is empty, "
              f"downloading data from 'rshare:{api_cfg.REMOTE_DATA_PATH}'...")
        copy_remote(frompath=api_cfg.REMOTE_DATA_PATH,
                    topath=data_path)

    # if zipped data in local data folder, unzip it
    zip_paths = list(data_path.rglob("*.zip"))
    print(f"Extracting data from {len(zip_paths)} .zip files...")
    for zip_path in zip_paths:
        unzip(zip_file=zip_path)
    
        print("Cleaning up zip file...")
        zip_path.unlink()

    # prepare data
    if not all(e in data_path_entries for e in ["masks", "train.txt", "test.txt"]):
        setup(
            data_path=data_path, 
            test_size=kwargs['test_size'],
            save_for_view=kwargs['save_for_viewing']
        )

    # # train model
    # logger.info("Starting training...")
    # kwargs['cfg_options'] = {'epochs': kwargs['epochs'],
    #                          'batch_size': kwargs['batch'],
    #                          'lr': kwargs['lr'],
    #                          'seed': kwargs['seed'],
    #                          'SIZE_W': kwargs['img_size'].split("x")[0],
    #                          'SIZE_H': kwargs['img_size'].split("x")[1],
    #                         }
    # cfg_options_str = ' '.join([f"{key}={value}" for key, value in kwargs['cfg_options'].items()])
    
    # train_cmd = ["/bin/bash", str(Path(cfg.SUBMODULE_PATH, 'scripts', 'segm_models', 'train.sh')),
    #              "-dst", str(api_cfg.MODELS_PATH),
    #              "--channels", str(kwargs['channels']),
    #              "--processing", str(kwargs['processing']),
    #              "--cfg-options", cfg_options_str,
    #              cfg.VERBOSITY]
    # run_bash_subprocess(train_cmd)

    # return training results
    train_result = {'result': 'not implemented'}
    logger.debug(f"[train()]: {train_result}")
    
    return train_result


if __name__ == '__main__':
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
    train(**ex_args)