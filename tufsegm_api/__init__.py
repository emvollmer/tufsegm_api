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
import tufsegm_api.api.config as api_cfg

from tufsegm_api.utils import copy_remote, unzip, setup, run_bash_subprocess
#from tufsegm_api.api.utils import copy_remote

from ThermUrbanFeatSegm.scripts.segm_models.infer_UNet import main as predict_func

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


def predict(**kwargs):
    """Main/public method to perform prediction
    """
    # If model_name is a remote directory ('rshare'), download it to the models directory
    if 'rshare:' in kwargs['model_name']:

        # check if folder of same name exists locally, don't copy if that's the case
        remote_folder_name = Path(kwargs['model_name']).name
        if remote_folder_name in os.listdir(api_cfg.MODELS_PATH):
            print(f"Model folder '{kwargs['model_name']}' contains 'rshare:' but exists locally. "
                  f"Using local folder instead...")
        else:
            print(f"Model folder '{kwargs['model_name']}' contains 'rshare:'. "
                  f"Downloading from '{api_cfg.REMOTE_MODELS_PATH}'...")
            copy_remote(frompath=Path(kwargs['model_name']),
                        topath=Path(api_cfg.MODELS_PATH, remote_folder_name))

        # redefine the model name as only the folder itself
        kwargs['model_name'] = remote_folder_name

    # define the model path
    model_path = Path(api_cfg.MODELS_PATH, kwargs['model_name'])
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model folder '{model_path}' does not exist!")

    print(f"Predicting with the model at: {model_path}")

    # If input_file is a remote ('rshare:'), download it to the data folder
    if 'rshare:' in kwargs['input_file']:

        # check if file of same name exists locally, don't copy if that's the case
        remote_file_name = Path(kwargs['input_file']).relative_to(api_cfg.REMOTE_PATH)
        if Path(api_cfg.DATA_PATH, remote_file_name).is_file():
            print(f"Input file '{kwargs['input_file']}' contains 'rshare:' but exists locally. "
                  f"Using local file instead...")
        else:
            print(f"Input file '{kwargs['input_file']}' contains 'rshare:'. "
                  f"Downloading from '{api_cfg.REMOTE_PATH}'...")
            copy_remote(frompath=Path(kwargs['input_file']),
                        topath=Path(api_cfg.DATA_PATH, remote_file_name).parent)

        # redefine the input_file as only the file itself
        kwargs['input_file'] = remote_file_name

    # define the input file path
    input_file_path = Path(api_cfg.DATA_PATH, kwargs['input_file'])
    
    print(f"Predicting on image: {input_file_path}")
    logger.info(f"Predicting on image: {input_file_path}")  # this does nothing!

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
                        if Path(kwargs['input_file']).name == f.name]
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
    data_path = Path(kwargs['dataset_path'] or Path(api_cfg.DATA_PATH))
    print(f"Training with the user defined parameters:\n{locals()}")

    # get file and folder names in data_path (non-recursive)
    data_path_entries = os.listdir(str(data_path))

    # if no data in local data folder, download it from Nextcloud
    if not all(e in data_path_entries for e in ["images", "annotations"]):
        print(f"Data folder '{data_path}' is empty, "
              f"downloading data from '{api_cfg.REMOTE_DATA_PATH}'...")
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

    # train model
    logger.info("Starting training...")
    kwargs['cfg_options'] = {'epochs': kwargs['epochs'],
                             'batch_size': kwargs['batch_size'],
                             'lr': kwargs['lr'],
                             'seed': kwargs['seed'],
                             'SIZE_W': kwargs['img_size'].split("x")[0],
                             'SIZE_H': kwargs['img_size'].split("x")[1],
                            }
    cfg_options_str = ' '.join([f"{key}={value}" for key, value in kwargs['cfg_options'].items()])
    
    train_cmd = ["/bin/bash", str(Path(cfg.SUBMODULE_PATH, 'scripts', 'segm_models', 'train.sh')),
                 "-dst", str(api_cfg.MODELS_PATH),
                 "--channels", str(kwargs['channels']),
                 "--processing", str(kwargs['processing']),
                 "--cfg-options", cfg_options_str, # "--default-log"
                 cfg.VERBOSITY]
    current_time = datetime.now()
    print(f"\nRunning training with arguments:\n{train_cmd}\n"
          f"...at current time: {current_time.strftime('%Y-%m-%d_%H-%M-%S')}\n")
    run_bash_subprocess(train_cmd)
    print(f"Training and evaluation completed.")

    # return training results - check existance of evaluation file in model folder and load from there
    try:
        model_path = sorted(api_cfg.MODELS_PATH.glob("[!.]*"))[-1]
        model_time = datetime.strptime(model_path.name, "%Y-%m-%d_%H-%M-%S")
        if current_time - model_time <= timedelta(minutes=1):
            eval_file = Path(model_path, "eval.json")
            if eval_file.is_file():
                with open(eval_file, "r") as f:
                    train_result = json.load(f)
            else:
                train_result = {'result': 'error during training or evaluation, no model scores saved.'}
        else:
            train_result = {'result': f'error during training, no model folder similar to '
                                      f'{current_time.strftime("%Y-%m-%d_%H-%M-%S")} exists.'}
    except IndexError:
        train_result = {'result': f'error during training, no model folders exist at {api_cfg.MODELS_PATH}'}

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
        'img_size': "320x256", # "640x512"
        'epochs': 2,
        'batch_size': 4,    # 8
        'lr': 0.001,
        'seed': 42
    }
    train(**ex_args)