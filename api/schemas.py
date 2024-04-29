"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods.
"""
import marshmallow
from pathlib import Path
from webargs import ValidationError, fields, validate
import os

from tufsegm_api.api import config, responses, utils


class ModelName(fields.String):
    """Field that takes a string and validates against current available
    models at config.MODELS_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.MODELS_PATH):
            raise ValidationError(f"Checkpoint `{value}` not found.")
        return str(config.MODELS_PATH / value)


class Dataset(fields.String):
    """Field that takes a string and validates against current available
    data files at config.DATA_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.DATA_PATH):
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / value)


class NpyFile(fields.String):
    """Field that takes a file path as a string and makes sure it exists
    either locally in repository directory or remotely on Nextcloud,
    whilst also ensuring it's a numpy file.
    """
    def _deserialize(self, value, attr, data, **kwargs):
        if Path(value).is_file():
            if value.endswith(".npy"):
                return value
            raise ValidationError(f"Provided file path `{value}` is not a numpy file.")
        else:
            raise ValidationError(f"Provided file path `{value}` does not exist.")


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    model_dir = fields.String(
        metadata={
            "description": "Model to be used for prediction. If a remote folder (/storage/)"
                           "is selected, prediction results will be saved there."
        },
        validate=validate.OneOf(
            utils.get_dirs(config.MODELS_PATH, entries={config.MODEL_TYPE + config.MODEL_SUFFIX}) + 
            utils.get_dirs(config.REMOTE_PATH, entries={config.MODEL_TYPE + config.MODEL_SUFFIX})
        ),
        required=True,
    )

    input_file = NpyFile(
        metadata={
            "description": f"Insert a .npy path of a four channels file to infer on. Provide this in either one of two ways:"
                           f"\n- local path (in 'data/')\tf.e.: 'images/KA_01/DJI_0_0001_R.npy'"
                           f"\n- remote path on Nextcloud\tf.e.: '/storage/tufsegm/.../KA_01/DJI_0_0001_R.npy'",
        },
        required=True,
    )

    display = fields.Boolean(
        metadata={
            "description": "Plot the resulting prediction to the console."
        },
        load_default=False,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        validate=validate.OneOf(list(responses.content_types)),
        load_default='application/json',
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    mlflow_username = fields.String(
        metadata={
            "description": "MLFlow username for experiment tracking "
                           "(password will be requested via terminal). "
                           "Leave blank if you don't want to use MLFlow.",
        },
        load_default=None,
    )

    # model_type = fields.String(
    #     metadata={
    #         "description": "Segmentation model type.",
    #     },
    #     validate=validate.OneOf(['UNet']),
    #     load_default="UNet",
    # )

    backbone = fields.String(
        metadata={
            "description": "Model backbone to use. Default is 'resnet152'.",
        },
        validate=validate.OneOf(['resnet152', 'mobilenetv2']),
        load_default="resnet152",
    )

    weights = fields.String(
        metadata={
            "description": "Encoder weights to load (pretrained or not). Default is 'imagenet'.",
        },
        validate=validate.OneOf(['imagenet', 'None']),
        load_default="imagenet",
    )

    dataset_path = fields.String(
        metadata={
            "description": "Path to the dataset. If none is provided, "
                           "the dataset in the 'data' folder will be used or else "
                           "downloaded from Nextcloud if local 'data' is empty.",
        },
        validate=validate.OneOf(
            utils.get_dirs(config.DATA_PATH, entries={'images', 'annotations'}) + 
            utils.get_dirs(config.REMOTE_PATH, entries={'images', 'annotations'})
        ),
        required=False,
        load_default=None
    )

    save_for_viewing = fields.Boolean(
        metadata={
            "description": "Save additional data such as segmentation masks "
                           "in .png for user viewing. ATTENTION: This will "
                           "fill up additional space!"
        },
        load_default=False,
    )

    test_size = fields.Float(
        metadata={
            "description": "Percentage of the dataset to be used for testing "
                           "and to calculate evaluation metrics with.",
        },
        load_default=0.2,
    )

    channels = fields.Integer(
        metadata={
            "description": "Process the data either in standard 4 channels (RGBT) "
                           "or as 3 channels (greyRGB+T+T).",
        },
        validate=validate.OneOf([3, 4]),
        load_default=4,
    )

    processing = fields.String(
        metadata={
            "description": "Use original data (basic) or apply preprocessing filters "
                           "(vignetting removal, retinex and unsharp).",
        },
        validate=validate.OneOf(["basic", "vignetting", "retinex_unsharp"]),
        load_default="basic",
    )

    img_size = fields.String(
        metadata={
            "description": "Use original image size (640x512) or downscale. "
                           "ATTENTION: The original size requires a lot of RAM memory "
                           "(> 25000) otherwise training will fail."
        },
        validate=validate.OneOf(["640x512", "320x256", "160x128"]),
        load_default="320x256",
    )

    epochs = fields.Integer(
        metadata={
            "description": "Number of epochs to train the model.",
        },
        validate=validate.Range(min=1), # minimum value has to be 1
        load_default=1,
    )

    batch_size = fields.Integer(
        metadata={'description': 'Batch size to load the data.'},
        validate=validate.Range(min=1), # minimum value has to be 1
        load_default=8,
    )

    lr = fields.Float(
        metadata={'description': 'Learning rate.'},
        load_default=0.001,
    )

    seed = fields.Integer(
        metadata={'description': 'Global seed number for training.'},
        validate=validate.Range(min=1), # minimum value has to be 1
        load_default=1000,
    )