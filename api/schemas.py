"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
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
        if value in utils.ls_files(Path(config.DATA_PATH), "**/*.npy"):
            return value
        elif value.startswith("rshare:"):
            if value in utils.ls_remote_files(".npy"):
                return value
            else:
                raise ValidationError(f"Provided file path `{value}` does not exist in NextCloud.")
        else:
            raise ValidationError(f"Provided file path `{value}` does not exist locally.")


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    model_folder = fields.String(
        metadata={
            "description": "Model to be used for prediction. If a remote NextCloud folder (rshare:)"
                           "is selected, it will be downloaded and all outputs will be saved to it.\n"
                           "This list doesn't display just recently trained models to be selected. "
                           "Enter the model folder path in the next field if you want the newest.\n"
                           "MUTUALLY EXCLUSIVE WITH model_folder_new"
        },
        validate=validate.OneOf(
            utils.ls_dirs(config.MODELS_PATH) + 
            utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude="perun_results")
        ),
        required=False,
    )

    model_folder_new = fields.String(
        metadata={
            "description": "Model to be used for prediction. Enter the model folder path here if the "
                           "model you want to use is not in the list above (as it's too new). \n"
                           "MUTUALLY EXCLUSIVE WITH model_folder"
        },
        required=False,
    )

    input_file = NpyFile(
        metadata={
            "description": f"Insert a .npy path of a four channels file to infer on. Provide this in either one of two ways:"
                           f"\n- local path (in 'data/')\tf.e.: 'images/KA_01/DJI_0_0001_R.npy'"
                           f"\n- remote path on Nextcloud\tf.e.: 'rshare:/tufsegm/.../KA_01/DJI_0_0001_R.npy'",
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

    # @marshmallow.post_load
    # def validate_required_fields(self, data, **kwargs):
    #     if 'model_folder' != None and 'model_folder_new' != None:
    #         raise marshmallow.ValidationError(
    #             'Only one model can be selected for performing inference - either from the '
    #             'offered list or entered folder path.'
    #         )
    #     if 'model_folder' == None and 'model_folder_new' == None:
    #         raise marshmallow.ValidationError(
    #             'No model folder path for inference was selected / provided! Select either '
    #             'an option from the "model_folder" list or enter a value into the "model_folder_new" field.'
    #         )
    #     return data


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    mlflow_username = fields.String(
        metadata={
            "description": "MLFlow username to be used for experiment tracking. Leave blank if you don't want to use MLFlow.",
        },
        load_default=None,
    )

    model_type = fields.String(
        metadata={
            "description": "Segmentation model type.",
        },
        validate=validate.OneOf(['UNet']),
        load_default="UNet",
    )

    dataset_path = fields.String(
        metadata={
            "description": "Path to the dataset. If none is provided, "
                           "the dataset in the 'data' folder will be used or else "
                           "downloaded from Nextcloud.",
        },
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