"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""
import marshmallow
from webargs import ValidationError, fields, validate

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
        if value not in utils.ls_dir(config.DATA_PATH / "processed"):
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / "processed" / value)


# EXAMPLE of Prediction Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =
class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    model_name = fields.String(
        metadata={
            "description": "Model to be used for prediction. If a remote folder (rshare:)"
                           "is selected, it will automatically be downloaded from Nextcloud."
        },
        validate=validate.OneOf(
            utils.ls_dirs(config.MODELS_PATH) + 
            utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude="perun_results")
        ),
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "Input image file with .npy extension consisting of four channels for predictions.",
            "type": "file",
            "location": "form",
            "accept": ".npy",
        },
        required=True,
    )

    display = fields.Boolean(
        metadata={
            "description": "Plot the resulting prediction to the console."
        },
        load_default=False,
    )

    save = fields.Boolean(
        metadata={
            "description": "Save the resulting prediction to a 'predictions' subfolder" 
                           "in the model directory."
        },
        load_default=True,
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