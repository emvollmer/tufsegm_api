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
            'enum': utils.ls_dirs(config.MODELS_PATH) + utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude="perun_results"),
            'description': 'Model to be used for prediction. If only remote folders '
                           'are available, the selected one will be downloaded.'
        },
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "Input image file with .npy extension consisting of four channels for predictions.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    display = fields.Bool(
        metadata={
            'enum': [True, False],
            'description': 'Plot the resulting prediction to the console.'
        },
        required=False,
        load_default=False,
    )

    save = fields.Bool(
        metadata={
            'enum': [True, False],
            'description': 'Save the resulting prediction to a "predictions" subfolder in the model directory.'
        },
        required=False,
        load_default=True,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(list(responses.content_types)),
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        ordered = True

    model_type = fields.String(
        metadata={
            "enum": ['UNet'],
            "description": "Segmentation model type.",
        },
        required=False,
        load_default="UNet"
    )

    dataset_path = fields.String(
        metadata={
            "description": "Path to the dataset. If none is provided, "
                           "the dataset in the 'data' folder will be used or else "
                           "downloaded from Nextcloud.",
        },
        required=False,
    )

    test_size = fields.Float(
        metadata={
            "description": "Percentage of the dataset to be used for testing "
                           "and to calculate evaluation metrics with.",
        },
        required=False,
        load_default=0.2,
    )

    channels = fields.Integer(
        metadata={
            "enum": [3, 4],
            "description": "Process the data either in standard 4 channels (RGBT) "
                           "or as 3 channels (greyRGB+T+T).",
        },
        required=False,
        load_default=4,
    )

    processing = fields.String(
        metadata={
            "enum": ["basic", "vignetting", "retinex_unsharp"],
            "description": "Use original data (basic) or apply preprocessing filters "
                           "(vignetting removal, retinex and unsharp).",
        },
        required=False,
        load_default="basic",
    )

    img_size = fields.String(
        metadata={
            "enum": ["640x512", "320x256"],
            "description": "Use original image size (640x512) or downscale.",
        },
        required=False,
        load_default="640x512",
    )

    epochs = fields.Integer(
        metadata={
            "description": "Number of epochs to train the model.",
        },
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )

    batch_size = fields.Integer(
        metadata={'description': 'Batch size to load the data.'},
        required=False,
        load_default=8,
    )

    lr = fields.Float(
        metadata={'description': 'Learning rate.'},
        required=False,
        load_default=0.001,
    )

    seed = fields.Integer(
        metadata={'description': 'Global seed number for training.'},
        required=False,
        load_default=1000,
    )