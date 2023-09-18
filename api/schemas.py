"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""
from marshmallow import Schema
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
class PredArgsSchema(Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model_name = fields.String(
        metadata={
            "enum": utils.ls_dirs(config.MODELS_PATH) + utils.ls_remote_dirs(suffix=config.MODEL_SUFFIX, exclude="perun_results"),
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
        required=False,
        load_default=False,
        metadata={
            'enum': [True, False],
            'description': 'Plot the resulting prediction to the console.'
        }
    )

    save = fields.Bool(
        required=False,
        load_default=True,
        metadata={
            'enum': [True, False],
            'description': 'Save the resulting prediction to a "predictions" subfolder in the model directory.'
        }
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(list(responses.content_types)),
    )


# EXAMPLE of Training Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =
class TrainArgsSchema(Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

#    model_name = ModelName(
#        metadata={
#            "description": "String/Path identification for models.",
#        },
#        required=True,
#        load_default=None
#    )

    dataset = fields.String(
        metadata={
            "description": "Path to the training dataset. If none is provided, "
                           "the dataset in the 'data' folder will be used or else "
                           "downloaded from Nextcloud.",
        },
        required=False,
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
    # SIZE_W = img_size.split("x")[0]
    # SIZE_H = img_size.split("x")[1]

    epochs = fields.Integer(
        metadata={
            "description": "Number of epochs to train the model.",
        },
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )

    batch_size = fields.Integer(
        required=False,
        load_default=8,
        metadata={'description': 'Batch size to load the data.'}
    )

    lr = fields.Float(
        required=False,
        load_default=0.001,
        metadata={'description': 'Learning rate.'}
    )

    seed = fields.Integer(
        required=False,
        load_default=1000,
        metadata={'description': 'Global seed number for training.'}
    )