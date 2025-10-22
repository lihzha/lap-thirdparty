"""Tiger policy for converting data between dataset and model formats."""

import dataclasses

import numpy as np
from openpi import transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image


def make_tiger_example() -> dict:
    """Creates a random input example for the Tiger policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class TigerInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format.
    It is used for both training and inference.

    Tiger dataset format:
    - State: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
    - Actions: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
    - Images: base_image (third-person) and wrist_image
    """

    # Determines which model will be used.
    model_type: ExtendedModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C)

        base_image = parse_image(data["image"])
        wrist_image = parse_image(data["wrist_image"])

        # Pi0 models support three image inputs: one third-person view and two wrist views
        # Tiger has base and wrist, so we use zeros for the second wrist camera
        image = [base_image, wrist_image, np.zeros_like(wrist_image)]
        image_mask = [
            np.True_,
            np.True_,
            np.False_,
            # np.True_ if self.model_type == ExtendedModelType.PI0_FAST else np.False_,
        ]

        # Create inputs dict
        num_images_allowed = len(IMAGE_KEYS)
        inputs = {
            "state": data["state"],
            "image": {IMAGE_KEYS[i]: image[i] for i in range(num_images_allowed)},
            "image_mask": {IMAGE_KEYS[i]: image_mask[i] for i in range(num_images_allowed)},
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # # Pass the prompt (language instruction) to the model
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]
        # elif "task" in data:
        #     # Some datasets use "task" instead of "prompt"
        #     inputs["prompt"] = data["task"]
        inputs["prompt"] = "pick up the tiger"

        return inputs


@dataclasses.dataclass(frozen=True)
class TigerOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format.
    It is used for inference only.

    Tiger actions are 8-dimensional: arm_pos (3) + arm_quat (4) + gripper_pos (1)
    """

    def __call__(self, data: dict) -> dict:
        # Tiger uses 8-dimensional actions, so we only return the first 8 dimensions
        # (in case the model outputs padding)
        actions = data["actions"]
        if actions.shape[-1] > 8:
            actions = actions[..., :8]

        return {"actions": actions}
