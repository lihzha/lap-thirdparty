"""Policy transforms for planning dataset.

This module provides input and output transforms for the planning dataset,
which uses a fixed language instruction and loads data from HDF5 via TFDS.
"""

import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image


@dataclasses.dataclass(frozen=True)
class PlanningInputs(upstream_transforms.DataTransformFn):
    """Transform planning dataset to model inputs.

    The planning dataset has:
    - Images: base_image (84x84x3), wrist_image (84x84x3) from HDF5
    - State: 8D [arm_pos(3), arm_quat(4), gripper_pos(1)]
    - Actions: 10D action vector
    - Language: Fixed instruction per demo
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def _prepare_inputs(self, data: dict) -> dict:
        """Prepare inputs from planning dataset."""
        assert self.model_type == ExtendedModelType.PI_COT
        assert "observation" in data

        # Parse base image
        assert IMAGE_KEYS[0] in data["observation"], f"Missing {IMAGE_KEYS[0]} in observation"
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_

        # Parse wrist image
        assert IMAGE_KEYS[1] in data["observation"], f"Missing {IMAGE_KEYS[1]} in observation"
        wrist_image = parse_image(data["observation"][IMAGE_KEYS[1]])
        wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_

        # For planning dataset, we don't have a right wrist camera
        # Use zeros as placeholder
        right_wrist_image = np.zeros_like(base_image)
        right_wrist_image_mask = np.False_

        images = [base_image, wrist_image, right_wrist_image]
        image_masks = [base_image_mask, wrist_image_mask, right_wrist_image_mask]

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
        }

        # Get language instruction
        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"
        if isinstance(prompt, bytes):  # training time
            prompt_str = prompt.decode("utf-8")
        elif isinstance(prompt, str):  # inference time
            prompt_str = prompt
        else:
            raise ValueError(f"Prompt is not a string or bytes: {prompt}")

        inputs["prompt"] = prompt_str

        # Add dataset name
        dataset_name = data.get("dataset_name", b"planning_dataset")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")
        inputs["dataset_name"] = dataset_name

        # Add actions if available
        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        # Planning dataset doesn't use language actions for reasoning
        # Set sample_mask to True (always use these samples)
        inputs["sample_mask"] = True

        return inputs

    def __call__(self, data: dict) -> dict:
        """Transform planning dataset sample to model input format."""
        return self._prepare_inputs(data)


@dataclasses.dataclass(frozen=True)
class PlanningOutputs(upstream_transforms.DataTransformFn):
    """Transform model outputs for planning dataset.

    For planning dataset, we primarily use the action predictions directly.
    Reasoning/language outputs are optional for debugging.
    """

    def __call__(self, data: dict) -> dict:
        """Transform model outputs.

        Args:
            data: Dict containing model outputs, typically:
                - "actions": predicted actions
                - "reasoning": optional reasoning text

        Returns:
            Dict with processed outputs
        """
        # For planning dataset, we primarily care about action predictions
        # Return as-is for now, can add post-processing if needed
        output = {}

        if "actions" in data:
            output["actions"] = data["actions"]

        if "reasoning" in data:
            output["reasoning"] = data["reasoning"]

        return output
