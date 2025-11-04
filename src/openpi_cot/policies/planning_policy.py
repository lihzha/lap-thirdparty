"""Policy transforms for planning dataset.

This module provides input and output transforms for the planning dataset,
which uses a fixed language instruction and loads data from HDF5 via TFDS.
"""

import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms
from scipy.spatial.transform import Rotation as R

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
            "image": dict(zip(IMAGE_KEYS, images[: len(IMAGE_KEYS)], strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks[: len(IMAGE_KEYS)], strict=True)),
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
        inputs["is_prediction_sample"] = False
        inputs["is_vqa_sample"] = False

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
            actions = data["actions"]
            output["actions"] = np.concatenate(
                [actions[..., :6], rot6_to_quat(actions[..., 6:12]), actions[..., 12:13]], axis=-1
            )

        if "reasoning" in data:
            output["reasoning"] = data["reasoning"]

        return output


def rot6_to_quat(r6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representations into quaternions (w, x, y, z).
    Supports both single input (6,) and batched input (B, 6).

    Args:
        r6 (np.ndarray): Array of shape (..., 6), representing
            [r11, r12, r13, r21, r22, r23].
            The missing third row is reconstructed as a cross product
            to ensure a right-handed orthonormal frame.

    Returns:
        np.ndarray: Quaternion(s) in (w, x, y, z) format.
            Shape: (..., 4)
    """
    r6 = np.asarray(r6)
    assert r6.shape[-1] == 6, "Input must have 6 elements in the last dimension"

    # Split into r1, r2
    r1 = r6[..., 0:3]
    r2 = r6[..., 3:6]
    r3 = np.cross(r1, r2, axis=-1)
    R_mat = np.stack([r1, r2, r3], axis=-2)  # (..., 3, 3)

    # Orthonormalize using SVD
    U, _, Vt = np.linalg.svd(R_mat)
    R_mat = np.matmul(U, Vt)

    # Convert to quaternions (SciPy returns [x, y, z, w])
    quat_xyzw = R.from_matrix(R_mat.reshape(-1, 3, 3)).as_quat()
    quat_xyzw = quat_xyzw.reshape(R_mat.shape[:-2] + (4,))

    # Reorder to (w, x, y, z)
    # quat_wxyz = np.concatenate(
    #     [quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1
    # )

    # Normalize
    quat_wxyz = quat_xyzw
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)

    return quat_wxyz
