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
        # inputs["prompt"] = "pick up the tiger"
        inputs["prompt"] = data["prompt"]

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
        # if actions.shape[-1] > 8:
        # actions = actions[..., :8]
        actions = np.concatenate([actions[:3], rot6_to_quat(actions[3:9]), actions[9:10]], axis=-1)

        return {"actions": actions}


def rot6_to_quat(r6: np.ndarray) -> np.ndarray:
    """
    Convert the first 6 elements of a rotation matrix (r11..r23)
    into a quaternion (w, x, y, z).

    Args:
        r6: np.ndarray of shape (6,), representing
            [r11, r12, r13, r21, r22, r23].
            The missing third row will be reconstructed assuming a right-handed rotation matrix.

    Returns:
        np.ndarray: Quaternion (w, x, y, z)
    """
    assert r6.shape[-1] == 6, "Input must have 6 elements"

    # reconstruct first 2 rows
    r1 = np.array([r6[0], r6[1], r6[2]])
    r2 = np.array([r6[3], r6[4], r6[5]])
    # infer the third row as cross product to ensure orthonormality
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=0)

    # ensure R is normalized
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    # convert rotation matrix to quaternion (w, x, y, z)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q
