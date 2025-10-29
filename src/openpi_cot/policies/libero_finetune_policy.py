import dataclasses

import numpy as np
from openpi import transforms
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image


@dataclasses.dataclass(frozen=True)
class LiberoFinetuneInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: ExtendedModelType
    action_dim: int = 32
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0

    def __call__(self, data: dict) -> dict:
        assert self.model_type == ExtendedModelType.PI_COT
        assert "observation" in data
        assert IMAGE_KEYS[0] in data["observation"]
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
        images = [base_image]
        image_masks = [base_image_mask]
        needs_wrist_rotation = False

        for k in IMAGE_KEYS[1:]:
            if k in data["observation"]:
                wrist_image = parse_image(data["observation"][k])
                wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_

                # Rotate wrist image by 180 degrees for specific datasets
                if needs_wrist_rotation and wrist_image_mask:
                    wrist_image = np.rot90(wrist_image, k=2)
            else:
                wrist_image = np.zeros_like(base_image)
                wrist_image_mask = np.False_

            # Optional dropout: randomly mask out wrist image
            if self.wrist_image_dropout_prob > 0.0 and np.random.rand() < float(self.wrist_image_dropout_prob):
                wrist_image_mask = np.False_

            images.append(wrist_image)
            image_masks.append(wrist_image_mask)

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
        }

        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"
        if isinstance(prompt, bytes):  # training time
            prompt_str = prompt.decode("utf-8")
        elif isinstance(prompt, str):  # inference time
            prompt_str = prompt
        else:
            raise ValueError(f"Prompt is not a string or bytes: {prompt}")

        inputs["prompt"] = prompt_str

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoFinetuneOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
