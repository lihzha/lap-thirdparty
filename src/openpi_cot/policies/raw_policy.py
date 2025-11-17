import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.policies.utils import parse_image

from openpi_cot.models.adapters.model_adapter import ExtendedModelType


@dataclasses.dataclass(frozen=True)
class RawActionInputs(upstream_transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        assert "observation" in data
        assert IMAGE_KEYS[0] in data["observation"]
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")


        images = []
        image_masks = []

        # Training/validation: base image without rotation, wrist images with rotation if needed
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
        images.append(base_image)
        image_masks.append(base_image_mask)

        # Process wrist images (may need rotation)
        for k in IMAGE_KEYS[1:]:
            if k in data["observation"]:
                wrist_image = parse_image(data["observation"][k])
                wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_

            else:
                wrist_image = np.zeros_like(base_image)
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

        if "dataset_name" in data and "r1_lite" in data["dataset_name"].decode():
            prompt_str = prompt_str.split("@")[-1]

        inputs["prompt"] = prompt_str
        if "dataset_name" in data:
            inputs["dataset_name"] = data["dataset_name"].decode()


        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        inputs["is_prediction_sample"] = False
        inputs["is_vqa_sample"] = False
        inputs["sample_mask"] = True
        return inputs

@dataclasses.dataclass(frozen=True)
class RawActionOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
