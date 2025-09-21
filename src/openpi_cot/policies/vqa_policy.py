import dataclasses

import einops
import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import ExtendedModelType


def _parse_image(image) -> np.ndarray:
    if image is None:
        return None
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# TODO: during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class VQAInputs(upstream_transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = _parse_image(data["observation"]["exterior_image_1_left"])

        wrist_image = np.zeros_like(base_image)
        wrist_image_mask = np.False_

        assert self.model_type == ExtendedModelType.PI_COT
        names = ("base_0_rgb", "left_wrist_0_rgb")
        images = (
            base_image,
            wrist_image,
        )
        image_masks = (
            np.True_,
            wrist_image_mask,
        )

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        assert "prompt" in data, f"Prompt not found in data: {data}"
        # Normalize prompt to python str
        prompt_val = data["prompt"]
        if isinstance(prompt_val, bytes):
            prompt_str = prompt_val.decode("utf-8")
        elif isinstance(prompt_val, str):
            prompt_str = prompt_val
        else:
            prompt_item = np.asarray(prompt_val).item()
            prompt_str = (
                prompt_item.decode("utf-8") if isinstance(prompt_item, (bytes, np.bytes_)) else str(prompt_item)
            )

        inputs["prompt"] = prompt_str

        return inputs


@dataclasses.dataclass(frozen=True)
class VQAOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"reasoning": data.get("reasoning")}
