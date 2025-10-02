import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image


@dataclasses.dataclass(frozen=True)
class RawActionsInputs(upstream_transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        # Expect images and continuous actions; no language actions required.
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = parse_image(data["observation"]["exterior_image_1_left"])

        wrist_image = data["observation"].get("wrist_image_left", None)
        wrist_image = parse_image(wrist_image) if wrist_image is not None else np.zeros_like(base_image)
        wrist_image_mask = np.True_ if wrist_image is not None else np.False_

        assert self.model_type == ExtendedModelType.PI_COT
        names = ("base_0_rgb", "left_wrist_0_rgb")
        images = (base_image, wrist_image)
        image_masks = (np.True_, wrist_image_mask)

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Optional prompt if present (won't be used for action-only loss but harmless to pass through)
        if "prompt" in data:
            prompt_val = data["prompt"]
            if isinstance(prompt_val, bytes):
                inputs["prompt"] = prompt_val.decode("utf-8")
            else:
                inputs["prompt"] = prompt_val if isinstance(prompt_val, str) else np.asarray(prompt_val).item()

        return inputs


@dataclasses.dataclass(frozen=True)
class RawActionsOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return actions for evaluation/inference convenience
        actions = data.get("actions")
        return {"actions": actions}
