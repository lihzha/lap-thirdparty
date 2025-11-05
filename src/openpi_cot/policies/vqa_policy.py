import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.cot_policy import CoTInputs
from openpi_cot.policies.utils import parse_image


# during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class VQAInputs(CoTInputs):
    # Determines which model will be used.
    action_dim: int = 32
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = parse_image(data["observation"]["exterior_image_1_left"])
        if base_image is None:
            raise ValueError("Base image missing from observation")

        if np.all(base_image == 0):
            raise ValueError("Base image is all zeros")

        assert self.model_type == ExtendedModelType.PI_COT
        names = IMAGE_KEYS

        images = (
            base_image,
            np.zeros_like(base_image),
        )
        image_masks = (
            np.True_,
            np.False_,
        )
        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        prompt_val = data.get("prompt")
        prompt_str = None
        if prompt_val is not None:
            if isinstance(prompt_val, bytes):
                prompt_str = prompt_val.decode("utf-8")
            elif isinstance(prompt_val, str):
                prompt_str = prompt_val
            else:
                prompt_item = np.asarray(prompt_val).item()
                prompt_str = (
                    prompt_item.decode("utf-8") if isinstance(prompt_item, (bytes, np.bytes_)) else str(prompt_item)
                )
        if prompt_str is not None:
            inputs["prompt"] = prompt_str

        inputs["is_vqa_sample"] = True
        inputs["is_prediction_sample"] = False

        return inputs


@dataclasses.dataclass(frozen=True)
class VQAOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"reasoning": data.get("reasoning")}
