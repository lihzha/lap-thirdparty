import dataclasses

import einops
import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.lang_action_util import sum_language_actions
from openpi_cot.dataloader.lang_action_util import summarize_numeric_actions
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


def _to_str_list(x):
    if isinstance(x, (list, tuple)):
        seq = x
    elif isinstance(x, np.ndarray):
        seq = x.tolist()
    else:
        return None
    out = []
    for item in seq:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    sum_decimal: str = "0f"
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = _parse_image(data["observation"]["exterior_image_1_left"])
        if "wrist_image_left" in data["observation"]:
            wrist_image = _parse_image(data["observation"]["wrist_image_left"])
            if wrist_image is None:
                wrist_image = np.zeros_like(base_image)
                wrist_image_mask = np.False_
            else:
                wrist_image_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_image_mask = np.False_

        # Optional dropout: randomly mask out wrist image
        if self.wrist_image_dropout_prob > 0.0:
            if np.random.rand() < float(self.wrist_image_dropout_prob):
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

        if "prompt" in data:
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

        if "language_actions" in data:
            breakpoint()
            if isinstance(data["language_actions"][0], np.ndarray):
                summed = summarize_numeric_actions(data["language_actions"], self.sum_decimal)
                inputs["language_actions"] = summed
            else:
                assert isinstance(data["language_actions"][0], bytes)
                seq = _to_str_list(data["language_actions"])
                if seq is not None:
                    summed = sum_language_actions(seq, self.sum_decimal)
                    if summed is not None and len(summed) > 0:
                        inputs["language_actions"] = summed
                else:
                    # Scalar/bytes case
                    la = data["language_actions"]
                    if isinstance(la, bytes):
                        la = la.decode("utf-8")
                    else:
                        raise ValueError(f"Language actions is not a bytes string: {la}")
                    inputs["language_actions"] = la

        # Optional calibration/context passthroughs for visualization
        for k in ("camera_intrinsics", "camera_extrinsics"):
            if k in data["observation"]:
                inputs[k] = np.asarray(data[k])
        if "cartesian_position_window" in data["observation"]:
            inputs["cartesian_position_window"] = np.asarray(data["observation"]["cartesian_position_window"])

        return inputs


@dataclasses.dataclass(frozen=True)
class CoTOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
