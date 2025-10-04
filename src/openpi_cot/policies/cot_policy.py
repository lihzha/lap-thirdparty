import dataclasses
import logging

import numpy as np
from openpi import transforms as upstream_transforms
import wandb

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import is_trivial_image
from openpi_cot.policies.utils import maybe_parse_serialized_tensor_to_ndarray
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import sum_language_actions
from openpi_cot.policies.utils import summarize_numeric_actions
from openpi_cot.policies.utils import to_str_list


# TODO: during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    sum_decimal: str = "0f"
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT
    include_rotation: bool = False
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS

    def _prepare_inputs(self, data: dict) -> tuple[dict, dict]:
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = parse_image(data["observation"]["exterior_image_1_left"])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        if "wrist_image_left" in data["observation"]:
            wrist_image = parse_image(data["observation"]["wrist_image_left"])
            if np.all(wrist_image == 0.0):
                wrist_image = np.zeros_like(base_image)
                wrist_image_mask = np.False_
            else:
                wrist_image_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_image_mask = np.False_

        if np.all(base_image == 0):
            base_image_mask = np.False_
        else:
            base_image_mask = np.True_

        # Optional dropout: randomly mask out wrist image
        if self.wrist_image_dropout_prob > 0.0:
            if np.random.rand() < float(self.wrist_image_dropout_prob):
                wrist_image_mask = np.False_

        assert self.model_type == ExtendedModelType.PI_COT
        names = IMAGE_KEYS

        images = (
            base_image,
            wrist_image,
        )
        image_masks = (
            base_image_mask,
            wrist_image_mask,
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

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        images_for_check = {
            "base_0_rgb": [base_image, image_masks[0]],
            "left_wrist_0_rgb": [wrist_image, image_masks[1]],
        }

        if any(is_trivial_image(img, mask) for img, mask in images_for_check.values()) or (
            prompt_str is None or prompt_str.strip() == ""
        ):
            log_payload = {
                "policy/anomaly_base": wandb.Image(
                    base_image, caption=f"Dataset: {data['dataset_name'].decode('utf-8')}, prompt: {prompt_str}"
                )
                if base_image is not None
                else None,
                "policy/anomaly_wrist": wandb.Image(
                    wrist_image,
                    caption=f"Dataset: {data['dataset_name'].decode('utf-8')}, language actions: {inputs['language_actions']}",
                )
                if wrist_image is not None
                else None,
            }
            wandb.log({k: v for k, v in log_payload.items() if v is not None})
            logging.warning("Invalid policy inputs: trivial image or missing prompt")

        return inputs

    def _prepare_language_actions(self, data: dict) -> dict:
        if "language_actions" in data:
            la = data["language_actions"]
            assert isinstance(la[0], bytes)
            if maybe_parse_serialized_tensor_to_ndarray(la[0]) is not None:  # oxe case
                # Only use the non-padded portion according to control_frequency, if present
                cf_val = data.get("control_frequency")
                try:
                    cf = int(np.asarray(cf_val).item()) if cf_val is not None else None
                except Exception:
                    cf = None
                if cf is not None:
                    la_used = la[: int(cf)]
                else:
                    la_used = la
                raw_array = [maybe_parse_serialized_tensor_to_ndarray(x) for x in la_used]
                summed = summarize_numeric_actions(raw_array, self.sum_decimal, self.include_rotation)
                return summed
            seq = to_str_list(la)
            if seq is not None:
                summed = sum_language_actions(seq, self.sum_decimal, self.include_rotation)
                if summed is not None and len(summed) > 0:
                    return summed
            else:
                # Scalar/bytes case
                if isinstance(la, bytes):
                    la = la.decode("utf-8")
                else:
                    raise ValueError(f"Language actions is not a bytes string: {la}")
                return la
        return None

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)
        if self.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"
        language_actions = self._prepare_language_actions(data)
        if language_actions is not None:
            inputs["language_actions"] = language_actions

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
