import dataclasses
import logging

import einops
import numpy as np
from openpi import transforms as upstream_transforms
import wandb

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.lang_action_util import sum_language_actions
from openpi_cot.dataloader.lang_action_util import summarize_numeric_actions
from openpi_cot.models.adapters.model_adapter import ExtendedModelType


def _maybe_parse_serialized_tensor_to_ndarray(b) -> np.ndarray | None:
    try:
        if not isinstance(b, (bytes, np.bytes_)):
            return None
        import tensorflow as tf  # Lazy import to avoid TF dependency at module import time

        t = tf.io.parse_tensor(b, out_type=tf.float32)
        return t.numpy()
    except Exception:
        return None


def _parse_image(image) -> np.ndarray:
    if image is None:
        return None
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _safe_decode_bytes(value: bytes | np.bytes_) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("utf-8", errors="replace")


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
            out.append(_safe_decode_bytes(item))
        else:
            out.append(str(item))
    return out


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

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        if self.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"
        assert "observation" in data
        assert "exterior_image_1_left" in data["observation"]
        base_image = _parse_image(data["observation"]["exterior_image_1_left"])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        if "wrist_image_left" in data["observation"]:
            wrist_image = _parse_image(data["observation"]["wrist_image_left"])
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
        names = ("base_0_rgb", "left_wrist_0_rgb")
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

        if "language_actions" in data:
            la = data["language_actions"]
            assert isinstance(la[0], bytes)
            if _maybe_parse_serialized_tensor_to_ndarray(la[0]) is not None:  # oxe case
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
                raw_array = [_maybe_parse_serialized_tensor_to_ndarray(x) for x in la_used]
                summed = summarize_numeric_actions(raw_array, self.sum_decimal, self.include_rotation)
                inputs["language_actions"] = summed
            else:
                seq = _to_str_list(la)
                if seq is not None:
                    summed = sum_language_actions(seq, self.sum_decimal, self.include_rotation)
                    if summed is not None and len(summed) > 0:
                        inputs["language_actions"] = summed
                else:
                    # Scalar/bytes case
                    if isinstance(la, bytes):
                        la = la.decode("utf-8")
                    else:
                        raise ValueError(f"Language actions is not a bytes string: {la}")
                    inputs["language_actions"] = la

        def _is_trivial_image(img: np.ndarray, mask: np.ndarray) -> bool:
            if np.all(img == 0):
                if mask == np.False_:
                    return False
                return True
            return np.all(img == 0) or np.all(img == 255)

        images_for_check = {
            "base_0_rgb": [base_image, image_masks[0]],
            "left_wrist_0_rgb": [wrist_image, image_masks[1]],
        }

        if any(_is_trivial_image(img, mask) for img, mask in images_for_check.values()) or (
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
