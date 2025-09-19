import dataclasses

import einops
import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.lang_action_util import sum_language_actions
from openpi_cot.dataloader.lang_action_util import summarize_numeric_actions
from openpi_cot.models.adapters.model_adapter import ExtendedModelType


def _maybe_parse_serialized_tensor_to_ndarray(b) -> np.ndarray | None:
    if not isinstance(b, (bytes, np.bytes_)):
        return None
    import tensorflow as tf  # Lazy import to avoid TF dependency at module import time

    t = tf.io.parse_tensor(b, out_type=tf.float32)
    return t.numpy()


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
            la = data["language_actions"]

            # Normalize to a Python list of elements
            if isinstance(la, np.ndarray):
                # try:
                #     seq_like = la.tolist()
                # except Exception:
                #     seq_like = list(la)
                raise ValueError("Language actions should be a list of strings")
            if isinstance(la, (list, tuple)):
                # seq_like = list(la)
                raise ValueError("Language actions should be a list of strings")
            seq_like = [la]

            numeric_rows: list[np.ndarray] = []
            nl_rows: list[str] = []

            for item in seq_like:
                # If already numeric array
                if isinstance(item, np.ndarray):
                    numeric_rows.append(item)
                    continue
                # Bytes: may be serialized tensor or a utf-8 NL string
                if isinstance(item, (bytes, np.bytes_)):
                    parsed = _maybe_parse_serialized_tensor_to_ndarray(item)
                    if parsed is not None:
                        numeric_rows.append(parsed)
                        continue
                    nl_rows.append(item.decode("utf-8"))
                    continue
                # Plain string case
                if isinstance(item, str):
                    nl_rows.append(item)

            summed: str | None = None
            if len(numeric_rows) > 0:
                # Convert to array [W, A] if possible
                numeric_window = np.asarray(numeric_rows, dtype=float)
                summed = summarize_numeric_actions(numeric_window, self.sum_decimal)
            elif len(nl_rows) > 0:
                summed = sum_language_actions(nl_rows, self.sum_decimal)

            if summed is not None and len(summed) > 0:
                inputs["language_actions"] = summed

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
