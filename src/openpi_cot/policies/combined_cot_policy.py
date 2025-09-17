import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.droid_cot_policy import _parse_image
from openpi_cot.policies.droid_cot_policy import _sum_language_actions
from openpi_cot.policies.droid_cot_policy import _to_str_list


def _get_first_present(data: dict, *keys: str):
    for k in keys:
        if k in data:
            return data[k]
    return None


def _format_numeric(val: float, sum_decimal: str) -> str:
    # Match droid policy formatting for numbers
    import re

    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m:
            try:
                decimals = int(m.group(1))
            except Exception:
                decimals = 0
    return f"{val:.{decimals}f}"


def _summarize_numeric_actions(arr_like, sum_decimal: str) -> str | None:
    """Convert numeric delta EE actions ([..., 7]) into a language string.

    Expects translation in indices [0,1,2] (meters) and gripper at index 6.
    Sums over time, converts meters→cm, emits signed directional commands and final gripper setting.
    """
    try:
        arr = np.asarray(arr_like, dtype=float)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    # Sum translations over the window
    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    # Convert to centimeters
    dx = abs(dx_m * 100.0)
    dy = abs(dy_m * 100.0)
    dz = abs(dz_m * 100.0)

    parts: list[str] = []

    if sum_decimal == "no_number":
        if dx_m > 0:
            parts.append("move right")
        elif dx_m < 0:
            parts.append("move left")
        if dy_m > 0:
            parts.append("move forward")
        elif dy_m < 0:
            parts.append("move backward")
        if dz_m > 0:
            parts.append("move up")
        elif dz_m < 0:
            parts.append("move down")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx > 0:
            parts.append(f"move right {fmt_dx} cm")
        elif dx_m < 0 and dx > 0:
            parts.append(f"move left {fmt_dx} cm")
        if dz_m > 0 and dz > 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz > 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy > 0:
            parts.append(f"move forward {fmt_dy} cm")
        elif dy_m < 0 and dy > 0:
            parts.append(f"move backward {fmt_dy} cm")

    # Final gripper value from last step
    try:
        g_last = float(arr[-1, 6])
        parts.append(f"set gripper to {g_last:.2f}")
    except Exception:
        pass

    return " and ".join(parts)


@dataclasses.dataclass(frozen=True)
class CombinedCoTInputs(upstream_transforms.DataTransformFn):
    action_dim: int
    sum_decimal: str = "0f"
    wrist_image_dropout_prob: float = 0.0
    text_state_dropout_prob: float = 0.0
    state_norm_stats: upstream_transforms.NormStats | None = None
    model_type: ExtendedModelType = ExtendedModelType.PI_COT

    def __call__(self, data: dict) -> dict:
        # State: prefer explicit cartesian + gripper (DROID); else fallback to proprio (OXE)
        if (cart := data.get("observation/cartesian_position")) is not None:
            gripper = data.get("observation/gripper_position")
            if gripper is None:
                gripper = np.zeros((1,), dtype=np.float32)
            gripper = np.asarray(gripper)
            if gripper.ndim == 0:
                gripper = gripper[np.newaxis]
            state = np.concatenate([np.asarray(cart), gripper])
        else:
            prop = _get_first_present(data, "observation/proprio")
            if prop is None:
                raise KeyError("Missing state: expected cartesian+gripper or proprio in observation")
            state = np.asarray(prop)

        # Images: support both DROID and OXE canonical keys (after repack)
        base_img = _get_first_present(
            data,
            "observation/exterior_image_1_left",
            "observation/image_primary",
            "observation/image",
        )
        if base_img is None:
            raise KeyError("Missing base image in observation")
        base_image = _parse_image(base_img)

        wrist_img = _get_first_present(
            data,
            "observation/wrist_image_left",
            "observation/image_wrist",
            "observation/wrist_image",
        )
        if wrist_img is None:
            wrist_image = np.zeros_like(base_image)
            wrist_image_mask = np.False_
        else:
            wrist_image = _parse_image(wrist_img)
            wrist_image_mask = np.True_

        # Optional dropout: randomly mask out wrist image
        if self.wrist_image_dropout_prob > 0.0:
            if np.random.rand() < float(self.wrist_image_dropout_prob):
                wrist_image_mask = np.False_

        # Assemble images
        names = ("base_0_rgb", "left_wrist_0_rgb")
        images = (base_image, wrist_image)
        image_masks = (np.True_, wrist_image_mask)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions: accept either key
        if "actions" in data or "action" in data:
            actions_val = data.get("actions", data.get("action"))
            actions = upstream_transforms.pad_to_dim(actions_val, self.action_dim)
            inputs["actions"] = np.asarray(actions)

        # Prompt always required upstream; normalize to python str
        if "prompt" in data:
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

        # Language actions: support DROID-style list-of-strings OR numeric arrays (OXE)
        if "language_actions" in data:
            la_val = data["language_actions"]
            seq = _to_str_list(la_val)  # None if numeric array
            summarized = None
            if seq is not None:
                summarized = _sum_language_actions(seq, self.sum_decimal)
            else:
                summarized = _summarize_numeric_actions(la_val, self.sum_decimal)
            if summarized is not None and len(summarized) > 0:
                inputs["language_actions"] = summarized

        # Optional passthroughs for visualization/calibration
        for k in ("camera_intrinsics", "camera_extrinsics", "observation/cartesian_position_window"):
            if k in data:
                nk = "cartesian_position_window" if k.endswith("cartesian_position_window") else k
                inputs[nk] = np.asarray(data[k])

        return inputs


@dataclasses.dataclass(frozen=True)
class CombinedCoTOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
