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


def _sum_language_actions(actions_list, sum_decimal):
    import re

    # Determine rounding/formatting behavior from sum_decimal
    decimals = 0
    no_number = False
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            no_number = True
        else:
            m = re.fullmatch(r"(\d+)f", sum_decimal)
            if m:
                try:
                    decimals = int(m.group(1))
                except Exception:
                    decimals = 0

    # Accumulate per-direction totals
    totals = {
        "left": 0.0,
        "right": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "up": 0.0,
        "down": 0.0,
    }
    units = dict.fromkeys(totals.keys(), "cm")
    last_gripper_value_str = None
    if actions_list is None:
        return None
    for action in actions_list:
        if not action:
            continue
        parts = action.split(" and ")
        for mv in parts:
            mv = mv.strip()
            # Capture the last gripper command in the chunk
            g = re.match(r"set\s+gripper\s+to\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", mv)
            if g:
                last_gripper_value_str = g.group(1)
                continue
            # Sum directional move commands
            m = re.match(r"move\s+(\w+)\s+([\d.]+)\s*(\w+)", mv)
            if not m:
                continue
            direction = m.group(1)
            value = float(m.group(2))
            unit = m.group(3)
            if direction in totals:
                totals[direction] += value
                units[direction] = unit

    # Helper to format a magnitude with configured decimals
    def _fmt_mag(x: float) -> str:
        return f"{x:.{decimals}f}"

    # Compute axis-wise nets
    result = []
    # X axis: right/left
    net = totals["right"] - totals["left"]
    if no_number:
        if net > 0:
            result.append("move right")
        elif net < 0:
            result.append("move left")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move right {_fmt_mag(mag)} {units['right']}")
        elif net < 0 and mag > 0:
            result.append(f"move left {_fmt_mag(mag)} {units['left']}")
    # Y axis: forward/backward
    net = totals["forward"] - totals["backward"]
    if no_number:
        if net > 0:
            result.append("move forward")
        elif net < 0:
            result.append("move backward")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move forward {_fmt_mag(mag)} {units['forward']}")
        elif net < 0 and mag > 0:
            result.append(f"move backward {_fmt_mag(mag)} {units['backward']}")
    # Z axis: up/down
    net = totals["up"] - totals["down"]
    if no_number:
        if net > 0:
            result.append("move up")
        elif net < 0:
            result.append("move down")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move up {_fmt_mag(mag)} {units['up']}")
        elif net < 0 and mag > 0:
            result.append(f"move down {_fmt_mag(mag)} {units['down']}")

    # Append the final gripper setting if present (rounded to 1 decimal place)
    if last_gripper_value_str is not None:
        try:
            gv = float(last_gripper_value_str)
            result.append(f"set gripper to {gv:.2f}")
        except Exception:
            # Fallback to raw string if parsing fails
            result.append(f"set gripper to {last_gripper_value_str}")

    return " and ".join(result)


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
        assert "observation/exterior_image_1_left" in data
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        if "observation/wrist_image_left" in data:
            wrist_image = _parse_image(data["observation/wrist_image_left"])
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
            "state": data["observation/state"],
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
            la_val = data["language_actions"]
            seq = _to_str_list(la_val)  # None if numeric array
            summarized = None
            if seq is not None:
                summarized = _sum_language_actions(seq, self.sum_decimal)
            else:
                summarized = _summarize_numeric_actions(la_val, self.sum_decimal)
            if summarized is not None and len(summarized) > 0:
                inputs["language_actions"] = summarized

        # if "language_actions" in data:
        #     seq = _to_str_list(data["language_actions"])
        #     if seq is not None:
        #         summed = _sum_language_actions(seq, self.sum_decimal)
        #         if summed is not None and len(summed) > 0:
        #             inputs["language_actions"] = summed
        #     else:
        #         # Scalar/bytes case
        #         la = data["language_actions"]
        #         if isinstance(la, bytes):
        #             la = la.decode("utf-8")
        #         else:
        #             raise ValueError(f"Language actions is not a bytes string: {la}")
        #         inputs["language_actions"] = la

        # Optional calibration/context passthroughs for visualization
        for k in ("camera_intrinsics", "camera_extrinsics", "observation/cartesian_position_window"):
            if k in data:
                nk = "cartesian_position_window" if k.endswith("cartesian_position_window") else k
                inputs[nk] = np.asarray(data[k])

        return inputs


@dataclasses.dataclass(frozen=True)
class CoTOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
