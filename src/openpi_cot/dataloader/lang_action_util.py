import re

import numpy as np

AXIS_PERM = np.array([0, 2, 1])  # X -> dx (right/left), Z -> dy (forward/backward), Y -> dz (down/up)
AXIS_SIGN = np.array([1, 1, 1])  # start with no flips


def _format_numeric(val: float, sum_decimal: str) -> str:
    # Match droid policy formatting for numbers

    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m:
            decimals = int(m.group(1))
    return f"{val:.{decimals}f}"


def summarize_numeric_actions(arr_like, sum_decimal: str, include_rotation: bool = False) -> str | None:
    """Convert numeric delta EE actions ([..., 7]) into a language string.

    Expects translation in indices [0,1,2] (meters) and gripper at index 6.
    Sums over time, converts meters→cm, emits signed directional commands and final gripper setting.
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    # Convert to centimeters
    if sum_decimal == "no_number":
        decimals = 0
    else:
        decimals = int(re.fullmatch(r"(\d+)f", sum_decimal).group(1))

    # Sum translations over the window
    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = round(abs(droll_rad * 180.0 / np.pi))
        dpitch = round(abs(dpitch_rad * 180.0 / np.pi))
        dyaw = round(abs(dyaw_rad * 180.0 / np.pi))

    parts: list[str] = []

    if sum_decimal == "no_number":
        if dx_m > 0:
            parts.append("move forward")
        elif dx_m < 0:
            parts.append("move back")
        if dy_m > 0:
            parts.append("move left")
        elif dy_m < 0:
            parts.append("move right")
        if dz_m > 0:
            parts.append("move up")
        elif dz_m < 0:
            parts.append("move down")
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt up")
            elif dpitch_rad < 0:
                parts.append("tilt down")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0:
                parts.append(f"tilt up {dpitch} degrees")
            elif dpitch_rad < 0:
                parts.append(f"tilt down {dpitch} degrees")
            if dyaw_rad > 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    # Final gripper value from last step
    g_last = float(arr[-1, 6])
    parts.append(f"set gripper to {g_last:.2f}")

    return " and ".join(parts)


def sum_language_actions(actions_list, sum_decimal, include_rotation=False):
    assert not include_rotation, "Rotation not supported yet"
    # Determine rounding/formatting behavior from sum_decimal
    decimals = 0
    no_number = False
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            no_number = True
        else:
            m = re.fullmatch(r"(\d+)f", sum_decimal)
            if m:
                decimals = int(m.group(1))

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
