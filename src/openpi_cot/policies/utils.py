import re

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R

AXIS_PERM = np.array([0, 2, 1])  # X -> dx (right/left), Z -> dy (forward/backward), Y -> dz (down/up)
AXIS_SIGN = np.array([1, 1, 1])  # start with no flips


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    """Round a value to the nearest multiple of n."""
    return int(round(value / n) * n)


def parse_image(image) -> np.ndarray:
    if image is None:
        return None
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[1] == 3 and len(image.shape) == 4:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def _safe_decode_bytes(value: bytes | np.bytes_) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("utf-8", errors="replace")


def to_str_list(x):
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


def transform_actions_to_eef_frame(actions: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """Transform actions from base frame to end effector frame.

    Args:
        actions: Array of shape (n_steps, action_dim) where action_dim >= 7
                 Format: [dx, dy, dz, droll, dpitch, dyaw, gripper, ...]
                 All deltas are in base frame
        initial_state: State vector containing initial EEF pose
                       Expected format: [x, y, z, qx, qy, qz, qw, ...]
                       or [x, y, z, roll, pitch, yaw, ...]

    Returns:
        Transformed actions in EEF frame with same shape as input
    """
    actions = np.asarray(actions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)

    assert actions.ndim == 1
    transformed_actions = actions.copy()

    assert len(initial_state.shape) == 1 and initial_state[7] == 0, "Only supporting euler angle now"  # noqa: PT018
    euler = initial_state[3:6]
    initial_rotation = R.from_euler("xyz", euler)

    # Get rotation matrix: base -> EEF
    R_base_to_eef = initial_rotation.as_matrix().T  # Transpose to get EEF <- base

    delta_pos_base = actions[:3]
    delta_pos_eef = R_base_to_eef @ delta_pos_base
    # Apply additional transformation: y -> -y, z -> -z
    delta_pos_eef[1] = -delta_pos_eef[1]
    delta_pos_eef[2] = -delta_pos_eef[2]
    transformed_actions[:3] = delta_pos_eef

    delta_rot_base = actions[3:6]  # [roll, pitch, yaw] in radians
    # Convert euler angles to rotation matrix
    R_delta_base = R.from_euler("xyz", delta_rot_base).as_matrix()
    # Transform to EEF frame: R_delta_eef = R_base_to_eef @ R_delta_base @ R_base_to_eef.T
    R_delta_eef = R_base_to_eef @ R_delta_base @ R_base_to_eef.T
    # Convert back to euler angles
    delta_rot_eef = R.from_matrix(R_delta_eef).as_euler("xyz")
    # Apply additional transformation: y -> -y, z -> -z to rotation as well
    delta_rot_eef[1] = -delta_rot_eef[1]
    delta_rot_eef[2] = -delta_rot_eef[2]
    transformed_actions[3:6] = delta_rot_eef

    return transformed_actions


def transform_actions_from_eef_frame(actions: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """Transform actions from end effector frame to base frame (inverse of transform_actions_to_eef_frame).

    Args:
        actions: Array of shape (n_steps, action_dim) where action_dim >= 7
                 Format: [dx, dy, dz, droll, dpitch, dyaw, gripper, ...]
                 All deltas are in EEF frame
        initial_state: State vector containing initial EEF pose
                       Expected format: [x, y, z, qx, qy, qz, qw, ...]
                       or [x, y, z, roll, pitch, yaw, ...]

    Returns:
        Transformed actions in base frame with same shape as input
    """
    actions = np.asarray(actions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    if len(initial_state.shape) == 2:
        assert initial_state.shape[0] == 1
        initial_state = initial_state[0]
    # assert len(initial_state.shape) == 1 and initial_state[7] == 0, "Only supporting euler angle now"
    # assert len(initial_state.shape) == 1 and initial_state[7] == 0, "Only supporting euler angle now"

    if actions.ndim == 1:
        actions = actions[None, :]

    transformed_actions = actions.copy()

    # Extract initial EEF orientation from state
    # Try to detect if state contains quaternions (length 7+) or euler angles (length 6+)

    # Assume euler angle format: [x, y, z, roll, pitch, yaw, ...]
    euler = initial_state[3:6]
    initial_rotation = R.from_euler("xyz", euler)

    # Get rotation matrix: EEF -> base (inverse of base -> EEF)
    R_eef_to_base = initial_rotation.as_matrix()

    # Transform translation deltas from EEF frame to base frame
    for i in range(len(transformed_actions)):
        delta_pos_eef = actions[i, :3].copy()
        # Apply inverse transformation: y -> -y, z -> -z (same as forward since it's a sign flip)
        delta_pos_eef[1] = -delta_pos_eef[1]
        delta_pos_eef[2] = -delta_pos_eef[2]
        delta_pos_base = R_eef_to_base @ delta_pos_eef
        transformed_actions[i, :3] = delta_pos_base

        # Transform rotation deltas from EEF frame to base frame
        if actions.shape[-1] >= 6:
            delta_rot_eef = actions[i, 3:6].copy()  # [roll, pitch, yaw] in radians
            # Apply inverse transformation: y -> -y, z -> -z
            delta_rot_eef[1] = -delta_rot_eef[1]
            delta_rot_eef[2] = -delta_rot_eef[2]
            # Convert euler angles to rotation matrix
            R_delta_eef = R.from_euler("xyz", delta_rot_eef).as_matrix()
            # Transform to base frame: R_delta_base = R_eef_to_base @ R_delta_eef @ R_eef_to_base.T
            R_delta_base = R_eef_to_base @ R_delta_eef @ R_eef_to_base.T
            # Convert back to euler angles
            delta_rot_base = R.from_matrix(R_delta_base).as_euler("xyz")
            transformed_actions[i, 3:6] = delta_rot_base

    return transformed_actions


def _summarize_compact_numeric_actions(arr_like, include_rotation: bool = False) -> str:
    """Convert numeric delta EE actions to compact format: <+03 +05 -08 +10 +00 +02 1>

    Format: <dx dy dz droll dpitch dyaw grip>
    - dx, dy, dz in cm (signed, 2-digit integers)
    - droll, dpitch, dyaw in degrees (signed, 2-digit integers) if include_rotation
    - grip as 0 or 1
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    # Sum translations over the window and convert to cm
    dx_cm = int(round(float(arr[..., 0].sum()) * 100.0))
    dy_cm = int(round(float(arr[..., 1].sum()) * 100.0))
    dz_cm = int(round(float(arr[..., 2].sum()) * 100.0))

    # Format with sign and 2 digits
    parts = [
        f"{dx_cm:+03d}",
        f"{dy_cm:+03d}",
        f"{dz_cm:+03d}",
    ]

    if include_rotation:
        # Convert rotations to degrees and round to nearest 5
        droll_deg = _round_to_nearest_n(float(arr[..., 3].sum()) * 180.0 / np.pi, 5)
        dpitch_deg = _round_to_nearest_n(float(arr[..., 4].sum()) * 180.0 / np.pi, 5)
        dyaw_deg = _round_to_nearest_n(float(arr[..., 5].sum()) * 180.0 / np.pi, 5)
        parts.extend(
            [
                f"{droll_deg:+03d}",
                f"{dpitch_deg:+03d}",
                f"{dyaw_deg:+03d}",
            ]
        )

    # Gripper: threshold at 0.5 to convert to binary
    g_last = float(arr[-1, 6])
    grip_binary = 1 if g_last >= 0.5 else 0
    parts.append(str(grip_binary))

    return "<" + " ".join(parts) + ">"
    # return " ".join(parts)


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

    # Handle compact format
    if sum_decimal == "compact":
        return _summarize_compact_numeric_actions(arr, include_rotation)

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
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), 5)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), 5)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), 5)

    parts: list[str] = []

    if sum_decimal == "no_number":
        if dx_m > 0 and dx != 0:
            parts.append("move forward")
        elif dx_m < 0 and dx != 0:
            parts.append("move back")
        if dy_m > 0 and dy != 0:
            parts.append("move left")
        if dy_m < 0 and dy != 0:
            parts.append("move right")
        if dz_m > 0 and dz != 0:
            parts.append("move up")
        elif dz_m < 0 and dz != 0:
            parts.append("move down")
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    # Final gripper value from last step
    g_last = float(arr[-1, 6])
    if g_last > 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    # parts.append(f"set gripper to {g_last:.0f}")
    return ", ".join(parts)


def summarize_bimanual_numeric_actions(arr_like, sum_decimal: str, include_rotation: bool = False) -> str | None:
    """Convert bimanual numeric delta EE actions into a language string.

    Expects format: [left_ee_pose (6), left_gripper (1), right_ee_pose (6), right_gripper (1)] = 14 dims
    left_ee_pose and right_ee_pose are [x, y, z, roll, pitch, yaw] in meters and radians.
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 14:
        return None

    # Split into left and right arms
    left_actions = arr[..., :7]  # [x, y, z, r, p, y, gripper]
    right_actions = arr[..., 7:14]  # [x, y, z, r, p, y, gripper]

    # Handle compact format with bimanual structure: <L ... R ...>
    if sum_decimal == "compact":
        left_compact = _summarize_compact_numeric_actions(left_actions, include_rotation)
        right_compact = _summarize_compact_numeric_actions(right_actions, include_rotation)

        # Remove the angle brackets from individual summaries and combine
        left_values = left_compact[1:-1]  # Remove < and >
        right_values = right_compact[1:-1]  # Remove < and >

        return f"<L {left_values} R {right_values}>"

    # Summarize each arm separately for verbose formats
    left_summary = summarize_numeric_actions(left_actions, sum_decimal, include_rotation)
    right_summary = summarize_numeric_actions(right_actions, sum_decimal, include_rotation)

    if left_summary is None or right_summary is None:
        return None

    return f"Left arm: {left_summary}. Right arm: {right_summary}"


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


def is_idle_language_action(
    language_action: str,
    sum_decimal: str,
    include_rotation: bool = False,
    translation_threshold: float = 1.0,
    rotation_threshold_deg: float = 10.0,
) -> bool:
    """Check if a language action represents idle (minimal movement).

    Args:
        language_action: Language action string in compact or verbose format
        sum_decimal: Format specifier ("compact", "no_number", or "Xf")
        include_rotation: Whether to consider rotation in idle detection
        translation_threshold: L2 norm threshold for translation (in cm)
        rotation_threshold_deg: L2 norm threshold for rotation (in degrees)

    Returns:
        True if the action is considered idle, False otherwise
    """
    if not language_action or not isinstance(language_action, str):
        return True  # Empty or invalid action is considered idle

    # Parse based on format
    if sum_decimal == "compact":
        # Compact format: <+dx +dy +dz [+droll +dpitch +dyaw] grip>
        if include_rotation:
            # Format: <+09 +09 -08 +10 -05 +15 1>
            match = re.search(
                r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>", language_action
            )
            if match:
                dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))
                droll_deg, dpitch_deg, dyaw_deg = int(match.group(4)), int(match.group(5)), int(match.group(6))

                translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
                rotation_l2 = np.sqrt(droll_deg**2 + dpitch_deg**2 + dyaw_deg**2)
                return translation_l2 < translation_threshold and rotation_l2 < rotation_threshold_deg
            return True  # Failed to parse, treat as idle
        # Format: <+09 +09 -08 1>
        match = re.search(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>", language_action)
        if match:
            dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))
            translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
            return translation_l2 < translation_threshold
        return True  # Failed to parse, treat as idle
    # Verbose format: "move forward X cm and move right Y cm..."
    # Special handling for directional_only format (no_number)
    if sum_decimal == "no_number":
        # For directional_only, just check if any directional movement exists
        # Pattern: "move [direction]" without numeric values
        move_pattern_no_number = re.compile(
            r"move\s+(right|left|forward|backward|back|up|down)(?!\s+[\d.])", re.IGNORECASE
        )
        has_movement = bool(move_pattern_no_number.search(language_action))

        if not include_rotation:
            # If any movement command exists, it's not idle
            return not has_movement

        # Check for rotation commands without numbers
        rotation_pattern_no_number = re.compile(
            r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)(?!\s+[\d.])",
            re.IGNORECASE,
        )
        has_rotation = bool(rotation_pattern_no_number.search(language_action))

        # If any movement or rotation exists, it's not idle
        return not (has_movement or has_rotation)

    # Parse all movement commands with numeric values
    move_pattern = re.compile(r"move\s+(right|left|forward|backward|back|up|down)\s+([\d.]+)\s*cm", re.IGNORECASE)

    dx_cm = dy_cm = dz_cm = 0.0
    for match in move_pattern.finditer(language_action):
        direction = match.group(1).lower()
        value = float(match.group(2))

        if direction == "forward":
            dx_cm += value
        elif direction in ("backward", "back"):
            dx_cm -= value
        elif direction == "left":
            dy_cm += value
        elif direction == "right":
            dy_cm -= value
        elif direction == "up":
            dz_cm += value
        elif direction == "down":
            dz_cm -= value

    translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)

    if not include_rotation:
        return translation_l2 < translation_threshold
    # Parse rotation commands
    rotation_pattern = re.compile(
        r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
        re.IGNORECASE,
    )

    droll_deg = dpitch_deg = dyaw_deg = 0.0
    for match in rotation_pattern.finditer(language_action):
        rotation_type = match.group(1).lower()
        value = float(match.group(2))

        if rotation_type == "tilt left":
            droll_deg += value
        elif rotation_type == "tilt right":
            droll_deg -= value
        elif rotation_type in {"tilt up", "tilt forward"}:
            dpitch_deg += value
        elif rotation_type in {"tilt down", "tilt back"}:
            dpitch_deg -= value
        elif rotation_type == "rotate counterclockwise":
            dyaw_deg += value
        elif rotation_type == "rotate clockwise":
            dyaw_deg -= value

    rotation_l2 = np.sqrt(droll_deg**2 + dpitch_deg**2 + dyaw_deg**2)
    return translation_l2 < translation_threshold and rotation_l2 < rotation_threshold_deg


def is_all_1s_language_action(
    language_action: str,
    sum_decimal: str,
    include_rotation: bool = False,
) -> bool:
    """Check if all directional movements are exactly 1 cm (suggesting noisy/unreliable data).

    Args:
        language_action: Language action string in compact or verbose format
        sum_decimal: Format specifier ("compact", "no_number", or "Xf")
        include_rotation: Whether to consider rotation in the check

    Returns:
        True if all non-zero movements are exactly 1 cm, False otherwise
    """
    if not language_action or not isinstance(language_action, str):
        return False  # Empty or invalid action is not all 1s

    # Parse based on format
    if sum_decimal == "compact":
        # Compact format: <+dx +dy +dz [+droll +dpitch +dyaw] grip>
        if include_rotation:
            # Format: <+09 +09 -08 +10 -05 +15 1>
            match = re.search(
                r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>", language_action
            )
            if match:
                dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))

                # Get absolute values of non-zero movements
                translation_values = [abs(v) for v in [dx_cm, dy_cm, dz_cm] if v != 0]

                # Check if all non-zero translations are exactly 1 cm
                all_translations_are_1 = all(v == 1 for v in translation_values) if translation_values else False

                # For rotation, we don't check "all 1s" since rotation is in degrees
                # Just return whether translations are all 1s
                return all_translations_are_1 and len(translation_values) > 0
            return False
        # Format: <+09 +09 -08 1>
        match = re.search(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>", language_action)
        if match:
            dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))
            # Get absolute values of non-zero movements
            translation_values = [abs(v) for v in [dx_cm, dy_cm, dz_cm] if v != 0]
            # Check if all non-zero movements are exactly 1 cm
            return all(v == 1 for v in translation_values) and len(translation_values) > 0
        return False

    # Verbose format: "move forward X cm and move right Y cm..."
    # Special handling for directional_only format (no_number)
    if sum_decimal == "no_number":
        # For directional_only, we can't check if movements are 1 cm since no numbers
        return False

    # Parse all movement commands with numeric values
    move_pattern = re.compile(r"move\s+(right|left|forward|backward|back|up|down)\s+([\d.]+)\s*cm", re.IGNORECASE)

    movement_values = []
    for match in move_pattern.finditer(language_action):
        value = float(match.group(2))
        if value != 0:
            movement_values.append(value)

    # Check if all non-zero movements are exactly 1 cm
    if not movement_values:
        return False

    return all(abs(v - 1.0) < 0.01 for v in movement_values)  # Use small epsilon for float comparison
