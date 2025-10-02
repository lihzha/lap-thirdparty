import dataclasses
import re

import numpy as np
from openpi import transforms

from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def reasoning_to_action(
    sentence: str,
) -> np.ndarray:
    """
    Invert `describe_movement`.
    Now supports rotation phrases:
      - "tilt left/right <deg> degrees"  -> roll (+/-)
      - "tilt up/down <deg> degrees"     -> pitch (+/-)
      - "rotate counterclockwise/clockwise <deg> degrees" -> yaw (+/-)
    Conventions:
      tilt left  => droll > 0
      tilt right => droll < 0
      tilt up    => dpitch > 0
      tilt down  => dpitch < 0
      rotate counterclockwise => dyaw > 0
      rotate clockwise        => dyaw < 0
    Rotations are applied as intrinsic XYZ (roll→pitch→yaw) increments each step.
    Translations are interpreted in the camera frame and added directly to the pose.
    """
    print(sentence)

    # Regex patterns
    move_pat = re.compile(r"move\s+(right|left|forward|backward|up|down)\s+([\-+]?\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
    grip_pat = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+(?:\.\d+)?)", re.IGNORECASE)

    tilt_pat = re.compile(r"tilt\s+(left|right|up|down)\s+([\-+]?\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)
    rot_pat = re.compile(r"rotate\s+(counterclockwise|clockwise)\s+([\-+]?\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)

    # --- Parse translations in cm along (dx, dy, dz) camera frame ---
    dx_cm = dy_cm = dz_cm = 0.0
    for m in move_pat.finditer(sentence):
        direction = m.group(1).lower()
        value_cm = float(m.group(2))
        if direction == "forward":
            dx_cm += value_cm
        elif direction == "backward":
            dx_cm -= value_cm
        elif direction == "left":
            dy_cm += value_cm
        elif direction == "right":
            dy_cm -= value_cm
        elif direction == "up":
            dz_cm += value_cm
        elif direction == "down":
            dz_cm -= value_cm

    # --- Parse rotations in degrees (roll, pitch, yaw) ---
    droll_deg = dpitch_deg = dyaw_deg = 0.0

    # tilt: affects roll (left/right) and pitch (up/down)
    for m in tilt_pat.finditer(sentence):
        sense = m.group(1).lower()
        val = float(m.group(2))
        if sense == "left":
            droll_deg += val
        elif sense == "right":
            droll_deg -= val
        elif sense == "up":
            dpitch_deg += val
        elif sense == "down":
            dpitch_deg -= val

    # rotate: affects yaw (ccw/cw)
    for m in rot_pat.finditer(sentence):
        sense = m.group(1).lower()
        val = float(m.group(2))
        if sense == "counterclockwise":
            dyaw_deg += val
        elif sense == "clockwise":
            dyaw_deg -= val

    # --- Gripper action (sticky; defaults to previous or 0.0 for first) ---
    grip_match = grip_pat.search(sentence)
    if grip_match:
        grip_action = float(grip_match.group(1))
    else:
        grip_action = 0.0

    # --- Translation integration (cm -> m), camera frame ---
    v_m = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

    # --- Rotation integration (degrees -> radians), intrinsic XYZ ---
    droll = np.deg2rad(droll_deg)
    dpitch = np.deg2rad(dpitch_deg)
    dyaw = np.deg2rad(dyaw_deg)

    # Build rotation matrices about fixed axes; R = Rz(dyaw) @ Ry(dpitch) @ Rx(droll)
    # (This equals intrinsic XYZ about body axes in the order roll->pitch->yaw.)
    cx, sx = np.cos(droll), np.sin(droll)
    cy, sy = np.cos(dpitch), np.sin(dpitch)
    cz, sz = np.cos(dyaw), np.sin(dyaw)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)

    R = Rz @ Ry @ Rx

    # Convert rotation matrix to axis-angle vector (axis * angle)
    # theta = arccos((trace(R)-1)/2); axis = (1/(2 sin theta)) * [R32-R23, R13-R31, R21-R12]
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    eps = 1e-9
    if theta < 1e-6:
        # For very small rotations, use first-order approximation:
        # skew-symmetric part S ~ [ [0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0] ]
        # where (wx, wy, wz) is the axis-angle vector directly.
        # Extract from R ~ I + S
        wx = 0.5 * (R[2, 1] - R[1, 2])
        wy = 0.5 * (R[0, 2] - R[2, 0])
        wz = 0.5 * (R[1, 0] - R[0, 1])
        axis_angle = np.array([wx, wy, wz], dtype=float)
    else:
        denom = 2.0 * np.sin(theta) + eps
        axis = np.array(
            [
                (R[2, 1] - R[1, 2]) / denom,
                (R[0, 2] - R[2, 0]) / denom,
                (R[1, 0] - R[0, 1]) / denom,
            ],
            dtype=float,
        )
        axis_angle = axis * theta

    # Assemble [dx, dy, dz, ax, ay, az, grip]
    out = np.concatenate([v_m, axis_angle, np.array([grip_action], dtype=float)], axis=0)
    return out


import numpy as np


def _axis_angle_to_quat(w: np.ndarray) -> np.ndarray:
    """Axis-angle vector (so(3)) -> unit quaternion [w, x, y, z]."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = w / theta
    half = 0.5 * theta
    s = np.sin(half)
    return np.array([np.cos(half), *(axis * s)], dtype=float)


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> axis-angle vector (so(3))."""
    # Ensure q[0] >= 0 to keep the short path
    if q[0] < 0:
        q = -q
    w, x, y, z = q
    w = np.clip(w, -1.0, 1.0)
    half = np.arccos(w)
    theta = 2.0 * half
    s = np.sin(half)
    if theta < 1e-12 or s < 1e-12:
        return np.zeros(3, dtype=float)
    axis = np.array([x, y, z], dtype=float) / s
    return axis * theta


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """(w, x, y, z) Hamilton product."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between unit quats q0->q1 at t in [0,1]."""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    # Shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # If very close, fall back to lerp
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return _quat_normalize(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta0 = np.sin(theta0)
    s0 = np.sin((1.0 - t) * theta0) / sin_theta0
    s1 = np.sin(t * theta0) / sin_theta0
    return s0 * q0 + s1 * q1


def interpolate_delta_action_slerp(delta_action: np.ndarray, N: int) -> np.ndarray:
    """
    Split one delta action [dx,dy,dz, ax,ay,az, grip] into N steps.
      - Translation: linear (each step = /N)
      - Rotation: SLERP from Identity to R, then per-step increments are
                  log(R_{k-1}^* R_k) so composition equals the original R
      - Grip (last dim): NOT interpolated; zero for steps 1..N-1, full value on step N

    Returns:
        steps: (N, 7)
    """
    assert delta_action.shape[-1] == 7, "Expected shape (7,) for delta_action"
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive")

    # Split components
    v = np.asarray(delta_action[:3], dtype=float)
    w = np.asarray(delta_action[3:6], dtype=float)  # axis-angle
    grip = float(delta_action[6])

    # Translations: uniform split
    v_step = v / N

    # Rotation target and identity
    qI = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    qT = _axis_angle_to_quat(w)

    steps = np.zeros((N, 7), dtype=float)

    # Precompute cumulative slerp keyframes q_k along I->T
    cum_quats = [qI]
    for k in range(1, N + 1):
        t = k / N
        qk = _quat_slerp(qI, qT, t)
        cum_quats.append(_quat_normalize(qk))

    # Per-step rotational delta: dQ_k = Q_{k-1}^* * Q_k  (right-increment)
    for k in range(1, N + 1):
        q_prev = cum_quats[k - 1]
        q_curr = cum_quats[k]
        dQ = _quat_mul(_quat_conj(q_prev), q_curr)
        w_k = _quat_to_axis_angle(_quat_normalize(dQ))

        steps[k - 1, :3] = v_step
        steps[k - 1, 3:6] = w_k

    # Grip: not interpolated — assign full value on the last step only
    steps[-1, 6] = grip

    # (Optional) exactness check: composition should equal original
    # Sum translations = v; composed rotations product ≈ qT; sum grip = grip.
    return steps


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: ExtendedModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = parse_image(data["observation/image"])
        wrist_image = parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == ExtendedModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # Only return the first 7 dims.
        reasoning = data.get("reasoning")
        action = reasoning_to_action(reasoning)
        actions = interpolate_delta_action_slerp(action, 5)
        return {"actions": actions, "reasoning": reasoning}
