import dataclasses
import re
from typing import Literal

import numpy as np
from openpi import transforms

from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import transform_actions_from_eef_frame


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def euler_to_axisangle(droll: float, dpitch: float, dyaw: float) -> np.ndarray:
    # Build rotation matrices about fixed axes; R = Rz(dyaw) @ Ry(dpitch) @ Rx(droll)
    # (This equals extrinsic XYZ about body axes in the order roll->pitch->yaw.)
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
    return axis_angle


# def reasoning_to_action(
#     sentence: str,
# ) -> np.ndarray:
#     """
#     Invert `describe_movement`.
#     Now supports rotation phrases:
#       - "tilt left/right <deg> degrees"  -> roll (+/-)
#       - "tilt up/down <deg> degrees"     -> pitch (+/-)
#       - "rotate counterclockwise/clockwise <deg> degrees" -> yaw (+/-)
#     Conventions:
#       tilt left  => droll > 0
#       tilt right => droll < 0
#       tilt up    => dpitch > 0
#       tilt down  => dpitch < 0
#       rotate counterclockwise => dyaw > 0
#       rotate clockwise        => dyaw < 0
#     Rotations are applied as intrinsic XYZ (roll→pitch→yaw) increments each step.
#     Translations are interpreted in the camera frame and added directly to the pose.
#     """
#     print(sentence)

#     # Regex patterns
#     move_pat = re.compile(r"move\s+(right|left|forward|backward|up|down)\s+([\-+]?\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
#     grip_pat = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+(?:\.\d+)?)", re.IGNORECASE)

#     tilt_pat = re.compile(r"tilt\s+(left|right|up|down)\s+([\-+]?\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)
#     rot_pat = re.compile(r"rotate\s+(counterclockwise|clockwise)\s+([\-+]?\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)

#     # --- Parse translations in cm along (dx, dy, dz) camera frame ---
#     dx_cm = dy_cm = dz_cm = 0.0
#     for m in move_pat.finditer(sentence):
#         direction = m.group(1).lower()
#         value_cm = float(m.group(2))
#         if direction == "forward":
#             dx_cm += value_cm
#         elif direction == "backward":
#             dx_cm -= value_cm
#         elif direction == "left":
#             dy_cm += value_cm
#         elif direction == "right":
#             dy_cm -= value_cm
#         elif direction == "up":
#             dz_cm += value_cm
#         elif direction == "down":
#             dz_cm -= value_cm

#     # --- Parse rotations in degrees (roll, pitch, yaw) ---
#     droll_deg = dpitch_deg = dyaw_deg = 0.0

#     # tilt: affects roll (left/right) and pitch (up/down)
#     for m in tilt_pat.finditer(sentence):
#         sense = m.group(1).lower()
#         val = float(m.group(2))
#         if sense == "left":
#             droll_deg += val
#         elif sense == "right":
#             droll_deg -= val
#         elif sense == "up":
#             dpitch_deg += val
#         elif sense == "down":
#             dpitch_deg -= val

#     # rotate: affects yaw (ccw/cw)
#     for m in rot_pat.finditer(sentence):
#         sense = m.group(1).lower()
#         val = float(m.group(2))
#         if sense == "counterclockwise":
#             dyaw_deg += val
#         elif sense == "clockwise":
#             dyaw_deg -= val

#     # --- Gripper action (sticky; defaults to previous or 0.0 for first) ---
#     grip_match = grip_pat.search(sentence)
#     if grip_match:
#         grip_action = float(grip_match.group(1))
#     else:
#         grip_action = 0.0

#     # --- Translation integration (cm -> m), camera frame ---
#     v_m = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

#     # --- Rotation integration (degrees -> radians), intrinsic XYZ ---
#     droll = np.deg2rad(droll_deg)
#     dpitch = np.deg2rad(dpitch_deg)
#     dyaw = np.deg2rad(dyaw_deg)

#     # Build rotation matrices about fixed axes; R = Rz(dyaw) @ Ry(dpitch) @ Rx(droll)
#     # (This equals intrinsic XYZ about body axes in the order roll->pitch->yaw.)
#     cx, sx = np.cos(droll), np.sin(droll)
#     cy, sy = np.cos(dpitch), np.sin(dpitch)
#     cz, sz = np.cos(dyaw), np.sin(dyaw)

#     Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
#     Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
#     Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)

#     R = Rz @ Ry @ Rx

#     # Convert rotation matrix to axis-angle vector (axis * angle)
#     # theta = arccos((trace(R)-1)/2); axis = (1/(2 sin theta)) * [R32-R23, R13-R31, R21-R12]
#     trace = np.trace(R)
#     cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
#     theta = np.arccos(cos_theta)

#     eps = 1e-9
#     if theta < 1e-6:
#         # For very small rotations, use first-order approximation:
#         # skew-symmetric part S ~ [ [0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0] ]
#         # where (wx, wy, wz) is the axis-angle vector directly.
#         # Extract from R ~ I + S
#         wx = 0.5 * (R[2, 1] - R[1, 2])
#         wy = 0.5 * (R[0, 2] - R[2, 0])
#         wz = 0.5 * (R[1, 0] - R[0, 1])
#         axis_angle = np.array([wx, wy, wz], dtype=float)
#     else:
#         denom = 2.0 * np.sin(theta) + eps
#         axis = np.array(
#             [
#                 (R[2, 1] - R[1, 2]) / denom,
#                 (R[0, 2] - R[2, 0]) / denom,
#                 (R[1, 0] - R[0, 1]) / denom,
#             ],
#             dtype=float,
#         )
#         axis_angle = axis * theta

#     # Assemble [dx, dy, dz, ax, ay, az, grip]
#     out = np.concatenate([v_m, axis_angle, np.array([grip_action], dtype=float)], axis=0)
#     return out


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
    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, replace the missing ones with zeros.
        base_image = parse_image(data["observation/image"])  # (H, W, C)
        wrist_image = parse_image(data["observation/wrist_image"])  # (H, W, C)
        # Add a singleton time dimension T=1 so downstream expects [B, T, H, W, C]
        base_image_t = np.expand_dims(base_image, axis=0)  # (1, H, W, C)
        wrist_image_t = np.expand_dims(wrist_image, axis=0)  # (1, H, W, C)
        # If the model defines only two image keys, don't add a third entry.

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                IMAGE_KEYS[0]: base_image_t,
                IMAGE_KEYS[1]: wrist_image_t,
            },
            "image_mask": {
                IMAGE_KEYS[0]: np.True_,
                IMAGE_KEYS[1]: np.True_,
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

        # Pass language_actions during training for CoT supervision
        if "language_actions" in data:
            inputs["language_actions"] = data["language_actions"]

        inputs["is_vqa_sample"] = False
        inputs["is_prediction_sample"] = False

        return inputs


@dataclasses.dataclass(frozen=True)
class ActionDecodingSchema:
    """Defines how to decode language actions into numeric actions.

    This corresponds to the LanguageActionFormat used for encoding,
    allowing consistent round-trip conversion between actions and language.
    """

    name: str
    # Style of language action format
    style: Literal["verbose", "compact", "directional_only"] = "verbose"
    # Whether rotation is included
    include_rotation: bool = False
    # Translation unit (cm, m, mm)
    translation_unit: str = "cm"
    # Rotation unit (deg, rad)
    rotation_unit: str = "deg"
    # Schema format (for compact style)
    use_schema_format: bool = False
    # Coordinate frame axis permutation and sign (for camera frame)
    axis_perm: tuple[int, int, int] = (0, 2, 1)
    axis_sign: tuple[int, int, int] = (1, 1, 1)
    # Whether actions are in EEF frame (if True, will transform to base frame)
    use_eef_frame: bool = False

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        in_camera_frame: bool = False,
        initial_state: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse language action(s) into translation deltas, rotation deltas, and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences
            in_camera_frame: Whether the output should be in camera frame coordinates
            initial_state: Initial EEF state for EEF frame transformation (optional)

        Returns:
            (translation_deltas, rotation_deltas, gripper_actions)
            - translation_deltas: array of shape (num_steps, 3) in meters
            - rotation_deltas: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - gripper_actions: array of shape (num_steps,)
        """
        sentences = [reasoning] if isinstance(reasoning, str) else reasoning

        num_steps = len(sentences)
        translations = np.zeros((num_steps, 3), dtype=float)
        rotations = np.zeros((num_steps, 3), dtype=float)  # [roll, pitch, yaw] in radians
        gripper_actions = np.zeros((num_steps,), dtype=float)

        if self.use_schema_format and self.style == "compact":
            # Parse compact schema format
            if self.include_rotation:
                # Format with rotation: <+09 +09 -08 +10 -05 +15 1>
                pattern = re.compile(
                    r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, droll, dpitch, dyaw, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        # Convert degrees to radians
                        rotations[i] = [
                            int(droll) * np.pi / 180.0,
                            int(dpitch) * np.pi / 180.0,
                            int(dyaw) * np.pi / 180.0,
                        ]
                        gripper_actions[i] = float(grip)
            else:
                # Format without rotation: <+09 +09 -08 1>
                pattern = re.compile(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>")
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        gripper_actions[i] = float(grip)
        else:
            # Parse verbose format: "move right X cm and move forward Y cm..."

            if self.style == "directional_only":
                # For directional_only, accept format with optional numeric values
                # e.g., "move right" or "move right 5 cm"
                move_pattern = re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*{self.translation_unit})?",
                    re.IGNORECASE,
                )
            else:
                # For verbose formats, require explicit numeric values
                move_pattern = re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)\s+([\-\d\.]+)\s*{self.translation_unit}",
                    re.IGNORECASE,
                )
            # Rotation pattern for verbose format
            rotation_pattern = re.compile(
                r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
                re.IGNORECASE,
            )

            for i, sentence in enumerate(sentences):
                # Parse movements in language frame (right=+x, forward=+y, up=-z)
                dx_cm = dy_cm = dz_cm = 0.0
                for match in move_pattern.finditer(sentence):
                    direction = match.group(1).lower()
                    # Default to 2cm if no numeric value provided (directional_only mode)
                    value = float(match.group(2)) if match.group(2) is not None else 2.0
                    value *= 1
                    # if direction == "right":
                    #     dx_cm += value
                    # elif direction == "left":
                    #     dx_cm -= value
                    # elif direction == "forward":
                    #     dy_cm += value
                    # elif direction == "backward":
                    #     dy_cm -= value
                    # elif direction == "down":
                    #     dz_cm += value
                    # elif direction == "up":
                    #     dz_cm -= value
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
                        # dz_cm -= 0

                # Convert to meters
                v_m = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

                # Transform to camera or robot frame if needed
                if in_camera_frame:
                    # Invert the axis permutation and sign used in encoding
                    t_cam = np.zeros(3, dtype=float)
                    axis_perm = np.array(self.axis_perm)
                    axis_sign = np.array(self.axis_sign, dtype=float)
                    sign_safe = np.where(axis_sign == 0, 1.0, axis_sign)
                    t_mapped = v_m / sign_safe
                    t_cam[axis_perm] = t_mapped
                    translations[i] = t_cam
                else:
                    translations[i] = v_m

                # Parse rotation actions (if include_rotation is enabled)
                if self.include_rotation:
                    droll_deg = dpitch_deg = dyaw_deg = 0.0
                    for match in rotation_pattern.finditer(sentence):
                        rotation_type = match.group(1).lower()
                        value = float(match.group(2))

                        if rotation_type == "tilt left":
                            droll_deg += value
                        elif rotation_type == "tilt right":
                            droll_deg -= value
                        elif rotation_type in {"tilt down", "tilt back"}:
                            dpitch_deg += value
                        elif rotation_type in {"tilt up", "tilt forward"}:
                            dpitch_deg -= value
                        elif rotation_type == "rotate counterclockwise":
                            dyaw_deg += value
                        elif rotation_type == "rotate clockwise":
                            dyaw_deg -= value

                    # Convert degrees to radians
                    rotations[i] = [
                        droll_deg * np.pi / 180.0,
                        dpitch_deg * np.pi / 180.0,
                        dyaw_deg * np.pi / 180.0,
                    ]

                # Parse gripper action
                grip_pattern = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)
                grip_match = grip_pattern.search(sentence)
                if "open gripper" in sentence.lower():
                    gripper_actions[i] = 1.0
                elif "close gripper" in sentence.lower():
                    gripper_actions[i] = 0.0
                elif grip_match:
                    gripper_actions[i] = float(grip_match.group(1))
                else:
                    gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0
                #     # Maintain previous gripper state
                #     gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0

        # Transform from EEF frame to base frame if needed
        if self.use_eef_frame and initial_state is not None:
            # Combine translations and rotations into action array
            actions = np.concatenate([translations, rotations, gripper_actions[:, None]], axis=1)
            # Transform from EEF frame to base frame
            actions = transform_actions_from_eef_frame(actions, initial_state)
            # Split back into components
            translations = actions[:, :3]
            rotations = actions[:, 3:6]
            gripper_actions = actions[:, 6]

        return translations, rotations, gripper_actions


# Predefined decoding schemas matching language action formats
VERBOSE_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)

DIRECTIONAL_SCHEMA = ActionDecodingSchema(
    name="directional",
    style="directional_only",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)


VERBOSE_EEF_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_eef",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
    use_eef_frame=True,
)

VERBOSE_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_rotation",
    style="verbose",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=False,
)

VERBOSE_EEF_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_eef_rotation",
    style="verbose",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=False,
    use_eef_frame=True,
)


COMPACT_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact",
    style="compact",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_with_rotation",
    style="compact",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_BIMANUAL_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_bimanual",
    style="compact",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_bimanual_with_rotation",
    style="compact",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=True,
)

DECODING_SCHEMA_REGISTRY = {
    "verbose": VERBOSE_DECODING_SCHEMA,
    "verbose_with_rotation": VERBOSE_WITH_ROTATION_DECODING_SCHEMA,
    "compact": COMPACT_DECODING_SCHEMA,
    "compact_with_rotation": COMPACT_WITH_ROTATION_DECODING_SCHEMA,
    "compact_bimanual": COMPACT_BIMANUAL_DECODING_SCHEMA,
    "compact_bimanual_with_rotation": COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA,
    "verbose_eef": VERBOSE_EEF_DECODING_SCHEMA,
    "verbose_eef_with_rotation": VERBOSE_EEF_WITH_ROTATION_DECODING_SCHEMA,
    "directional_only": DIRECTIONAL_SCHEMA,
}


def get_decoding_schema(name: str) -> ActionDecodingSchema:
    """Get an action decoding schema by name."""
    if name not in DECODING_SCHEMA_REGISTRY:
        raise ValueError(f"Unknown decoding schema: {name}. Available schemas: {list(DECODING_SCHEMA_REGISTRY.keys())}")
    return DECODING_SCHEMA_REGISTRY[name]


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    # Optional decoding schema for parsing language actions to numeric actions
    decoding_schema: ActionDecodingSchema | str | None = None
    # Whether decoded actions should be in camera frame
    in_camera_frame: bool = False
    # Interpolatation steps

    def __post_init__(self):
        """Resolve string schema name to ActionDecodingSchema instance."""
        if isinstance(self.decoding_schema, str):
            schema = get_decoding_schema(self.decoding_schema)
            object.__setattr__(self, "decoding_schema", schema)

    def __call__(self, data: dict) -> dict:
        # Get actions and reasoning from data

        if "reasoning" not in data:
            return {"actions": np.asarray(data["actions"][:, :7]), "reasoning": None}
        reasoning = data.get("reasoning")

        # If decoding schema is provided and we have reasoning, parse it to get actions
        assert self.decoding_schema is not None
        assert reasoning is not None

        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.decoding_schema.use_eef_frame and "state" in data:
            initial_state = np.asarray(data["state"])

        # Parse reasoning to translation deltas, rotation deltas, and gripper actions
        translations, rotations, gripper_actions = self.decoding_schema.parse_language_to_deltas(
            reasoning, in_camera_frame=self.in_camera_frame, initial_state=initial_state
        )
        assert rotations.shape[0] == 1
        rotations = euler_to_axisangle(rotations[0, 0], rotations[0, 1], rotations[0, 2])[None, :]

        # If we don't have actions from the model, use the parsed actions
        # Shape: (num_steps, 7) -> [dx, dy, dz, droll, dpitch, dyaw, gripper]
        parsed_actions = np.concatenate(
            [
                translations,  # (num_steps, 3)
                rotations,  # (num_steps, 3) - rotation deltas in radians
                gripper_actions[:, None],  # (num_steps, 1)
            ],
            axis=1,
        )[0]

        parsed_actions = interpolate_delta_action_slerp(parsed_actions, 5)

        # Store parsed actions separately for inspection
        data["parsed_actions"] = parsed_actions
        print(reasoning, parsed_actions)

        return {"actions": parsed_actions, "reasoning": reasoning}


# @dataclasses.dataclass(frozen=True)
# class LiberoOutputs(transforms.DataTransformFn):
#     """
#     This class is used to convert outputs from the model back the the dataset specific format. It is
#     used for inference only.

#     For your own dataset, you can copy this class and modify the action dimension based on the comments below.
#     """

#     def __call__(self, data: dict) -> dict:
#         # Only return the first N actions -- since we padded actions above to fit the model action
#         # dimension, we need to now parse out the correct number of actions in the return dict.
#         # For Libero, we only return the first 7 actions (since the rest is padding).
#         # For your own dataset, replace `7` with the action dimension of your dataset.
#         # Only return the first 7 dims.
#         reasoning = data.get("reasoning")
#         action = reasoning_to_action(reasoning)
#         actions = interpolate_delta_action_slerp(action, 5)
#         return {"actions": actions, "reasoning": reasoning}
