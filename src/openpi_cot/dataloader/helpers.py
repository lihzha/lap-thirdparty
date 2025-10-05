from enum import Enum
from enum import IntEnum

import numpy as np
import tensorflow as tf


# Note: Both DROID and OXE use roll-pitch-yaw convention.
# Note: quaternion is in xyzw order.
# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    NONE = -1  # No Proprioceptive State
    POS_EULER = 1  # EEF XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)  Note: no <PAD>
    POS_QUAT = 2  # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3  # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4  # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    EEF_R6 = 5  # EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    EEF_POS = 1  # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1).
    JOINT_POS = 2  # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4  # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    ABS_EEF_POS = 5  # EEF Absolute XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


def euler_xyz_to_rot(rx, ry, rz):
    # Build rotation matrix from XYZ intrinsic rotations
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def extract_episode_path_from_file_path(file_path):
    """Extract episode path from a full file path using regex.

    Removes everything up to and including 'r2d2-data/' or
    'r2d2-data-full/', then trims anything from '/trajectory' onwards.
    """
    # Strip dataset prefix up to r2d2-data or r2d2-data-full
    rel = tf.strings.regex_replace(
        file_path,
        r"^.*r2d2-data(?:-full)?/",
        "",
    )
    # Remove trailing '/trajectory...' suffix
    episode_path = tf.strings.regex_replace(
        rel,
        r"/trajectory.*$",
        "",
    )
    return episode_path


def project_in_bounds(xyz, intr4, extr44):
    xyz = tf.cast(xyz, tf.float32)
    intr4 = tf.cast(intr4, tf.float32)
    extr44 = tf.cast(extr44, tf.float32)
    # xyz: [N,3], intr4: [N,4], extr44: [N,4,4]
    # Compute camera coordinates
    ones = tf.ones_like(xyz[..., :1], dtype=tf.float32)
    p_base = tf.concat([xyz, ones], axis=-1)  # [N,4]
    base_to_cam = tf.linalg.inv(extr44)
    p_cam = tf.einsum("nij,nj->ni", base_to_cam, p_base)
    z = p_cam[..., 2]
    fx = intr4[..., 0]
    fy = intr4[..., 1]
    cx = intr4[..., 2]
    cy = intr4[..., 3]
    valid = tf.logical_and(
        z > tf.constant(1e-6, tf.float32),
        tf.logical_and(fx > 0.0, fy > 0.0),
    )
    # Pixel at calibration resolution
    u = fx * (p_cam[..., 0] / z) + cx
    v = fy * (p_cam[..., 1] / z) + cy
    # Letterbox to 224x224 using same math as resize_with_pad
    Wt = tf.constant(224.0, dtype=tf.float32)
    Ht = tf.constant(224.0, dtype=tf.float32)
    Wc = tf.maximum(tf.constant(1.0, tf.float32), 2.0 * cx)
    Hc = tf.maximum(tf.constant(1.0, tf.float32), 2.0 * cy)
    ratio = tf.maximum(Wc / Wt, Hc / Ht)
    resized_w = Wc / ratio
    resized_h = Hc / ratio
    pad_w0 = (Wt - resized_w) / 2.0
    pad_h0 = (Ht - resized_h) / 2.0
    x = u * (resized_w / Wc) + pad_w0
    y = v * (resized_h / Hc) + pad_h0
    in_x = tf.logical_and(
        x >= tf.constant(0.0, tf.float32),
        x <= (Wt - tf.constant(1.0, tf.float32)),
    )
    in_y = tf.logical_and(
        y >= tf.constant(0.0, tf.float32),
        y <= (Ht - tf.constant(1.0, tf.float32)),
    )
    return tf.logical_and(valid, tf.logical_and(in_x, in_y))


def convert_state_encoding(state: tf.Tensor, from_encoding: StateEncoding, to_encoding: StateEncoding) -> tf.Tensor:
    """
    Convert state representation between different encodings.

    Args:
        state: Input state tensor
        from_encoding: Source encoding type
        to_encoding: Target encoding type

    Returns:
        Converted state tensor
    """
    if from_encoding == to_encoding:
        return state

    # Handle conversions between POS_EULER, POS_QUAT, and EEF_R6
    if from_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT} and to_encoding in {
        StateEncoding.POS_EULER,
        StateEncoding.POS_QUAT,
    }:
        return _convert_pos_euler_quat(state, from_encoding, to_encoding)
    if from_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT} and to_encoding == StateEncoding.EEF_R6:
        return _convert_pos_to_eef_r6(state, from_encoding)
    if from_encoding == StateEncoding.EEF_R6 and to_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT}:
        return _convert_eef_r6_to_pos(state, to_encoding)
    # raise ValueError(f"Unsupported state encoding conversion: {from_encoding} -> {to_encoding}")
    # logging.warning(f"Unsupported state encoding conversion: {from_encoding} -> {to_encoding}")
    return state


def convert_action_encoding(
    action: tf.Tensor, from_encoding: ActionEncoding, to_encoding: ActionEncoding, to_delta_cartesian_pose: bool = False
) -> tf.Tensor:
    """
    Convert action representation between different encodings.

    Args:
        action: Input action tensor
        from_encoding: Source encoding type
        to_encoding: Target encoding type

    Returns:
        Converted action tensor
    """
    if to_delta_cartesian_pose:
        action = _convert_abs_eef_pose_to_delta_eef_pose(action)
    if from_encoding == to_encoding:
        return action

    # Handle conversions between EEF_POS and EEF_R6
    if (from_encoding in (ActionEncoding.ABS_EEF_POS, ActionEncoding.EEF_POS)) and to_encoding == ActionEncoding.EEF_R6:
        return _convert_eef_pos_to_eef_r6(action)
    if from_encoding == ActionEncoding.EEF_R6 and to_encoding in (ActionEncoding.ABS_EEF_POS, ActionEncoding.EEF_POS):
        return _convert_eef_r6_to_eef_pos(action)
    if from_encoding == ActionEncoding.ABS_EEF_POS and to_encoding == ActionEncoding.EEF_POS:
        return action
    if from_encoding == ActionEncoding.EEF_POS and to_encoding == ActionEncoding.ABS_EEF_POS:
        return action
    raise ValueError(f"Unsupported action encoding conversion: {from_encoding} -> {to_encoding}")


def _convert_abs_eef_pose_to_delta_eef_pose(action: tf.Tensor) -> tf.Tensor:
    """Convert absolute EEF pose [pos(3), euler(3)] to delta EEF pose.
    Assumes action shape [..., 6] = [x,y,z,roll,pitch,yaw].
    Pads last timestep with zeros so output has same shape as input.
    Works in TF graph mode.
    """

    # Positions: simple difference
    delta_pos = action[1:, ..., :3] - action[:-1, ..., :3]

    # Euler angles: use tf.atan2(sin, cos) to handle wrap-around correctly
    e1 = action[1:, ..., 3:6]
    e2 = action[:-1, ..., 3:6]
    raw_delta = e1 - e2
    delta_euler = tf.atan2(tf.sin(raw_delta), tf.cos(raw_delta))  # wrap to [-pi, pi]

    # Concatenate delta position + delta euler
    delta_eef = tf.concat([delta_pos, delta_euler], axis=-1)

    # Pad last timestep with zeros so output shape == input shape
    pad_shape = tf.concat([[1], tf.shape(delta_eef)[1:]], axis=0)
    last_zero = tf.zeros(pad_shape, dtype=delta_eef.dtype)
    delta_eef = tf.concat([delta_eef, last_zero], axis=0)
    delta_eef = tf.concat([delta_eef, action[..., -1:]], axis=-1)

    return delta_eef


def _convert_pos_euler_quat(state: tf.Tensor, from_encoding: StateEncoding, to_encoding: StateEncoding) -> tf.Tensor:
    """Convert between POS_EULER and POS_QUAT encodings."""
    if from_encoding == StateEncoding.POS_EULER and to_encoding == StateEncoding.POS_QUAT:
        # POS_EULER: [x, y, z, rx, ry, rz, gripper] -> POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper]
        xyz = state[..., :3]  # [..., 3]
        euler = state[..., 3:6]  # [..., 3] - rx, ry, rz
        gripper = state[..., -1:]  # [..., 1]

        # Convert euler angles to quaternion
        quat = _euler_to_quaternion(euler)

        return tf.concat([xyz, quat, gripper], axis=-1)

    if from_encoding == StateEncoding.POS_QUAT and to_encoding == StateEncoding.POS_EULER:
        # POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper] -> POS_EULER: [x, y, z, rx, ry, rz, gripper]
        xyz = state[..., :3]  # [..., 3]
        quat = state[..., 3:7]  # [..., 4] - qx, qy, qz, qw
        gripper = state[..., -1:]  # [..., 1]

        # Convert quaternion to euler angles
        euler = _quaternion_to_euler(quat)

        return tf.concat([xyz, euler, gripper], axis=-1)

    raise ValueError(f"Unsupported conversion: {from_encoding} -> {to_encoding}")


def _convert_pos_to_eef_r6(state: tf.Tensor, from_encoding: StateEncoding) -> tf.Tensor:
    """Convert POS_EULER or POS_QUAT to EEF_R6 encoding."""
    dtype = state.dtype
    if from_encoding == StateEncoding.POS_EULER:
        # POS_EULER: [x, y, z, rx, ry, rz, gripper] -> EEF_R6: [x, y, z, r11, r12, r13, r21, r22, r23, gripper]
        xyz = state[..., :3]  # [..., 3]
        euler = state[..., 3:6]  # [..., 3] - rx, ry, rz
        gripper = state[..., -1:]  # [..., 1]

        # Convert euler angles to rotation matrix (first 6 elements of 3x3 matrix)
        rot_matrix = _euler_to_rotation_matrix(euler)  # [..., 3, 3]
        r6 = _rotation_matrix_to_r6(rot_matrix)

        return tf.concat([tf.cast(xyz, dtype), tf.cast(r6, dtype), tf.cast(gripper, dtype)], axis=-1)

    if from_encoding == StateEncoding.POS_QUAT:
        # POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper] -> EEF_R6: [x, y, z, r11, r12, r13, r21, r22, r23, gripper]
        xyz = state[..., :3]  # [..., 3]
        quat = state[..., 3:7]  # [..., 4] - qx, qy, qz, qw
        gripper = state[..., -1:]  # [..., 1]

        # Convert quaternion to rotation matrix (first 6 elements of 3x3 matrix)
        rot_matrix = _quaternion_to_rotation_matrix(quat)  # [..., 3, 3]
        r6 = _rotation_matrix_to_r6(rot_matrix)

        return tf.concat([tf.cast(xyz, dtype), tf.cast(r6, dtype), tf.cast(gripper, dtype)], axis=-1)

    raise ValueError(f"Unsupported conversion from {from_encoding} to EEF_R6")


def _convert_eef_r6_to_eef_pos(action: tf.Tensor) -> tf.Tensor:
    """Convert EEF_R6 action to EEF_POS action."""
    # EEF_R6: [dx, dy, dz, dr11, dr12, dr13, dr21, dr22, dr23, gripper] -> EEF_POS: [dx, dy, dz, drx, dry, drz, gripper]
    xyz_delta = action[..., :3]  # [..., 3]
    r6_delta = action[..., 3:9]  # [..., 6] - dr11, dr12, dr13, dr21, dr22, dr23
    gripper = action[..., -1:]  # [..., 1]

    rot_matrix = _r6_to_rotation_matrix(r6_delta)

    # Convert rotation matrix to euler angles
    euler_delta = _rotation_matrix_to_euler(rot_matrix)

    return tf.concat([xyz_delta, euler_delta, gripper], axis=-1)


def _convert_eef_r6_to_pos(state: tf.Tensor, to_encoding: StateEncoding) -> tf.Tensor:
    """Convert EEF_R6 encoding to POS_EULER or POS_QUAT."""
    xyz = state[..., :3]  # [..., 3]
    r6 = state[..., 3:9]  # [..., 6] - r11, r12, r13, r21, r22, r23
    gripper = state[..., -1:]  # [..., 1]

    rot_matrix = _r6_to_rotation_matrix(r6)

    if to_encoding == StateEncoding.POS_EULER:
        # Convert rotation matrix to euler angles
        euler = _rotation_matrix_to_euler(rot_matrix)
        return tf.concat([xyz, euler, gripper], axis=-1)

    if to_encoding == StateEncoding.POS_QUAT:
        # Convert rotation matrix to quaternion
        quat = _rotation_matrix_to_quaternion(rot_matrix)
        return tf.concat([xyz, quat, gripper], axis=-1)

    raise ValueError(f"Unsupported conversion from EEF_R6 to {to_encoding}")


def _convert_eef_pos_to_eef_r6(action: tf.Tensor) -> tf.Tensor:
    """Convert EEF_POS action to EEF_R6 action."""
    # EEF_POS: [dx, dy, dz, drx, dry, drz, gripper] -> EEF_R6: [dx, dy, dz, dr11, dr12, dr13, dr21, dr22, dr23, gripper]
    xyz_delta = action[..., :3]  # [..., 3]
    euler_delta = action[..., 3:6]  # [..., 3] - drx, dry, drz
    gripper = action[..., -1:]  # [..., 1]

    dtype = action.dtype

    # Convert euler angle deltas to rotation matrix deltas
    rot_delta = _euler_to_rotation_matrix(euler_delta)  # [..., 3, 3]
    r6_delta = _rotation_matrix_to_r6(rot_delta)

    return tf.concat([tf.cast(xyz_delta, dtype), tf.cast(r6_delta, dtype), tf.cast(gripper, dtype)], axis=-1)


# Helper functions for quaternion and rotation matrix conversions


def _euler_to_quaternion(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles (rx, ry, rz) to quaternion (qx, qy, qz, qw)."""
    rx, ry, rz = tf.unstack(euler, axis=-1)

    # Half angles
    cx, sx = tf.cos(rx * 0.5), tf.sin(rx * 0.5)
    cy, sy = tf.cos(ry * 0.5), tf.sin(ry * 0.5)
    cz, sz = tf.cos(rz * 0.5), tf.sin(rz * 0.5)

    # Quaternion components
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    return tf.stack([qx, qy, qz, qw], axis=-1)


def _euler_to_rotation_matrix(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles (rx, ry, rz) to rotation matrix."""
    rx, ry, rz = tf.unstack(euler, axis=-1)

    cx, sx = tf.cos(rx), tf.sin(rx)
    cy, sy = tf.cos(ry), tf.sin(ry)
    cz, sz = tf.cos(rz), tf.sin(rz)

    # Create rotation matrices with proper batch dimensions
    # Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    # Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    # Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

    # Stack to create [batch, 3, 3] matrices
    ones = tf.ones_like(rx)
    zeros = tf.zeros_like(rx)

    Rx = tf.stack(
        [
            tf.stack([ones, zeros, zeros], axis=-1),
            tf.stack([zeros, cx, -sx], axis=-1),
            tf.stack([zeros, sx, cx], axis=-1),
        ],
        axis=-2,
    )

    Ry = tf.stack(
        [
            tf.stack([cy, zeros, sy], axis=-1),
            tf.stack([zeros, ones, zeros], axis=-1),
            tf.stack([-sy, zeros, cy], axis=-1),
        ],
        axis=-2,
    )

    Rz = tf.stack(
        [
            tf.stack([cz, -sz, zeros], axis=-1),
            tf.stack([sz, cz, zeros], axis=-1),
            tf.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    # Combine rotations: R = Rz * Ry * Rx (intrinsic XYZ)
    return tf.linalg.matmul(tf.linalg.matmul(Rz, Ry), Rx)


def _quaternion_to_euler(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (qx, qy, qz, qw) to euler angles (rx, ry, rz) - XYZ intrinsic."""
    qx, qy, qz, qw = tf.unstack(quat, axis=-1)

    # Normalize quaternion
    norm = tf.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Convert to euler angles (XYZ intrinsic)
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    rx = tf.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = tf.clip_by_value(sinp, -1.0, 1.0)
    ry = tf.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    rz = tf.atan2(siny_cosp, cosy_cosp)

    return tf.stack([rx, ry, rz], axis=-1)


def _quaternion_to_rotation_matrix(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (qx, qy, qz, qw) to rotation matrix."""
    eps = tf.constant(1e-8, quat.dtype)
    qx, qy, qz, qw = tf.unstack(quat, axis=-1)
    norm = tf.sqrt(tf.maximum(qw**2 + qx**2 + qy**2 + qz**2, eps))
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Rotation matrix elements
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx * qy - qw * qz)
    r13 = 2 * (qx * qz + qw * qy)

    r21 = 2 * (qx * qy + qw * qz)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy * qz - qw * qx)

    r31 = 2 * (qx * qz - qw * qy)
    r32 = 2 * (qy * qz + qw * qx)
    r33 = 1 - 2 * (qx**2 + qy**2)

    return tf.stack(
        [tf.stack([r11, r12, r13], axis=-1), tf.stack([r21, r22, r23], axis=-1), tf.stack([r31, r32, r33], axis=-1)],
        axis=-2,
    )


def _rotation_matrix_to_euler(rot_matrix: tf.Tensor) -> tf.Tensor:
    r11, r12, r13 = tf.unstack(rot_matrix[..., 0, :], axis=-1)
    r21, r22, r23 = tf.unstack(rot_matrix[..., 1, :], axis=-1)
    r31, r32, r33 = tf.unstack(rot_matrix[..., 2, :], axis=-1)

    rx = tf.atan2(r32, r33)
    ry = tf.asin(tf.clip_by_value(-r31, -1.0, 1.0))
    rz = tf.atan2(r21, r11)
    return tf.stack([rx, ry, rz], axis=-1)


def _rotation_matrix_to_quaternion(rot_matrix: tf.Tensor) -> tf.Tensor:
    eps = tf.constant(1e-8, rot_matrix.dtype)
    r11, r12, r13 = tf.unstack(rot_matrix[..., 0, :], axis=-1)
    r21, r22, r23 = tf.unstack(rot_matrix[..., 1, :], axis=-1)
    r31, r32, r33 = tf.unstack(rot_matrix[..., 2, :], axis=-1)

    trace = r11 + r22 + r33
    cond1 = trace > 0
    cond2 = tf.logical_and(r11 > r22, r11 > r33)
    cond3 = r22 > r33

    s1 = tf.sqrt(tf.maximum(trace + 1.0, eps)) * 2.0
    s2 = tf.sqrt(tf.maximum(1.0 + r11 - r22 - r33, eps)) * 2.0
    s3 = tf.sqrt(tf.maximum(1.0 + r22 - r11 - r33, eps)) * 2.0
    s4 = tf.sqrt(tf.maximum(1.0 + r33 - r11 - r22, eps)) * 2.0

    qw1 = 0.25 * s1
    qx1 = (r32 - r23) / (s1 + eps)
    qy1 = (r13 - r31) / (s1 + eps)
    qz1 = (r21 - r12) / (s1 + eps)

    qw2 = (r32 - r23) / (s2 + eps)
    qx2 = 0.25 * s2
    qy2 = (r12 + r21) / (s2 + eps)
    qz2 = (r13 + r31) / (s2 + eps)

    qw3 = (r13 - r31) / (s3 + eps)
    qx3 = (r12 + r21) / (s3 + eps)
    qy3 = 0.25 * s3
    qz3 = (r23 + r32) / (s3 + eps)

    qw4 = (r21 - r12) / (s4 + eps)
    qx4 = (r13 + r31) / (s4 + eps)
    qy4 = (r23 + r32) / (s4 + eps)
    qz4 = 0.25 * s4

    qw = tf.where(cond1, qw1, tf.where(cond2, qw2, tf.where(cond3, qw3, qw4)))
    qx = tf.where(cond1, qx1, tf.where(cond2, qx2, tf.where(cond3, qx3, qx4)))
    qy = tf.where(cond1, qy1, tf.where(cond2, qy2, tf.where(cond3, qy3, qy4)))
    qz = tf.where(cond1, qz1, tf.where(cond2, qz2, tf.where(cond3, qz3, qz4)))

    # Final normalize to kill residual numeric drift
    q = tf.stack([qx, qy, qz, qw], axis=-1)
    q = q / (tf.norm(q, axis=-1, keepdims=True) + eps)
    return q


def _rotation_matrix_to_r6(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Flatten the first two rotation matrix rows into the 6D (R6) representation."""
    upper_two_rows = rot_matrix[..., :2, :]  # [..., 2, 3]
    return tf.reshape(upper_two_rows, tf.concat([tf.shape(rot_matrix)[:-2], [6]], axis=0))


def _r6_to_rotation_matrix(r6: tf.Tensor) -> tf.Tensor:
    """Reconstruct an orthonormal rotation matrix from the 6D (R6) representation."""
    dtype = r6.dtype
    eps = tf.constant(1e-8, dtype)

    def _normalize(vec: tf.Tensor) -> tf.Tensor:
        norm = tf.maximum(tf.norm(vec, axis=-1, keepdims=True), eps)
        return vec / norm

    r1 = r6[..., :3]
    r2 = r6[..., 3:]

    r1 = _normalize(r1)
    # Remove component of r2 along r1 before normalizing again (Gram-Schmidt)
    r2 = r2 - tf.reduce_sum(r2 * r1, axis=-1, keepdims=True) * r1
    r2 = _normalize(r2)

    r3 = tf.linalg.cross(r1, r2)
    r3 = _normalize(r3)

    return tf.stack([r1, r2, r3], axis=-2)
