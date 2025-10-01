import numpy as np
import tensorflow as tf

# Fixed frame transform: x'=-y, y'=-x, z'=-z
_C = tf.constant([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=tf.float32)


@tf.function
def _rot_x(a):
    ca, sa = tf.cos(a), tf.sin(a)
    z = tf.zeros_like(a)
    o = tf.ones_like(a)
    return tf.stack(
        [
            tf.stack([o, z, z], axis=-1),
            tf.stack([z, ca, -sa], axis=-1),
            tf.stack([z, sa, ca], axis=-1),
        ],
        axis=-2,
    )


@tf.function
def _rot_y(a):
    ca, sa = tf.cos(a), tf.sin(a)
    z = tf.zeros_like(a)
    o = tf.ones_like(a)
    return tf.stack(
        [
            tf.stack([ca, z, sa], axis=-1),
            tf.stack([z, o, z], axis=-1),
            tf.stack([-sa, z, ca], axis=-1),
        ],
        axis=-2,
    )


@tf.function
def _rot_z(a):
    ca, sa = tf.cos(a), tf.sin(a)
    z = tf.zeros_like(a)
    o = tf.ones_like(a)
    return tf.stack(
        [
            tf.stack([ca, -sa, z], axis=-1),
            tf.stack([sa, ca, z], axis=-1),
            tf.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def _R_from_euler_xyz(angles):
    """Intrinsic XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    roll, pitch, yaw = angles[..., 0], angles[..., 1], angles[..., 2]
    return tf.linalg.matmul(tf.linalg.matmul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


def _euler_xyz_from_R(R, eps=1e-6):
    """
    Extract intrinsic XYZ (roll, pitch, yaw) from rotation matrix R.
    Handles gimbal lock with tf.where (graph-safe).
    """
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r11 = R[..., 1, 1]

    cond = tf.less(tf.abs(r20), 1.0 - tf.convert_to_tensor(eps, R.dtype))

    pitch_reg = tf.asin(tf.clip_by_value(-r20, -1.0, 1.0))
    roll_reg = tf.atan2(r21, r22)
    yaw_reg = tf.atan2(r10, r00)

    # Gimbal lock: |cos(pitch)| ~ 0
    pitch_gl = (np.pi / 2) * tf.sign(-r20)
    roll_gl = tf.zeros_like(pitch_gl)
    yaw_gl = tf.where(r20 < 0.0, tf.atan2(-r01, r11), tf.atan2(r01, r11))

    roll = tf.where(cond, roll_reg, roll_gl)
    pitch = tf.where(cond, pitch_reg, pitch_gl)
    yaw = tf.where(cond, yaw_reg, yaw_gl)
    return tf.stack([roll, pitch, yaw], axis=-1)


@tf.function
def zxy_to_xyz_tf(angles, degrees=False, eps=1e-6):
    """
    Convert intrinsic Z-X-Y Euler angles to intrinsic X-Y-Z Euler angles.
    angles: tensor of shape (..., 3) -> (az, ax, ay) in radians by default.
    returns: tensor of shape (..., 3) -> (roll[x], pitch[y], yaw[z]) in same unit.
    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if degrees:
        angles = tf.math.multiply(angles, np.pi / 180.0)

    az = angles[..., 0]  # rotate about z
    ax = angles[..., 1]  # then about x
    ay = angles[..., 2]  # then about y

    # Build rotation for intrinsic "zxy": R = Rz(az) @ Rx(ax) @ Ry(ay)
    Rz = _rot_z(az)
    Rx = _rot_x(ax)
    Ry = _rot_y(ay)
    R = tf.linalg.matmul(tf.linalg.matmul(Rz, Rx), Ry)

    # Elements we need
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r11 = R[..., 1, 1]

    # Regular branch (no gimbal lock): |r20| < 1 - eps
    cond = tf.less(tf.abs(r20), 1.0 - tf.convert_to_tensor(eps, R.dtype))
    theta_reg = tf.asin(tf.clip_by_value(-r20, -1.0, 1.0))  # pitch (y)
    phi_reg = tf.atan2(r21, r22)  # roll  (x)
    psi_reg = tf.atan2(r10, r00)  # yaw   (z)

    # Gimbal lock branch: |cos(theta)| ~ 0  -> theta = ±pi/2, set roll=0, solve yaw
    theta_gl = (np.pi / 2) * tf.sign(-r20)
    phi_gl = tf.zeros_like(theta_gl)
    # If r20 < 0 (theta ≈ +pi/2):  psi = atan2(-r01, r11)
    # Else (theta ≈ -pi/2):       psi = atan2( r01, r11)
    psi_gl = tf.where(r20 < 0.0, tf.atan2(-r01, r11), tf.atan2(r01, r11))

    # Select per element
    phi = tf.where(cond, phi_reg, phi_gl)
    theta = tf.where(cond, theta_reg, theta_gl)
    psi = tf.where(cond, psi_reg, psi_gl)

    out = tf.stack([phi, theta, psi], axis=-1)
    if degrees:
        out = tf.math.multiply(out, 180.0 / np.pi)
    return out


def euler_diff(angles1, angles2, order="xyz", degrees=False):
    """
    Compute relative Euler angle difference: angles_rel such that
        R(angles2) * R(angles_rel) = R(angles1)

    Args:
        angles1: (..., 3) tensor of Euler angles [a1, a2, a3]
        angles2: (..., 3) tensor of Euler angles [a1, a2, a3]
        order:   rotation order string, e.g. "xyz" (intrinsic)
        degrees: whether input/output are in degrees
    Returns:
        (..., 3) tensor of relative Euler angles (same order)
    """
    if degrees:
        angles1 = tf.math.multiply(angles1, np.pi / 180.0)
        angles2 = tf.math.multiply(angles2, np.pi / 180.0)

    # map axis char -> rotation fn
    rot_map = {"x": _rot_x, "y": _rot_y, "z": _rot_z}

    def build_R(angles):
        R = None
        for i, ax in enumerate(order):
            Ri = rot_map[ax](angles[..., i])
            R = Ri if R is None else tf.linalg.matmul(R, Ri)
        return R

    R1 = build_R(angles1)
    R2 = build_R(angles2)

    Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)  # Rrel = R2^T * R1

    # Extract back Euler from Rrel, here only for "xyz" (extendable)
    r20 = Rrel[..., 2, 0]
    r21 = Rrel[..., 2, 1]
    r22 = Rrel[..., 2, 2]
    r10 = Rrel[..., 1, 0]
    r00 = Rrel[..., 0, 0]

    theta = tf.asin(-r20)
    phi = tf.atan2(r21, r22)
    psi = tf.atan2(r10, r00)

    out = tf.stack([phi, theta, psi], axis=-1)
    if degrees:
        out = tf.math.multiply(out, 180.0 / np.pi)
    return out


def axis_angle_to_euler(axis_angle):
    """Convert axis-angle to euler angles."""
    import tensorflow_graphics.geometry.transformation as tft

    euler = tft.euler.from_axis_angle(axis_angle)
    return euler


def transform_actions_xyz(movement_actions):
    """
    movement_actions: (..., 6) where [:3] = translation deltas (xyz),
                      [3:6] = Euler deltas in intrinsic XYZ (roll,pitch,yaw).
    Returns transformed actions under x'=-y, y'=-x, z'=-z.
    """
    t = movement_actions[..., :3]
    e = movement_actions[..., 3:6]

    # translations: t' = C t
    t_prime = tf.einsum("ij,...j->...i", _C, t)

    # rotations: R' = C R C^T, then back to Euler XYZ
    R = _R_from_euler_xyz(e)
    R_prime = tf.linalg.matmul(tf.linalg.matmul(_C, R), tf.transpose(_C))
    e_prime = _euler_xyz_from_R(R_prime)

    return tf.concat([t_prime, e_prime], axis=-1)
