import numpy as np
import tensorflow as tf


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
