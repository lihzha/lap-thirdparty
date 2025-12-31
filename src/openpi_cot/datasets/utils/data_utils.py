"""
data_utils.py

Additional RLDS-specific data utilities.
"""

from collections.abc import Callable
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import dlimp as dl
import numpy as np
import tensorflow as tf

from openpi_cot.datasets.utils.configs import OXE_DATASET_CONFIGS
from openpi_cot.datasets.utils.configs import ActionEncoding
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.transforms import OXE_STANDARDIZATION_TRANSFORMS


def _tf_pi(dtype):
    return tf.constant(3.141592653589793, dtype=dtype)


def tree_map(fn: Callable, tree: dict) -> dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: dict) -> dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    if tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


# === State / Action Processing Primitives ===


def normalize_action_and_proprio(
    traj: dict,
    norm_stats=None,
    normalization_type: NormalizationType | str = NormalizationType.NORMAL,
    action_key: str = "action",
    state_key: str = "proprio",
    metadata=None,  # Back-compat: some callers pass `metadata` instead of `norm_stats`
):
    """Normalizes the action and proprio fields of a trajectory using provided stats.

    Accepts either `norm_stats` or `metadata` for backwards compatibility. Supports
    stats structures with keys "actions" or "action", and values that may be
    dicts, dataclass-like objects (with attributes), lists, NumPy arrays or
    tensors. All stats are coerced to tf.Tensor before use.
    """

    # Back-compat argument name
    if norm_stats is None and metadata is not None:
        norm_stats = metadata

    if norm_stats is None:
        return traj

    # Normalize enum input
    if isinstance(normalization_type, str):
        try:
            normalization_type = NormalizationType(normalization_type)
        except ValueError:
            raise ValueError(f"Unknown Normalization Type {normalization_type}")

    def _get_group(stats_root, group_name: str):
        # support both "actions" and "action"
        if isinstance(stats_root, dict):
            group = stats_root.get(group_name)
            if group is None and group_name.endswith("s"):
                group = stats_root.get(group_name[:-1])
            return group
        # if a non-dict is provided, there's nothing to select
        return None

    def _get_value(group_stats, key: str):
        if group_stats is None:
            return None
        if isinstance(group_stats, dict):
            value = group_stats.get(key)
        else:
            # dataclass-like objects
            value = getattr(group_stats, key, None)
        if value is None:
            return None
        # Coerce to tf.Tensor[float32]
        if isinstance(value, tf.Tensor):
            return tf.cast(value, tf.float32)
        return tf.convert_to_tensor(value, dtype=tf.float32)

    def normal(x, mean, std):
        return (x - mean) / (std + 1e-6)

    def bounds(x, _min, _max):
        return tf.clip_by_value(2 * (x - _min) / (_max - _min + 1e-8) - 1, -1, 1)

    actions_stats = _get_group(norm_stats, "actions")
    state_stats = _get_group(norm_stats, "state")

    if normalization_type == NormalizationType.NORMAL:
        a_mean = _get_value(actions_stats, "mean")
        a_std = _get_value(actions_stats, "std")
        s_mean = _get_value(state_stats, "mean")
        s_std = _get_value(state_stats, "std")

        if a_mean is not None and a_std is not None:
            traj[action_key] = normal(traj[action_key], a_mean, a_std)
        if s_mean is not None and s_std is not None and state_key in traj.get("observation", {}):
            traj["observation"][state_key] = normal(traj["observation"][state_key], s_mean, s_std)
    elif normalization_type in (NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99):
        low_key = "min" if normalization_type == NormalizationType.BOUNDS else "q01"
        high_key = "max" if normalization_type == NormalizationType.BOUNDS else "q99"

        action_low = _get_value(actions_stats, low_key)
        action_high = _get_value(actions_stats, high_key)
        state_low = _get_value(state_stats, low_key)
        state_high = _get_value(state_stats, high_key)

        if action_low is not None and action_high is not None:
            traj[action_key] = bounds(traj[action_key], action_low, action_high)
            zeros_mask = tf.equal(action_low, action_high)
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == action_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        if state_low is not None and state_high is not None and state_key in traj.get("observation", {}):
            traj["observation"][state_key] = bounds(traj["observation"][state_key], state_low, state_high)
            zeros_mask = tf.equal(state_low, state_high)
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == state_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

    return traj

    # for key, traj_key in keys_to_normalize.items():
    #     mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
    #     traj = dl.transforms.selective_tree_map(
    #         traj,
    #         match=lambda k, _: k == traj_key,
    #         map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
    #     )

    # return traj

    # if normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
    #     for key, traj_key in keys_to_normalize.items():
    #         if normalization_type == NormalizationType.BOUNDS:
    #             low = metadata[key]["min"]
    #             high = metadata[key]["max"]
    #         elif normalization_type == NormalizationType.BOUNDS_Q99:
    #             low = metadata[key]["q01"]
    #             high = metadata[key]["q99"]
    #         mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
    #         traj = dl.transforms.selective_tree_map(
    #             traj,
    #             match=lambda k, _: k == traj_key,
    #             map_fn=lambda x: tf.where(
    #                 mask,
    #                 tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
    #                 x,
    #             ),
    #         )

    #         # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
    #         zeros_mask = metadata[key]["min"] == metadata[key]["max"]
    #         traj = dl.transforms.selective_tree_map(
    #             traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
    #         )

    #     return traj

    # raise ValueError(f"Unknown Normalization Type {normalization_type}")


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_names: list[str], dataset_weights: list[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_names)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_name, weight in zip(dataset_names, dataset_weights):
        pad = 80 - len(dataset_name)
        print(f"# {dataset_name}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def allocate_threads(n: int | None, weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation


def load_dataset_kwargs(
    dataset_name: str,
    rlds_data_dir: Path,
    load_camera_views: tuple[str] = ("primary", "wrist"),
) -> dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:
        raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    language_annotations = dataset_kwargs.get("language_annotations")
    if not language_annotations or language_annotations.lower() == "none":
        raise ValueError(f"Cannot load `{dataset_name}`; language annotations required!")

    robot_morphology = dataset_kwargs.get("robot_morphology", "")
    is_bimanual = robot_morphology.lower() == "bi-manual"

    # Add bimanual flag to dataset kwargs
    dataset_kwargs["is_bimanual"] = is_bimanual

    has_suboptimal = dataset_kwargs.get("has_suboptimal")
    if isinstance(has_suboptimal, str):
        has_suboptimal = has_suboptimal.lower() == "yes"
    if has_suboptimal:
        logging.warning(f"Cannot load `{dataset_name}`; suboptimal datasets are not supported!")

    if (
        dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS
        or dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6
    ):
        pass
    else:
        raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    # For bimanual datasets, also load wrist_right camera if available
    camera_views_to_load = load_camera_views
    # if is_bimanual and "wrist_right" in dataset_kwargs["image_obs_keys"]:
    #     camera_views_to_load = tuple(set(load_camera_views) | {"wrist_right"})

    # # Adjust Loaded Camera Views
    # if len(missing_keys := (set(camera_views_to_load) - set(dataset_kwargs["image_obs_keys"]))) > 0:
    #     raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {k: dataset_kwargs["image_obs_keys"].get(k, None) for k in load_camera_views}

    # dataset_kwargs["image_obs_keys"] = {
    #     k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in camera_views_to_load
    # }
    for k, v in dataset_kwargs["image_obs_keys"].items():
        if k == "primary":
            assert v is not None, f"primary image is required for {dataset_name}"

    # Specify Standardization Transform
    # Use unified registry (superset), still supports all OXE datasets
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(rlds_data_dir), **dataset_kwargs}


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


def _R_from_euler_xyz(rpy: tf.Tensor) -> tf.Tensor:
    """Convert extrinsic XYZ Euler angles to a rotation matrix (R = Rz @ Ry @ Rx)."""
    # Explicit num avoids shape inference issues when the last dim is unknown.
    roll, pitch, yaw = tf.unstack(rpy, num=3, axis=-1)

    cr, sr = tf.cos(roll), tf.sin(roll)
    cp, sp = tf.cos(pitch), tf.sin(pitch)
    cy, sy = tf.cos(yaw), tf.sin(yaw)

    # Extrinsic XYZ is equivalent to intrinsic ZYX, so R = Rz * Ry * Rx
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return tf.stack(
        [
            tf.stack([r00, r01, r02], axis=-1),
            tf.stack([r10, r11, r12], axis=-1),
            tf.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


def euler_to_rot6d(euler_angles: tf.Tensor) -> tf.Tensor:
    """
    Convert extrinsic XYZ Euler angles to 6D rotation representation.
    Args:
        euler_angles: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)
    Returns:
        (..., 6) tensor of 6D rotation representation
    """
    R = _R_from_euler_xyz(euler_angles)
    # Use dynamic shape to avoid None dimensions inside tf.function.
    leading_shape = tf.shape(R)[:-2]
    rot6d = tf.reshape(R[..., :2, :], tf.concat([leading_shape, [6]], axis=0))
    return rot6d


@tf.function
def _euler_xyz_from_R(R, eps=1e-6):
    """
    Extract extrinsic XYZ (roll, pitch, yaw) from rotation matrix R (R = Rz @ Ry @ Rx).

    Handles gimbal lock via elementwise tf.where.
    """
    R = tf.convert_to_tensor(R)
    dtype = R.dtype
    eps_t = tf.cast(eps, dtype)

    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    sy = tf.sqrt(tf.maximum(r00 * r00 + r10 * r10, eps_t))
    singular = sy < eps_t

    roll_regular = tf.atan2(r21, r22)
    roll_singular = tf.atan2(-r12, r11)
    roll = tf.where(singular, roll_singular, roll_regular)

    pitch = tf.atan2(-r20, sy)

    yaw_regular = tf.atan2(r10, r00)
    yaw_singular = tf.zeros_like(yaw_regular)
    yaw = tf.where(singular, yaw_singular, yaw_regular)

    return tf.stack([roll, pitch, yaw], axis=-1)


@tf.function
def zxy_to_xyz_tf(angles, degrees=False, eps=1e-6):
    """
    Convert intrinsic Z-X-Y Euler angles to extrinsic X-Y-Z Euler angles.
    angles: tensor of shape (..., 3) -> (az, ax, ay) in radians by default (intrinsic ZXY).
    returns: tensor of shape (..., 3) -> (roll[x], pitch[y], yaw[z]) in same unit (extrinsic XYZ).
    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if degrees:
        angles = tf.math.multiply(angles, _tf_pi(tf.float32) / 180.0)

    az = angles[..., 0]  # rotate about z
    ax = angles[..., 1]  # then about x
    ay = angles[..., 2]  # then about y

    # Build rotation for intrinsic "zxy": R = Rz(az) @ Rx(ax) @ Ry(ay)
    Rz = _rot_z(az)
    Rx = _rot_x(ax)
    Ry = _rot_y(ay)
    R = tf.linalg.matmul(tf.linalg.matmul(Rz, Rx), Ry)

    # Extract extrinsic XYZ angles from rotation matrix
    out = _euler_xyz_from_R(R, eps=eps)
    if degrees:
        out = tf.math.multiply(out, 180.0 / _tf_pi(tf.float32))
    return out


@tf.function
def euler_diff(angles1, angles2, order="xyz", degrees=False):
    """
    Compute relative Euler angle difference: angles_rel such that
        R(angles2) * R(angles_rel) = R(angles1)

    Args:
        angles1: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)
        angles2: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)
        order:   rotation order string (currently only "xyz" extrinsic is supported)
        degrees: whether input/output are in degrees
    Returns:
        (..., 3) tensor of relative Euler angles [roll, pitch, yaw] (extrinsic XYZ)
    """
    if degrees:
        angles1 = tf.math.multiply(angles1, _tf_pi(tf.float32) / 180.0)
        angles2 = tf.math.multiply(angles2, _tf_pi(tf.float32) / 180.0)

    # Build rotation matrices using extrinsic XYZ convention
    R1 = _R_from_euler_xyz(angles1)
    R2 = _R_from_euler_xyz(angles2)

    # Compute relative rotation: Rrel = R2^T * R1
    Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)

    # Extract Euler angles from Rrel using robust method with gimbal lock handling
    out = _euler_xyz_from_R(Rrel)
    if degrees:
        out = tf.math.multiply(out, 180.0 / _tf_pi(tf.float32))
    return out


@tf.function
def matrix_to_xyzrpy(T, eps=1e-6):
    """
    Extract position and Euler angles from 4x4 transformation matrix.

    Args:
        T: tensor of shape (..., 4, 4) - homogeneous transformation matrix
        eps: epsilon for gimbal lock handling

    Returns:
        tensor of shape (..., 6) - [x, y, z, roll, pitch, yaw] (extrinsic XYZ)
    """
    T = tf.convert_to_tensor(T)

    # Extract translation (position)
    xyz = T[..., :3, 3]

    # Extract rotation matrix (top-left 3x3)
    R = T[..., :3, :3]

    # Convert rotation matrix to extrinsic XYZ Euler angles (roll, pitch, yaw)
    rpy = _euler_xyz_from_R(R, eps=eps)

    # Concatenate position and orientation
    return tf.concat([xyz, rpy], axis=-1)


@tf.function
def coordinate_transform_bcz(movement_actions):
    """
    movement_actions: (..., 6) where [:3] = translation deltas (xyz),
                      [3:6] = Euler deltas in extrinsic XYZ (roll,pitch,yaw).
    Returns transformed actions under x'=-y, y'=-x, z'=-z.

    """
    # Fixed frame transform: x'=-y, y'=-x, z'=-z
    movement_actions = tf.convert_to_tensor(movement_actions)
    dtype = movement_actions.dtype

    # Ensure _C matches input dtype
    C = tf.constant([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=dtype)

    # translations: t' = C t
    t = movement_actions[..., :3]
    t_prime = tf.linalg.matvec(C, t)

    # rotations: R' = C R C^T, then back to extrinsic XYZ Euler angles
    e = movement_actions[..., 3:6]
    R = _R_from_euler_xyz(e)
    # _C is symmetric here, but keep transpose for clarity / generality
    CT = tf.linalg.matrix_transpose(C)
    R_prime = tf.linalg.matmul(tf.linalg.matmul(C, R), CT)
    e_prime = _euler_xyz_from_R(R_prime)

    return tf.concat([t_prime, e_prime], axis=-1)


@tf.function
def coordinate_transform_dobbe(movement_actions):
    """
    movement_actions: (..., 6) where [:3] = translation deltas (xyz),
                      [3:6] = Euler deltas in extrinsic XYZ (roll,pitch,yaw).
    Returns transformed actions under x'=y, y'=-x, z'=z.

    """
    movement_actions = tf.convert_to_tensor(movement_actions)
    dtype = movement_actions.dtype

    # Ensure _C matches input dtype
    C = tf.constant([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    # translations: t' = C t
    t = movement_actions[..., :3]
    t_prime = tf.linalg.matvec(C, t)

    # rotations: R' = C R C^T, then back to extrinsic XYZ Euler angles
    e = movement_actions[..., 3:6]
    R = _R_from_euler_xyz(e)
    # _C is symmetric here, but keep transpose for clarity / generality
    CT = tf.linalg.matrix_transpose(C)
    R_prime = tf.linalg.matmul(tf.linalg.matmul(C, R), CT)
    e_prime = _euler_xyz_from_R(R_prime)

    return tf.concat([t_prime, e_prime], axis=-1)


@tf.function
def coordinate_transform_jaco(movement_actions):
    """
    movement_actions: (..., 6) where [:3] = translation deltas (xyz),
                      [3:6] = Euler deltas in extrinsic XYZ (roll,pitch,yaw).
    Returns transformed actions under x'=-y, y'=x, z'=z.

    """
    movement_actions = tf.convert_to_tensor(movement_actions)
    dtype = movement_actions.dtype

    # Ensure _C matches input dtype
    C = tf.constant([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    # translations: t' = C t
    t = movement_actions[..., :3]
    t_prime = tf.linalg.matvec(C, t)

    # rotations: R' = C R C^T, then back to extrinsic XYZ Euler angles
    e = movement_actions[..., 3:6]
    R = _R_from_euler_xyz(e)
    # _C is symmetric here, but keep transpose for clarity / generality
    CT = tf.linalg.matrix_transpose(C)
    R_prime = tf.linalg.matmul(tf.linalg.matmul(C, R), CT)
    e_prime = _euler_xyz_from_R(R_prime)

    return tf.concat([t_prime, e_prime], axis=-1)
