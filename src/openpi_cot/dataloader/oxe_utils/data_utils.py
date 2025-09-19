"""
data_utils.py

Additional RLDS-specific data utilities.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any

import dlimp as dl
import numpy as np
import tensorflow as tf


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


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


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


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: dict[str, Any]) -> dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


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
