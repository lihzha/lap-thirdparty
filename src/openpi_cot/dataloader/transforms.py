from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorflow as tf

from openpi_cot.dataloader.oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

# Re-export a unified registry that can be extended with additional datasets (e.g., DROID) later.
UNIFIED_STANDARDIZATION_TRANSFORMS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    **OXE_STANDARDIZATION_TRANSFORMS
}


def pad_action_and_state(traj: dict, *, action_key: str, state_path: tuple[str, str], action_dim: int) -> dict:
    """Pad action and state/proprio to a fixed last-dim `action_dim`.

    Args:
        traj: trajectory dict
        action_key: key for action tensor in `traj`
        state_path: tuple("observation", <state_key>) where the state tensor is stored
        action_dim: target last-dimension size
    """
    obs_key, state_key = state_path
    traj[action_key] = tf.pad(traj[action_key], [[0, 0], [0, action_dim - tf.shape(traj[action_key])[-1]]])
    traj[obs_key][state_key] = tf.pad(
        traj[obs_key][state_key], [[0, 0], [0, action_dim - tf.shape(traj[obs_key][state_key])[-1]]]
    )
    return traj


def chunk_sequence(traj: dict, *, seq_key: str, window_size: int) -> dict:
    """Chunk a 2D time-major sequence [T, D] into sliding windows [T, window, D]."""
    seq = traj[seq_key]
    T = tf.shape(seq)[0]
    base = tf.broadcast_to(tf.range(window_size)[None], [T, window_size])
    offsets = tf.broadcast_to(tf.range(T)[:, None], [T, window_size])
    idx = tf.minimum(base + offsets, T - 1)
    traj[seq_key] = tf.gather(seq, idx)
    return traj


def group_language_actions_text(
    traj: dict,
    *,
    language_key: str,
    summation_steps: int,
    control_frequency: int,
) -> dict:
    """Group string language actions into fixed-length windows per step.

    Trims each window to `min(control_frequency, summation_steps)` and pads with empty strings to length `summation_steps`.
    Stores grouped strings back to `language_key` with shape [T, summation_steps].
    """
    T = tf.shape(traj[language_key])[0]
    base = tf.broadcast_to(tf.range(summation_steps)[None], [T, summation_steps])
    offsets = tf.broadcast_to(tf.range(T)[:, None], [T, summation_steps])
    idx = tf.minimum(base + offsets, T - 1)

    trimmed_len = tf.minimum(tf.cast(control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))
    window = tf.gather(traj[language_key], idx[:, :trimmed_len])
    pad_len = summation_steps - trimmed_len

    def _pad():
        return tf.concat([window, tf.fill([tf.shape(window)[0], pad_len], tf.constant("", dtype=tf.string))], axis=1)

    traj[language_key] = tf.cond(pad_len > 0, _pad, lambda: window)
    return traj


def group_language_actions_numeric(
    traj: dict,
    *,
    raw_action_key: str,
    out_language_key: str,
    summation_steps: int,
    control_frequency: int,
) -> dict:
    """Group numeric actions over a future window, then serialize each row to tf.string.

    Produces `traj[out_language_key]` with shape [T, summation_steps] of serialized float rows.
    """
    T = tf.shape(traj[raw_action_key])[0]
    base = tf.broadcast_to(tf.range(summation_steps)[None], [T, summation_steps])
    offsets = tf.broadcast_to(tf.range(T)[:, None], [T, summation_steps])
    idx = tf.minimum(base + offsets, T - 1)

    trimmed_len = tf.minimum(tf.cast(control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))
    window_trim = tf.gather(traj[raw_action_key], idx[:, :trimmed_len])  # [T, trimmed_len, A]
    pad_len = summation_steps - trimmed_len

    def _pad():
        zeros = tf.zeros([tf.shape(window_trim)[0], pad_len, tf.shape(window_trim)[-1]], dtype=window_trim.dtype)
        return tf.concat([window_trim, zeros], axis=1)

    window = tf.cond(pad_len > 0, _pad, lambda: window_trim)
    flat = tf.reshape(window, [-1, tf.shape(window)[-1]])
    serialized = tf.map_fn(lambda v: tf.io.serialize_tensor(v), flat, fn_output_signature=tf.string)
    traj[out_language_key] = tf.reshape(serialized, [tf.shape(window)[0], summation_steps])
    return traj
