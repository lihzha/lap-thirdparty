from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf


class FrameFilter(Protocol):
    def __call__(self, frame: dict) -> tf.Tensor: ...


@dataclass(frozen=True)
class PromptFilter:
    """Keep frames whose `prompt` is a non-empty string after strip()."""

    def __call__(self, frame: dict) -> tf.Tensor:
        prompt = tf.strings.strip(frame["prompt"])  # scalar tf.string after flatten
        return tf.greater(tf.strings.length(prompt), 0)


@dataclass(frozen=True)
class IdleFilter:
    """Keep frames that are marked non-idle.

    If `example_mask` exists (produced by tokenization), use it directly.
    Otherwise, if `passes_filter` exists (produced earlier in the pipeline), use it.
    Fallback: keep everything.
    """

    key_priority: tuple[str, ...] = ("example_mask", "passes_filter")

    def __call__(self, frame: dict) -> tf.Tensor:
        for k in self.key_priority:
            if k in frame:
                v = frame[k]
                if v.dtype != tf.bool:
                    v = tf.cast(v, tf.bool)
                return v
        return tf.constant(True)


@dataclass(frozen=True)
class SuccessFilter:
    """Keep only samples originating from successful trajectories.

    Expects `traj_metadata.episode_metadata.file_path` to be available on the unflattened trajectory.
    When called on frames (post-flatten), this filter should be applied earlier at trajectory level.
    """

    pattern: tf.Tensor = tf.constant(".*success.*")

    def __call__(self, traj: dict) -> tf.Tensor:
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        return tf.strings.regex_full_match(file_path, self.pattern)


@dataclass(frozen=True)
class EmptyTrajectoryFilter:
    """Keep trajectories with non-zero length (based on `action` or `actions`)."""

    def __call__(self, traj: dict) -> tf.Tensor:
        if "action" in traj:
            return tf.shape(traj["action"])[0] > 0
        if "actions" in traj:
            return tf.shape(traj["actions"])[0] > 0
        # If neither key is present, conservatively keep
        return tf.constant(True)
