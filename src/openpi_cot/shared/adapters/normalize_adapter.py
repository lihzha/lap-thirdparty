import json
import logging
import os

import dlimp as dl
import jax
import numpy as np
from openpi.shared import normalize as _normalize
import pydantic
import tensorflow as tf
from tqdm_loggable.auto import tqdm


@pydantic.dataclasses.dataclass
class ExtendedNormStats(_normalize.NormStats):
    num_transitions: int | None = None
    num_trajectories: int | None = None


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, ExtendedNormStats]


def serialize_json(norm_stats: dict[str, ExtendedNormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, ExtendedNormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: str, norm_stats: dict[str, ExtendedNormStats]) -> None:
    """Save the normalization stats to a directory (supports gs:// or local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    tf.io.gfile.makedirs(os.path.dirname(str(path)))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(serialize_json(norm_stats))
    logging.info(f"Saved stats to: {path}")


def load(directory: str) -> dict[str, ExtendedNormStats]:
    """Load the normalization stats from a directory (supports gs:// and local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    with tf.io.gfile.GFile(path, "r") as f:
        return deserialize_json(f.read())


def check_dataset_statistics(save_dir: str | None = None) -> dict:
    """
    Checks if the dataset statistics are already computed and returns them if they are.
    """

    # Fallback local directory for when save_dir is not writable or not provided
    local_dir = os.path.expanduser(os.path.join("~", ".cache", "orca"))
    preferred_dir = save_dir if save_dir is not None else local_dir

    # Look for norm_stats.json in the preferred dir first, then in the local cache dir.
    if tf.io.gfile.exists(preferred_dir):
        try:
            stats = load(preferred_dir)
            return stats, preferred_dir, local_dir
        except Exception:
            pass

    if tf.io.gfile.exists(local_dir):
        try:
            stats = load(local_dir)
            return stats, local_dir, local_dir
        except Exception:
            pass

    return None, preferred_dir, local_dir


def get_dataset_statistics(
    dataset: dl.DLataset,
    save_dir: str | None = None,
    action_key: str = "action",
    state_key: str = "proprio",
) -> dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    metadata, output_dir, _ = check_dataset_statistics(save_dir)
    if metadata is not None:
        return metadata

    # dataset = dataset.traj_map(
    #     lambda traj: {
    #         action_key: traj[action_key],
    #         state_key: (
    #             traj["observation"][state_key] if state_key in traj["observation"] else tf.zeros_like(traj[action_key])
    #         ),
    #     }
    # )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY or cardinality == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality):
        actions.append(traj[action_key])
        proprios.append(traj["observation"][state_key])
        num_transitions += traj[action_key].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)

    norm_stats = {
        "state": ExtendedNormStats(
            mean=np.asarray(proprios.mean(0)),
            std=np.asarray(proprios.std(0)),
            q01=np.asarray(np.quantile(proprios, 0.01, axis=0)),
            q99=np.asarray(np.quantile(proprios, 0.99, axis=0)),
            num_transitions=num_transitions,
            num_trajectories=num_trajectories,
        ),
        "actions": ExtendedNormStats(
            mean=np.asarray(actions.mean(0)),
            std=np.asarray(actions.std(0)),
            q01=np.asarray(np.quantile(actions, 0.01, axis=0)),
            q99=np.asarray(np.quantile(actions, 0.99, axis=0)),
            num_transitions=num_transitions,
            num_trajectories=num_trajectories,
        ),
    }

    if jax.process_index() == 0:
        print(f"Writing stats to: {output_dir}")
        save(output_dir, norm_stats)

    return norm_stats
