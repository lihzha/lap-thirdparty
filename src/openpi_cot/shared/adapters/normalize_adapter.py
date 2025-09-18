import json
import os

import dlimp as dl
import numpy as np
from openpi.shared import normalize as _normalize
import tensorflow as tf
import tqdm

import openpi_cot.training.config as _config


def save(directory: str, norm_stats: dict[str, _normalize.NormStats]) -> None:
    """Save the normalization stats to a directory (supports gs:// or local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    tf.io.gfile.makedirs(os.path.dirname(str(path)))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(_normalize.serialize_json(norm_stats))


def load(directory: str) -> dict[str, _normalize.NormStats]:
    """Load the normalization stats from a directory (supports gs:// and local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    with tf.io.gfile.GFile(path, "r") as f:
        return _normalize.deserialize_json(f.read())


def check_dataset_statistics(save_dir: str | None = None, data_config: _config.CoTDataConfig | None = None) -> dict:
    """
    Checks if the dataset statistics are already computed and returns them if they are.
    """

    # Fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(
        os.path.join("~", ".cache", "orca", f"dataset_statistics_{data_config.repo_id}.json")
    )
    path = save_dir / data_config.repo_id if save_dir is not None else local_path

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        with tf.io.gfile.GFile(path, "r") as f:
            return json.load(f), path, local_path

    raise ValueError(f"Norm stats file not found at: {path}")
    # if os.path.exists(local_path):
    #     with open(local_path) as f:
    #         return json.load(f), local_path, local_path

    # return None, local_path, local_path


def get_dataset_statistics(
    dataset: dl.DLataset,
    save_dir: str | None = None,
    data_config: _config.CoTDataConfig | None = None,
    action_key: str = "action",
    state_key: str = "proprio",
) -> dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    metadata, output_path, _ = check_dataset_statistics(save_dir, data_config)
    if metadata is not None:
        return metadata

    dataset = dataset.traj_map(
        lambda traj: {
            action_key: traj[action_key],
            state_key: (
                traj["observation"][state_key] if state_key in traj["observation"] else tf.zeros_like(traj[action_key])
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj[action_key])
        proprios.append(traj[state_key])
        num_transitions += traj[action_key].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)

    norm_stats = {
        "state": _normalize.NormStats(
            mean=proprios.mean(0).tolist(),
            std=proprios.std(0).tolist(),
            q01=np.quantile(proprios, 0.01, axis=0).tolist(),
            q99=np.quantile(proprios, 0.99, axis=0).tolist(),
        ),
        "actions": _normalize.NormStats(
            mean=actions.mean(0).tolist(),
            std=actions.std(0).tolist(),
            q01=np.quantile(actions, 0.01, axis=0).tolist(),
            q99=np.quantile(actions, 0.99, axis=0).tolist(),
        ),
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    print(f"Writing stats to: {output_path}")
    save(output_path, norm_stats)

    return metadata
