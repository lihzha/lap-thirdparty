import tensorflow as tf

from openpi_cot.datasets.base_dataset import SingleCoTDataset
from openpi_cot.datasets.robot.droid_dataset import DroidCoTDataset
from openpi_cot.datasets.robot.oxe_datasets import SingleOXECoTDataset


class _FakeTrajectoryDataset:
    """Minimal dataset wrapper that supports the DLataset APIs used in tests."""

    def __init__(self, trajectories: list[dict]):
        self._trajectories = trajectories

    def traj_map(self, fn, num_parallel_calls=None):
        # Apply mapping eagerly over the stored trajectories.
        mapped = [fn(traj) for traj in self._trajectories]
        return _FakeTrajectoryDataset(mapped)

    def filter(self, predicate):
        filtered = []
        for traj in self._trajectories:
            keep = predicate(traj)
            if isinstance(keep, tf.Tensor):
                keep = bool(keep.numpy())
            else:
                keep = bool(keep)
            if keep:
                filtered.append(traj)
        return _FakeTrajectoryDataset(filtered)

    def as_numpy_iterator(self):
        for traj in self._trajectories:
            yield tf.nest.map_structure(
                lambda x: x.numpy() if isinstance(x, tf.Tensor) else x,
                traj,
            )

    @property
    def trajectories(self):
        return self._trajectories


def _build_dummy_trajectory_dataset(traj_ids: list[str]) -> tf.data.Dataset:
    """Create a tf.data dataset that mimics CoT trajectory elements."""

    def _to_traj(tid):
        # `trajectory_id` in production is a sequence; mimic by wrapping in length-1 tensor.
        return {"trajectory_id": tf.reshape(tid, [1])}

    return tf.data.Dataset.from_tensor_slices(traj_ids).map(_to_traj)


def _collect_traj_ids(dataset) -> set[str]:
    """Collect the (string) trajectory IDs from a dataset-like object."""
    traj_ids = set()
    if isinstance(dataset, _FakeTrajectoryDataset):
        iterator = dataset.trajectories
        for traj in iterator:
            traj_id = traj["trajectory_id"][0]
            if isinstance(traj_id, tf.Tensor):
                traj_id = traj_id.numpy()
            traj_ids.add(traj_id.decode("utf-8"))
        return traj_ids

    for traj in dataset.as_numpy_iterator():
        traj_ids.add(traj["trajectory_id"][0].decode("utf-8"))
    return traj_ids


def _make_fake_droid_trajectories(episode_names: list[str], traj_len: int = 3) -> list[dict]:
    """Construct simple DROID-like trajectories with metadata for testing."""
    trajectories = []
    base_action = tf.reshape(tf.range(traj_len, dtype=tf.float32), [traj_len, 1])
    for idx, episode in enumerate(episode_names):
        action = base_action + tf.cast(idx, tf.float32)
        trajectories.append(
            {
                "action": action,
                "traj_metadata": {
                    "episode_metadata": {
                        "file_path": tf.constant([f"{episode}/trajectory_0"], dtype=tf.string),
                    }
                },
            }
        )
    return trajectories


def _make_fake_oxe_trajectories(num_traj: int, traj_len: int = 4) -> list[dict]:
    """Create simple trajectories with per-trajectory action content."""
    trajectories = []
    base = tf.reshape(tf.range(traj_len, dtype=tf.float32), [traj_len, 1])
    for idx in range(num_traj):
        action = base + tf.cast(idx, tf.float32)
        trajectories.append({"action": action})
    return trajectories


def _build_droid_lookup_table(episodes: list[str], episode_ids: list[str]) -> tf.lookup.StaticHashTable:
    keys = tf.constant(episodes, dtype=tf.string)
    values = tf.constant(episode_ids, dtype=tf.string)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=tf.constant("unknown", dtype=tf.string),
    )


def _build_droid_dataset(
    *,
    want_val: bool,
    val_fraction: float,
    split_seed: int,
    episode_names: list[str],
    episode_ids: list[str],
) -> DroidCoTDataset:
    dataset = DroidCoTDataset.__new__(DroidCoTDataset)
    dataset.dataset = _FakeTrajectoryDataset(_make_fake_droid_trajectories(episode_names))
    dataset.want_val = want_val
    dataset.val_fraction = val_fraction
    dataset.seed = 1
    dataset.num_parallel_calls = None
    dataset.ep_table = _build_droid_lookup_table(episode_names, episode_ids)
    dataset.get_traj_identifier()
    dataset.split_val(split_seed=split_seed)
    return dataset


def _build_oxe_dataset(
    *,
    want_val: bool,
    val_fraction: float,
    split_seed: int,
    dataset_name: str,
    num_traj: int,
) -> SingleOXECoTDataset:
    dataset = SingleOXECoTDataset.__new__(SingleOXECoTDataset)
    dataset.dataset = _FakeTrajectoryDataset(_make_fake_oxe_trajectories(num_traj))
    dataset.want_val = want_val
    dataset.val_fraction = val_fraction
    dataset.seed = 7
    dataset.dataset_name = dataset_name
    dataset.standardize_fn = None
    dataset.num_parallel_calls = None
    dataset.get_traj_identifier()
    if split_seed is not None:
        dataset.split_val(split_seed=split_seed)
    return dataset


def test_split_val_produces_disjoint_train_and_val_sets():
    traj_ids = [f"trajectory_{i}" for i in range(25)]
    val_fraction = 0.3
    split_seed = 17

    # Build datasets for train/val using the same synthetic trajectories.
    val_dataset = SingleCoTDataset.__new__(SingleCoTDataset)
    val_dataset.dataset = _build_dummy_trajectory_dataset(traj_ids)
    val_dataset.want_val = True
    val_dataset.val_fraction = val_fraction
    val_dataset.split_val(split_seed=split_seed)

    train_dataset = SingleCoTDataset.__new__(SingleCoTDataset)
    train_dataset.dataset = _build_dummy_trajectory_dataset(traj_ids)
    train_dataset.want_val = False
    train_dataset.val_fraction = val_fraction
    train_dataset.split_val(split_seed=split_seed)

    val_ids = _collect_traj_ids(val_dataset.dataset)
    train_ids = _collect_traj_ids(train_dataset.dataset)

    # The hash-based split should partition the set of trajectories.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.union(val_ids) == set(traj_ids)


def test_droid_identifier_split_uses_episode_lookup():
    episode_names = [f"episode_{i}" for i in range(20)]
    episode_ids = [f"droid_ep_{i}" for i in range(20)]
    val_fraction = 0.4
    split_seed = 9

    val_dataset = _build_droid_dataset(
        want_val=True,
        val_fraction=val_fraction,
        split_seed=split_seed,
        episode_names=episode_names,
        episode_ids=episode_ids,
    )
    train_dataset = _build_droid_dataset(
        want_val=False,
        val_fraction=val_fraction,
        split_seed=split_seed,
        episode_names=episode_names,
        episode_ids=episode_ids,
    )

    val_ids = _collect_traj_ids(val_dataset.dataset)
    train_ids = _collect_traj_ids(train_dataset.dataset)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.union(val_ids) == set(episode_ids)


def test_oxe_identifier_split_preserves_hash_partitioning():
    dataset_name = "fake_oxe"
    num_traj = 9000
    val_fraction = 0.03
    split_seed = 21

    # Collect the unique identifiers produced before splitting.
    identifier_dataset = _build_oxe_dataset(
        want_val=True,
        val_fraction=val_fraction,
        split_seed=None,
        dataset_name=dataset_name,
        num_traj=num_traj,
    )
    all_ids = _collect_traj_ids(identifier_dataset.dataset)

    val_dataset = _build_oxe_dataset(
        want_val=True,
        val_fraction=val_fraction,
        split_seed=split_seed,
        dataset_name=dataset_name,
        num_traj=num_traj,
    )
    train_dataset = _build_oxe_dataset(
        want_val=False,
        val_fraction=val_fraction,
        split_seed=split_seed,
        dataset_name=dataset_name,
        num_traj=num_traj,
    )
    breakpoint()

    val_ids = _collect_traj_ids(val_dataset.dataset)
    train_ids = _collect_traj_ids(train_dataset.dataset)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.union(val_ids) == all_ids
