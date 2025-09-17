"""
RLDS-based data loader for DROID with CoT-style language actions.

This revision merges the standalone pre-processing logic from
`process_dataset.py` directly into the `DroidCoTRldsDataset` class so
that everything from raw RLDS → ready-to-train batches happens in one
place.  The main additions are:

*   **Episode calibration lookup** - we pre-compute, on initialisation,
    which exterior camera (ext 1 vs ext 2) should be used for every
    DROID episode based on the extrinsics files.  At run-time the correct
    image is selected without expensive Python branching inside the
    TensorFlow graph.
*   **Language-action loading** - natural-language low-level action
    strings are loaded from the `<episode_id>_language_action.json`
    files and stored in a `tf.lookup.StaticHashTable`, so they can be
    joined with the trajectory entirely on the TF side.  The final batch
    therefore contains a `language_actions` tensor of shape
    `(B, T_chunk)` aligned with the action chunk.
*   **Restructure pass rewritten** - now relies on the lookup tables for
    both the calibrated image key and language actions; the remaining
    logic stays in TF ops (no `py_function`) so the pipeline is fully
    traceable and fast.

Usage example
-------------
```python
loader = DroidCoTRldsDataset(
    data_dir="gs://gresearch/robotics",
    batch_size=32,
    action_space=DroidActionSpace.CARTESIAN_POSITION,
    language_action_dir="/n/fs/robot-data/vlm-syn/posed_droid",
)
for batch in loader:
    images   = batch["observation"]["image"]          # (B, L, H, W, 3)
    actions  = batch["actions"]                        # (B, L, 7)
    lang_act = batch["language_actions"]               # (B, L)
    ...
```

The rest of the public API and the chunking / filtering behaviour remain
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from enum import Enum
from enum import auto
from functools import partial
import inspect
import json
import logging
import os
from typing import ClassVar

import dlimp as dl
import jax
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds

from openpi_cot.dataloader.helpers import euler_xyz_to_rot
from openpi_cot.dataloader.helpers import extract_episode_path_from_file_path
from openpi_cot.dataloader.helpers import project_in_bounds
from openpi_cot.dataloader.oxe_utils.data_utils import NormalizationType
from openpi_cot.dataloader.oxe_utils.data_utils import allocate_threads
from openpi_cot.dataloader.oxe_utils.data_utils import get_dataset_statistics
from openpi_cot.dataloader.oxe_utils.data_utils import normalize_action_and_proprio
from openpi_cot.dataloader.oxe_utils.data_utils import pprint_data_mixture
from openpi_cot.dataloader.oxe_utils.data_utils import tree_map
from openpi_cot.dataloader.oxe_utils.materialize import get_oxe_dataset_kwargs_and_weights
from openpi_cot.dataloader.oxe_utils.mixtures import OXE_NAMED_MIXTURES


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


def batch_prefetch(dataset: dl.DLataset, batch_size: int) -> dl.DLataset:
    batched = dataset.batch(batch_size, drop_remainder=True)
    try:
        prefetched = batched.prefetch_to_device(2)
    except Exception:
        prefetched = batched.prefetch(2)
    # Note =>> Seems to reduce memory usage without affecting speed?
    return prefetched.with_ram_budget(1)


def maybe_shuffle_and_take(
    dataset: dl.DLataset,
    *,
    want_val: bool,
    shuffle: bool,
    shuffle_buffer_size: int,
    seed: int,
    max_samples: int | None,
) -> dl.DLataset:
    """Apply common finalization: optional shuffle and optional take/cache/repeat.

    Only shuffles during training (not in validation). If max_samples is set,
    takes that many samples and repeats indefinitely.
    """
    if (not want_val) and shuffle and max_samples is None:
        return dataset.shuffle(shuffle_buffer_size, seed=seed)
    if max_samples is not None:
        return dataset.take(int(max_samples)).cache().repeat()
    return dataset


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    use_wrist_image: bool,
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors."""

    def _decode_single(img_bytes):
        return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)

    def _decode_frame(traj: dict) -> dict:
        traj["observation"][primary_key] = _decode_single(traj["observation"][primary_key])
        if use_wrist_image and wrist_key is not None:
            traj["observation"][wrist_key] = _decode_single(traj["observation"][wrist_key])
        return traj

    return _decode_frame


def _param_info(fn: Callable):
    """Return (accepted_param_names, required_param_names, has_var_kwargs) for a callable.

    - Skips `self`.
    - Considers parameters without defaults (and not varargs) as required.
    """
    sig = inspect.signature(fn)
    accepted: set[str] = set()
    required: set[str] = set()
    has_var_kwargs = False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        accepted.add(name)
        if param.default is inspect.Signature.empty:
            required.add(name)
    return accepted, required, has_var_kwargs


def _filter_kwargs_for(fn: Callable, provided: dict) -> tuple[dict, list[str]]:
    """Filter `provided` to only kwargs accepted by `fn` and report missing required."""
    accepted, required, _ = _param_info(fn)
    filtered = {k: v for k, v in provided.items() if k in accepted}
    missing = [k for k in required if k not in provided]
    return filtered, missing


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_POSITION = auto()


class SingleCoTRldsDatasetRaw:
    fallback_instructions = tf.constant(
        [
            "Do something useful.",
            "Complete the task.",
            "Perform the task.",
            "Carry out the objective.",
            "Execute the current task.",
            "Accomplish the goal.",
            "Proceed with the task.",
            "Handle the task at hand.",
            "Continue the operation.",
            "Fulfill the task.",
            "Take meaningful steps.",
            "Demonstrate useful behavior.",
            "Act in a useful manner.",
            "Engage in productive actions.",
            "Make useful moves.",
            "Undertake useful actions.",
            "Behave purposefully.",
            "Start the activity.",
        ],
        dtype=tf.string,
    )

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        config,
        *,  # Force keyword-only arguments
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
    ):
        self.config = config
        self.seed = seed
        self.want_val = split == "val"
        self.dataset_name = dataset_name

        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls
        self.validation_mode = getattr(self.config, "validation_mode", "easy")
        self.validation_mode = (self.validation_mode or "easy").lower()
        assert self.validation_mode in {"easy", "hard"}, (
            f"validation_mode must be one of 'easy', 'hard'; got: {self.validation_mode}"
        )
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)

        # ------------------------------------------------------------------
        # Global seeding for reproducibility across dataset ops
        # ------------------------------------------------------------------
        tf.random.set_seed(self.seed)
        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        tf.config.set_visible_devices([], "GPU")
        with contextlib.suppress(Exception):
            tf.config.set_visible_devices([], "TPU")

        self.builder = self.build_dataset_builder(dataset_name, data_dir)
        self.dataset = self.build_dataset(self.builder)

    def build_dataset_builder(self, ds_name, data_dir):
        return tfds.builder(ds_name, data_dir=data_dir)

    def build_dataset(self, builder):
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(self.want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True
        cpu_count = psutil.cpu_count(logical=True) or 16
        opts.experimental_threading.private_threadpool_size = int(max(16, cpu_count))
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=bool(self.want_val),  # shuffle at file/shard level for first-level randomness
            num_parallel_reads=self.num_parallel_reads,
        )
        dataset = dataset.shard(jax.process_count(), jax.process_index())
        # Repeat early to increase interleaving across files/episodes
        dataset = dataset.with_options(opts)
        return dataset

    def apply_flatten(self):
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)


class DroidCoTRldsDatasetRaw(SingleCoTRldsDatasetRaw):
    lang_action_tfrecord_pattern = "tfds_language_actions-*.tfrecord.gz"
    lang_action_dir_name = "droid-lang-actions"
    lang_action_dir_name_base = "droid-base-lang-actions"
    metadata_path_name = "metadata"
    episode_id_to_path_file = "episode_id_to_path.json"
    cam2base_extrinsics_file = "cam2base_extrinsics.json"
    camera_serials_file = "camera_serials.json"
    intrinsics_file = "intrinsics.json"
    droid_instructions_file = "droid_instructions.json"
    droid_language_annotations_file = "droid_language_annotations.json"
    keep_ranges_file = "keep_ranges_1_0_1.json"
    default_lang_value = tf.constant(b"", dtype=tf.string)
    default_ep_value = tf.constant("", dtype=tf.string)
    default_intr_ser = tf.io.serialize_tensor(tf.zeros([4], tf.float32))
    default_extr_ser = tf.io.serialize_tensor(tf.reshape(tf.eye(4, dtype=tf.float32), [-1]))
    images_list: ClassVar[tuple[str, str]] = (
        "exterior_image_1_left",
        "exterior_image_2_left",
    )

    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lang_action_table(self, language_action_dir):
        # ---------------------------------------------------------------------
        # 1. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        features = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(record):
            ex = tf.io.parse_single_example(record, features)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        files = tf.io.gfile.glob(f"{language_action_dir}/{self.lang_action_tfrecord_pattern}")
        ds = (
            tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=self.num_parallel_reads)
            .map(_parse, num_parallel_calls=self.num_parallel_calls)
            .prefetch(tf.data.AUTOTUNE)
        )
        episodes, lang_serialized = [], []
        for ep_id, lang in ds:
            episodes.append(ep_id.numpy().decode())
            lang_serialized.append(tf.io.serialize_tensor(lang).numpy())

        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.default_lang_value,
        )
        print_memory_usage("After building lang_table")
        return lang_table

    def build_lookup_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 2. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.episode_id_to_path_file}", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.default_ep_value,
        )
        print_memory_usage("After building ep_table")
        return ep_table

    def build_cam_tables(self, metadata_path, need_calib: bool):
        # ---------------------------------------------------------------------
        # 3. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.cam2base_extrinsics_file}", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{metadata_path}/{self.camera_serials_file}", "r") as fp:
            camera_serials = json.load(fp)
        if need_calib:
            with tf.io.gfile.GFile(f"{metadata_path}/{self.intrinsics_file}", "r") as fp:
                intrinsics_json = json.load(fp)
            eid_to_intr_vec = {}
            eid_to_extr_mat = {}

        eid_to_cam_dict = {}

        for eid, extr in cam2base_extrinsics.items():
            cams = camera_serials[eid]
            camera_serial = next(k for k in extr if k.isdigit())
            serial_to_name = {v: k for k, v in cams.items()}
            if camera_serial not in serial_to_name:
                continue

            calib_camera_name = serial_to_name[camera_serial]
            if calib_camera_name == "ext1_cam_serial":
                calib_image_name = "exterior_image_1_left"
            elif calib_camera_name == "ext2_cam_serial":
                calib_image_name = "exterior_image_2_left"
            else:
                raise ValueError(f"Unknown camera name: {calib_camera_name}")

            calib_image_idx = self.images_list.index(calib_image_name)
            eid_to_cam_dict[eid] = calib_image_idx

            if need_calib:
                # Camera intrinsics as [fx, fy, cx, cy]
                try:
                    fx, cx, fy, cy = intrinsics_json[eid][camera_serial]["cameraMatrix"]
                    eid_to_intr_vec[eid] = [fx, fy, cx, cy]
                except Exception:
                    # Fallback to zeros
                    eid_to_intr_vec[eid] = [0.0, 0.0, 0.0, 0.0]

                # Camera extrinsics 4x4 from [tx,ty,tz,rx,ry,rz]
                try:
                    tx, ty, tz, rx, ry, rz = extr[camera_serial]
                    rot_matrix = euler_xyz_to_rot(rx, ry, rz)
                    transform_matrix = np.eye(4, dtype=np.float32)
                    transform_matrix[:3, :3] = rot_matrix
                    transform_matrix[:3, 3] = [tx, ty, tz]
                    eid_to_extr_mat[eid] = transform_matrix.reshape(-1)
                except Exception:
                    eid_to_extr_mat[eid] = np.eye(4, dtype=np.float32).reshape(-1)

        keys = tf.constant(list(eid_to_cam_dict.keys()), dtype=tf.string)
        values = tf.constant(list(eid_to_cam_dict.values()), dtype=tf.int32)
        cam_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,  # -1 ⇒ fallback camera
        )
        print_memory_usage("After building cam_table")

        # Camera intrinsics/extrinsics lookup tables (serialize tensors to tf.string to avoid shape issues)
        intr_table = None
        extr_table = None
        if need_calib:
            calib_eids = list(eid_to_cam_dict.keys())
            intr_ser = []
            extr_ser = []
            for _eid in calib_eids:
                _intr = eid_to_intr_vec.get(_eid, [0.0, 0.0, 0.0, 0.0])
                _extr = eid_to_extr_mat.get(_eid, np.zeros((16,), dtype=np.float32))
                intr_ser.append(tf.io.serialize_tensor(tf.constant(_intr, dtype=tf.float32)).numpy())
                extr_ser.append(tf.io.serialize_tensor(tf.constant(_extr, dtype=tf.float32)).numpy())
            calib_keys = tf.constant(calib_eids, dtype=tf.string)
            intr_vals = tf.constant(intr_ser, dtype=tf.string)
            extr_vals = tf.constant(extr_ser, dtype=tf.string)
            intr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, intr_vals),
                default_value=self.default_intr_ser,
            )
            extr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, extr_vals),
                default_value=self.default_extr_ser,
            )
            print_memory_usage("After building intr_table and extr_table")
        return cam_table, intr_table, extr_table

    def build_instr_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 6. Language-instruction table (merged; episode_id → serialized [K])
        # ---------------------------------------------------------------------
        instr_cache_path = f"{metadata_path}/{self.droid_instructions_file}"
        _instr_keys_py = []
        _instr_vals_ser = []
        if tf.io.gfile.exists(instr_cache_path):
            with tf.io.gfile.GFile(instr_cache_path, "r") as fp:
                instr_index = json.load(fp)
            _instr_keys_py = list(instr_index.keys())
            for _eid in _instr_keys_py:
                _arr = instr_index[_eid]
                if not isinstance(_arr, list):
                    _arr = []
                _arr = [s for s in _arr if isinstance(s, str) and len(s) > 0]
                if len(_arr) == 0:
                    _instr_vals_ser.append(self.default_lang_value)
                else:
                    _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())
        else:
            with tf.io.gfile.GFile(f"{metadata_path}/{self.droid_language_annotations_file}", "r") as fp:
                language_annotations = json.load(fp)
            _instr_keys_py = list(language_annotations.keys())
            for _eid in _instr_keys_py:
                _v = language_annotations[_eid]
                _arr = [
                    _v.get("language_instruction1", ""),
                    _v.get("language_instruction2", ""),
                    _v.get("language_instruction3", ""),
                ]
                _arr = [s for s in _arr if len(s) > 0]
                if len(_arr) == 0:
                    _instr_vals_ser.append(b"")
                else:
                    _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())
        _instr_keys = tf.constant(_instr_keys_py, dtype=tf.string)
        _instr_vals = tf.constant(_instr_vals_ser, dtype=tf.string)
        _instr_default = tf.constant(b"", dtype=tf.string)
        instr_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(_instr_keys, _instr_vals),
            default_value=_instr_default,
        )
        print_memory_usage("After building instr_table")
        return instr_table

    def build_filter_table(self, metadata_path, use_idle_filter: bool):
        filter_table = None
        if use_idle_filter:
            with tf.io.gfile.GFile(f"{metadata_path}/{self.keep_ranges_file}", "r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            ep_keys = []
            ranges_ser = []
            for episode_key, ranges in filter_dict.items():
                arr = np.array(ranges, dtype=np.int32)  # shape [M, 2]
                ep_keys.append(episode_key)
                ranges_ser.append(tf.io.serialize_tensor(tf.constant(arr)).numpy())

            keys = tf.constant(ep_keys, dtype=tf.string)
            vals = tf.constant(ranges_ser, dtype=tf.string)
            filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys, vals),
                default_value=tf.constant(b"", dtype=tf.string),
            )
            print_memory_usage("After building filter_table")

        return filter_table

    def apply_traj_transforms(
        self,
        lang_table: tf.lookup.StaticHashTable,
        ep_table: tf.lookup.StaticHashTable,
        instr_table: tf.lookup.StaticHashTable,
        cam_table: tf.lookup.StaticHashTable,
        intr_table: tf.lookup.StaticHashTable,
        extr_table: tf.lookup.StaticHashTable,
        filter_table: tf.lookup.StaticHashTable,
        action_chunk_size: int,
        summation_steps: int,
        use_wrist_image: bool,
        use_idle_filter: bool,
        need_calib: bool,
        drop_gripper_oob: bool,
    ):
        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    traj["action_dict"]["cartesian_position"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Align lengths across modalities
            traj_len = tf.shape(actions)[0]
            episode_id = self._episode_id_from_traj(traj, ep_table)
            lang_bytes = lang_table.lookup(episode_id)
            lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
            # Language actions may include an extra terminal step; crop to match action length
            lang_tensor = lang_tensor[:traj_len]
            # Sample instruction from merged table or fallback
            instr_bytes = instr_table.lookup(episode_id)
            fallback_index = tf.random.uniform(
                (),
                minval=0,
                maxval=tf.shape(self.fallback_instructions)[0],
                dtype=tf.int32,
                seed=self.seed,
            )
            fallback_instruction = self.fallback_instructions[fallback_index]

            def _sample_from_table():
                arr = tf.io.parse_tensor(instr_bytes, out_type=tf.string)
                return tf.random.shuffle(arr, seed=self.seed)[0]

            instruction = tf.cond(
                tf.greater(tf.strings.length(instr_bytes), 0),
                _sample_from_table,
                lambda: fallback_instruction,
            )

            instruction_vec = tf.fill([tf.shape(actions)[0]], instruction)

            cam_idx = cam_table.lookup(episode_id)
            cam_images = [
                traj["observation"]["exterior_image_1_left"],
                traj["observation"]["exterior_image_2_left"],
            ]
            cam_images = tf.stack(cam_images, axis=0)  # shape (2, H, W, C)
            cam_idx_clamped = tf.clip_by_value(cam_idx, 0, tf.shape(cam_images)[0] - 1)
            exterior_img = tf.gather(cam_images, cam_idx_clamped)

            # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # exterior_img = tf.cond(
            #     tf.random.uniform(shape=[], seed=seed) > 0.5,
            #     lambda: traj["observation"]["exterior_image_1_left"],
            #     lambda: traj["observation"]["exterior_image_2_left"],
            # )
            episode_id_vec = tf.fill([traj_len], episode_id)

            _return_dict = {
                "actions": actions,
                "observation": {
                    "image_primary": exterior_img,
                    "cartesian_position": traj["observation"]["cartesian_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "episode_id": episode_id_vec,
            }

            if use_idle_filter:
                # episode_id: scalar tf.string; traj_len: int
                ranges_bytes = filter_table.lookup(episode_id)

                def _compute_mask():
                    ranges = tf.io.parse_tensor(ranges_bytes, out_type=tf.int32)  # [M, 2]
                    t = tf.range(traj_len)[:, None]  # [T, 1]
                    starts = tf.cast(ranges[:, 0][None, :], tf.int32)  # [1, M]
                    ends = tf.cast(ranges[:, 1][None, :], tf.int32)  # [1, M]
                    in_any = tf.reduce_any((t >= starts) & (t < ends), axis=1)  # [T]
                    return in_any

                passes_filter = tf.cond(
                    tf.greater(tf.strings.length(ranges_bytes), 0),
                    _compute_mask,
                    lambda: tf.zeros([traj_len], dtype=tf.bool),
                )
                _return_dict["passes_filter"] = passes_filter

            if need_calib:
                intr = tf.io.parse_tensor(intr_table.lookup(episode_id), out_type=tf.float32)  # [4]
                extr = tf.reshape(
                    tf.io.parse_tensor(extr_table.lookup(episode_id), out_type=tf.float32),
                    [4, 4],
                )  # [4,4]
                intr_b = tf.broadcast_to(intr[None, :], [traj_len, 4])
                extr_b = tf.broadcast_to(extr[None, :, :], [traj_len, 4, 4])
                _return_dict["camera_intrinsics"] = intr_b  # [traj_len, 4]
                _return_dict["camera_extrinsics"] = extr_b  # [traj_len, 4, 4]

            if use_wrist_image:
                _return_dict["observation"]["image_wrist"] = traj["observation"]["wrist_image_left"]
            return _return_dict

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)
        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)

        def group_language_actions(traj):
            """Compute per-timestep summed language actions over future steps.

            For each timestep t, we sum the language actions from t to
            t + summation_steps - 1 (capped at trajectory end). We DO NOT
            chunk the language actions; after flattening, each sample will
            have a single language string aligned to its action chunk.
            """
            traj_len = tf.shape(traj["language_actions"])[0]

            # First, create indices for summation (current + future steps)
            summation_indices = tf.broadcast_to(
                tf.range(summation_steps)[None],
                [traj_len, summation_steps],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, summation_steps],
            )

            # Cap to length of the sequence (same as chunk_actions)
            summation_indices = tf.minimum(summation_indices, traj_len - 1)

            # Gather the language actions for summation
            language_actions_to_sum = tf.gather(traj["language_actions"], summation_indices)
            # Keep unsummed window for debugging: shape [traj_len, summation_steps]
            traj["language_actions"] = language_actions_to_sum

            # if vis_dataset:
            #     grouped_images = tf.gather(traj["observation"]["image"], summation_indices)
            #     traj["observation"]["image"] = grouped_images

            #     if use_wrist_image:
            #         grouped_wrist_images = tf.gather(traj["observation"]["wrist_image"], summation_indices)
            #         traj["observation"]["wrist_image"] = grouped_wrist_images

            # Group cartesian positions for start/end projection when needed
            if need_calib:
                grouped_cart = tf.gather(traj["observation"]["cartesian_position"], summation_indices)
                traj["observation"]["cartesian_position_window"] = grouped_cart
                # camera_intrinsics: [traj_len, 4] -> [traj_len, summation_steps, 4]
                assert "camera_intrinsics" in traj, "camera_intrinsics not found in traj"
                traj["camera_intrinsics"] = tf.gather(traj["camera_intrinsics"], summation_indices)
                assert "camera_extrinsics" in traj, "camera_extrinsics not found in traj"
                # camera_extrinsics: [traj_len, 4,4] -> [traj_len, summation_steps, 4,4]
                idx = summation_indices
                extrinsics_tensor = traj["camera_extrinsics"]
                traj["camera_extrinsics"] = tf.gather(extrinsics_tensor, idx)

            # Optional: compute in-view mask using calibration if requested
            if drop_gripper_oob:
                # Expect calibration present; if missing, mark as False to be safe

                # Use start and end positions per window
                cart = traj["observation"]["cartesian_position_window"]  # [traj_len, summation_steps, 6]
                start_xyz = cart[:, 0, :3]
                end_xyz = cart[:, -1, :3]

                intr = traj["camera_intrinsics"][:, 0, :]  # [traj_len, 4]
                extr = traj["camera_extrinsics"][:, 0, :, :]  # [traj_len, 4,4]

                start_ok = project_in_bounds(start_xyz, intr, extr)
                end_ok = project_in_bounds(end_xyz, intr, extr)
                keep_vec = tf.logical_and(start_ok, end_ok)  # [traj_len]
                traj["gripper_in_view"] = keep_vec

            return traj

        self.dataset = self.dataset.traj_map(group_language_actions, self.num_parallel_calls)

    def apply_frame_filters(self, use_idle_filter: bool, drop_gripper_oob: bool):
        if use_idle_filter:

            def filter_from_dict(frame):
                return frame["passes_filter"]

            self.dataset = self.dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            self.dataset = self.dataset.map(remove_passes_filter)

        # Optional filter: drop samples where gripper projects out of the view
        if drop_gripper_oob:

            def _filter_in_view(frame):
                return frame["gripper_in_view"]

            self.dataset = self.dataset.filter(_filter_in_view)

            def _remove_in_view(frame):
                frame.pop("gripper_in_view")
                return frame

            self.dataset = self.dataset.map(_remove_in_view)

    def apply_traj_filters(self, lang_table, ep_table):
        # ------------------------------------------------------------------
        # Regex helpers for robust path/id extraction
        # ------------------------------------------------------------------

        def _id_ok(traj):
            episode_id = self._episode_id_from_traj(traj, ep_table)
            if tf.equal(episode_id, self.default_ep_value):
                return tf.constant(value=False, dtype=tf.bool)
            # Look up by episode_id (NOT episode_path). Using episode_path here would filter everything out.
            lang = lang_table.lookup(episode_id)
            if tf.equal(lang, self.default_lang_value):
                return tf.constant(value=False, dtype=tf.bool)
            return tf.logical_and(
                tf.not_equal(episode_id, self.default_ep_value),
                tf.not_equal(lang, self.default_lang_value),
            )

        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        # Prefer cheap regex path filter first, then id/lang checks
        self.dataset = self.dataset.filter(_path_ok).filter(_id_ok)

    def split_val(self, ep_table, split_seed):
        def _lab_from_episode_id(episode_id):
            """Extract lab/environment name from an episode_id.

            Example episode_id: "AUTOLab+5d05c5aa+2023-07-07-10h-18m-41s" -> "AUTOLab".
            Uses regex to avoid RaggedTensor outputs from tf.strings.split.
            """
            return tf.strings.regex_replace(episode_id, r"\+.*$", "")

        def _split_filter(traj):
            episode_id = self._episode_id_from_traj(traj, ep_table)  # scalar tf.string

            # --- Deterministic hash split ---
            salt = tf.strings.as_string(split_seed)
            if self.validation_mode == "hard":
                # Environment-level split: hold out entire labs
                lab_name = _lab_from_episode_id(episode_id)
                key = tf.strings.join([salt, lab_name])
            else:  # "easy": per-trajectory split within seen labs
                key = tf.strings.join([salt, episode_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr

            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def __init__(
        self,
        data_dir: str,
        language_action_dir: str,
        config,
        *,  # Force keyword-only arguments
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.CARTESIAN_POSITION,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # Validation support
        split_seed: int = 0,
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(
            dataset_name=config.repo_id,
            data_dir=data_dir,
            config=config,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
        )
        summation_steps = getattr(self.config, "summation_steps", 15)
        vis_dataset = getattr(self.config, "vis_dataset", False)
        use_wrist_image = getattr(self.config, "use_wrist_image", False)
        use_idle_filter = getattr(self.config, "use_idle_filter", True)
        drop_gripper_oob = getattr(self.config, "drop_gripper_oob", False)
        need_calib = bool(vis_dataset or drop_gripper_oob)
        logging.info(
            f"validation_mode: {self.validation_mode}, val_fraction: {self.val_fraction}, need_calib: {need_calib}, \
                use_wrist_image: {use_wrist_image}, summation_steps: {summation_steps}, \
                    sum_decimal: {self.config.sum_decimal}"
        )
        assert action_space == DroidActionSpace.CARTESIAN_POSITION, "CoT only supports EEF actions for now"

        if self.lang_action_dir_name in language_action_dir:
            metadata_path = language_action_dir.replace(self.lang_action_dir_name, self.metadata_path_name)
        elif self.lang_action_dir_name_base in language_action_dir:
            metadata_path = language_action_dir.replace(self.lang_action_dir_name_base, self.metadata_path_name)
        else:
            raise ValueError(f"Unknown language action directory: {language_action_dir}")

        lang_table = self.build_lang_action_table(language_action_dir)
        ep_table = self.build_lookup_table(metadata_path)
        cam_table, intr_table, extr_table = self.build_cam_tables(metadata_path, need_calib=need_calib)
        instr_table = self.build_instr_table(metadata_path)
        filter_table = self.build_filter_table(metadata_path, use_idle_filter=use_idle_filter)

        self.apply_traj_filters(lang_table=lang_table, ep_table=ep_table)
        self.split_val(ep_table=ep_table, split_seed=split_seed)

        self.apply_traj_transforms(
            lang_table=lang_table,
            ep_table=ep_table,
            instr_table=instr_table,
            cam_table=cam_table,
            intr_table=intr_table,
            extr_table=extr_table,
            filter_table=filter_table,
            action_chunk_size=action_chunk_size,
            summation_steps=summation_steps,
            use_wrist_image=use_wrist_image,
            use_idle_filter=use_idle_filter,
            need_calib=need_calib,
            drop_gripper_oob=drop_gripper_oob,
        )

        self.apply_flatten()

        self.apply_frame_filters(use_idle_filter=use_idle_filter, drop_gripper_oob=drop_gripper_oob)


class DroidCoTRldsDataset(DroidCoTRldsDatasetRaw):
    def __init__(
        self,
        data_dir,
        language_action_dir,
        batch_size,
        shuffle,
        action_chunk_size,
        action_space,
        shuffle_buffer_size,
        split_seed,
        max_samples,
        seed,
        config,
        split,
    ):
        # Initialize the base class with only the raw kwargs
        super().__init__(
            data_dir,
            language_action_dir,
            config,
            action_chunk_size=action_chunk_size,
            action_space=action_space,
            shuffle_buffer_size=shuffle_buffer_size,
            split_seed=split_seed,
            max_samples=max_samples,
            seed=seed,
            split=split,
        )

        # Apply common shuffling/take/cache behavior
        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=self.want_val,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        # Apply frame transforms via shared decode helper
        self.apply_frame_transforms(use_wrist_image=config.use_wrist_image)

        # Batch and prefetch
        self.dataset = batch_prefetch(self.dataset, batch_size)

    def apply_frame_transforms(self, use_wrist_image: bool):
        # Retained for compatibility; use shared helper
        decode_fn = make_decode_images_fn(
            primary_key="image_primary",
            wrist_key="image_wrist",
            use_wrist_image=use_wrist_image,
        )
        self.dataset = self.dataset.frame_map(decode_fn, self.num_parallel_calls)

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        # Approximate size after filtering
        return 20_000_000


class SingleOXECoTRldsDataset(SingleCoTRldsDatasetRaw):
    """

    New features:
    - Language-action lookup (record_id -> serialized tf.string tensor) from TFRecords
    - Action chunking with future-horizon padding at sequence end
    - Language-action grouping into future windows (`summation_steps`)
    - Optional idle-chunk filtering
    """

    REQUIRED_KEYS: ClassVar[set[str]] = {"observation", "action"}

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        config,
        dataset_kwargs: dict,
        *,  # Force keyword-only arguments
        action_chunk_size: int = 16,
        summation_steps: int = 15,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        split_seed: int = 0,
        seed: int = 0,
        split: str = "train",
    ):
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            config=config,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
        )
        self.standardize_fn = dataset_kwargs["standardize_fn"]
        self.image_obs_keys = dataset_kwargs["image_obs_keys"]
        self.depth_obs_keys = dataset_kwargs["depth_obs_keys"]
        self.state_obs_keys = dataset_kwargs["state_obs_keys"]
        self.language_key = dataset_kwargs["language_key"]
        self.action_proprio_normalization_type = dataset_kwargs["action_proprio_normalization_type"]
        self.absolute_action_mask = dataset_kwargs["absolute_action_mask"]
        self.action_normalization_mask = dataset_kwargs["action_normalization_mask"]
        dataset_frame_transform_kwargs = dataset_kwargs["dataset_frame_transform_kwargs"]
        if self.depth_obs_keys is None:
            self.depth_obs_keys = {}
        if self.image_obs_keys is None:
            self.image_obs_keys = {}
        if self.language_key is not None:
            self.REQUIRED_KEYS.add(self.language_key)

        self.apply_restructure()
        dataset_statistics = get_dataset_statistics(
            self.dataset,
            hash_dependencies=(
                str(self.builder.info),
                str(self.state_obs_keys),
                inspect.getsource(self.standardize_fn) if self.standardize_fn is not None else "",
            ),
            save_dir=self.builder.data_dir,
        )
        dataset_statistics = tree_map(np.array, dataset_statistics)
        if self.action_normalization_mask is not None and "ego" not in self.dataset_name:
            if len(self.action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
                raise ValueError(
                    f"Length of skip_normalization_mask ({len(self.action_normalization_mask)}) "
                    f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
                )
            dataset_statistics["action"]["mask"] = np.array(self.action_normalization_mask)
        self.dataset_statistics = dataset_statistics
        self.apply_traj_filters()
        self.split_val(split_seed=split_seed)
        self.apply_traj_transforms(action_chunk_size=action_chunk_size, summation_steps=summation_steps)
        self.apply_flatten()
        self.apply_per_dataset_frame_filters(**dataset_frame_transform_kwargs)

    def apply_restructure(self):
        def restructure(traj):
            # apply a standardization function, if provided
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)

            if not all(k in traj for k in self.REQUIRED_KEYS):
                raise ValueError(
                    f"Trajectory is missing keys: {self.REQUIRED_KEYS - set(traj.keys())}. Did you write a `standardize_fn`?"
                )

            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            for new, old in self.image_obs_keys.items():
                # if new == "primary":
                #     img_key = "image"
                # elif new == "wrist":
                #     img_key = "wrist_image"
                # else:
                #     raise ValueError(f"Unknown image key: {new}")
                img_key = f"image_{new}"
                if old is None:
                    new_obs[img_key] = tf.repeat("", traj_len)  # padding
                else:
                    new_obs[img_key] = old_obs[old]

            if self.state_obs_keys:
                new_obs["proprio"] = tf.concat(
                    [
                        (
                            tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                            if key is None
                            else tf.cast(old_obs[key], tf.float32)
                        )
                        for key in self.state_obs_keys
                    ],
                    axis=1,
                )

            # add timestep info
            # new_obs["timestep"] = tf.range(traj_len)

            # extracts `language_key` into the "task" dict
            task = {}
            if self.language_key is not None:
                if traj[self.language_key].dtype != tf.string:
                    raise ValueError(
                        f"Language key {self.language_key} has dtype {traj[self.language_key].dtype}, but it must be tf.string."
                    )
                raw_language_instruction = traj.pop(self.language_key)

                # episode_id = self._episode_id_from_traj(traj, ep_table)
                # lang_bytes = lang_table.lookup(episode_id)
                # lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
                # # Language actions may include an extra terminal step; crop to match action length
                # lang_tensor = lang_tensor[:traj_len]
                # # Sample instruction from merged table or fallback
                fallback_index = tf.random.uniform(
                    (),
                    minval=0,
                    maxval=tf.shape(self.fallback_instructions)[0],
                    dtype=tf.int32,
                    seed=self.seed,
                )
                fallback_instruction = self.fallback_instructions[fallback_index]

                def _sample_from_table():
                    arr = raw_language_instruction
                    return tf.random.shuffle(arr, seed=self.seed)[0]

                instruction = tf.cond(
                    tf.greater(tf.strings.length(raw_language_instruction), 0),
                    _sample_from_table,
                    lambda: fallback_instruction,
                )

                instruction_vec = tf.fill([tf.shape(traj["action"])[0]], instruction)
                task["language_instruction"] = instruction_vec

            # Build a deterministic per-trajectory identifier using a strong hash
            # of the dataset name and the serialized action tensor. This avoids
            # relying on per-dataset metadata with inconsistent schemas.
            max_steps = 128
            action_for_hash = tf.cond(
                max_steps >= traj_len,
                lambda: traj["action"],
                lambda: tf.concat([traj["action"][:64], traj["action"][-64:]], axis=0),
            )
            serialized_action = tf.io.serialize_tensor(action_for_hash)
            name_tensor = tf.constant(self.dataset_name, dtype=tf.string)
            sep1 = tf.constant("::", dtype=tf.string)
            sep2 = tf.constant("-", dtype=tf.string)
            to_hash = tf.strings.join([name_tensor, sep1, serialized_action])
            hashed = tf.strings.to_hash_bucket_strong(to_hash, 2147483647, key=[self.seed, 1337])
            traj_uid = tf.strings.join([name_tensor, sep2, tf.strings.as_string(hashed)])

            traj = {
                "observation": new_obs,
                "task": task,
                "action": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": tf.repeat(traj_uid, traj_len),
            }

            if self.absolute_action_mask is not None:
                traj["absolute_action_mask"] = tf.tile(
                    tf.convert_to_tensor(self.absolute_action_mask, dtype=tf.bool)[None],
                    [traj_len, 1],
                )

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def apply_traj_filters(self):
        def is_nonzero_length(traj):
            return tf.shape(traj["action"])[0] > 0

        # self.dataset = self.dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

        self.dataset = self.dataset.filter(is_nonzero_length)

    def split_val(self, split_seed):
        def _split_filter(traj):
            salt = tf.strings.as_string(split_seed)
            # Use the per-trajectory identifier (constant along time) to split deterministically
            traj_id = traj["trajectory_id"][0]
            key = tf.strings.join([salt, traj_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr

            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def apply_traj_transforms(self, action_chunk_size: int, summation_steps: int):
        """
        Compare to original transforms, we omit the following:
        - skip_unlabeled
        - max_action
        - max_proprio
        - goal_relabeling
        - drop_goal_or_instruction
        - subsample_length
        """
        self.dataset = self.dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=self.dataset_statistics,
                normalization_type=self.action_proprio_normalization_type,
            ),
            self.num_parallel_calls,
        )

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["action"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["action"] = tf.gather(traj["action"], action_chunk_indices)
            return traj

        # self.dataset = self.dataset.traj_map(traj_transforms.add_pad_mask_dict, self.num_parallel_calls)

        # self.dataset = self.dataset.traj_map(
        #     partial(
        #         traj_transforms.chunk_act_obs,
        #         window_size=action_chunk_size,
        #         future_action_window_size=0,
        #     ),
        #     self.num_parallel_calls,
        # )
        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)

        def group_language_actions(traj):
            """Compute per-timestep summed language actions over future steps.

            For each timestep t, we sum the language actions from t to
            t + summation_steps - 1 (capped at trajectory end). We DO NOT
            chunk the language actions; after flattening, each sample will
            have a single language string aligned to its action chunk.
            """
            traj_len = tf.shape(traj["action"])[0]

            # First, create indices for summation (current + future steps)
            summation_indices = tf.broadcast_to(
                tf.range(summation_steps)[None],
                [traj_len, summation_steps],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, summation_steps],
            )

            # Cap to length of the sequence (same as chunk_actions)
            summation_indices = tf.minimum(summation_indices, traj_len - 1)

            # Gather the language actions for summation
            language_actions_to_sum = tf.gather(traj["action"], summation_indices)
            # Keep unsummed window for debugging: shape [traj_len, summation_steps]
            traj["language_actions"] = language_actions_to_sum

            return traj

        self.dataset = self.dataset.traj_map(group_language_actions, self.num_parallel_calls)

    def apply_per_dataset_frame_filters(
        self,
        chunk_filter_fn: Callable | None = None,
    ):
        """
        Optionally applied *per-dataset* transforms that happen at a frame level.

        Args:
            chunk_filter_fn (callable, optional): Filter function for chunks.
        """
        if chunk_filter_fn:
            self.dataset = self.dataset.filter(chunk_filter_fn)


class OXECoTRldsDatasetsRaw:
    def __init__(
        self,
        config,
        data_root_dir: str,
        data_mix: str,
        resize_resolution: tuple[int, int],
        shuffle: bool = True,  # noqa: FBT001, FBT002
        action_chunk_size: int = 16,
        split_seed: int = 0,
        max_samples: int | None = None,
        seed: int = 0,
        split: str = "train",
        shuffle_buffer_size: int = 250_000,
        window_size: int = 10,
        image_aug: bool = False,  # noqa: FBT001, FBT002
        balance_weights: bool = True,  # noqa: FBT001, FBT002
        traj_transform_threads: int | None = None,
        traj_read_threads: int | None = None,
        **kwargs,
    ):
        # Configure RLDS Dataset(s)
        if data_mix in OXE_NAMED_MIXTURES:  # noqa: SIM108
            mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(data_mix, 1.0)]

        # fmt: off
        dataset_kwargs_list, sample_weights = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        traj_transform_kwargs={
            "window_size": action_chunk_size,                            # If we wanted to feed / predict more than one step
            "future_action_window_size": 0,                        # For action chunking
            "skip_unlabeled": True,                                # Skip trajectories without language labels
        },
        frame_transform_kwargs={
            "resize_size": resize_resolution,
            "num_parallel_calls": tf.data.AUTOTUNE,                          # For CPU-intensive ops (decoding, resizing, etc.)
        },
        # If applicable, enable image augmentations
        if image_aug:
            image_augment_kwargs = {
                "random_resized_crop": dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),  # noqa: C408
                "random_brightness": [0.2],
                "random_contrast": [0.8, 1.2],
                "random_saturation": [0.8, 1.2],
                "random_hue": [0.05],
                "augment_order": [
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            }
            frame_transform_kwargs["image_augment_kwargs"] = image_augment_kwargs

        if not sample_weights:
            sample_weights = [1.0] * len(dataset_kwargs_list)

        if len(sample_weights) != len(dataset_kwargs_list):
            raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

        # Check valid `traj_transform_kwargs` and `frame_transform_kwargs`
        if traj_transform_kwargs is None or frame_transform_kwargs is None:
            raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

        # Allocate Threads based on Weights
        threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
        reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

        logging.info("Threads per Dataset: %s", threads_per_dataset)
        logging.info("Reads per Dataset: %s", reads_per_dataset)

        datasets, dataset_sizes, all_dataset_statistics = [], [], {}
        logging.info("Constructing datasets...")
        for dataset_kwargs, threads, reads in zip(  # noqa: B905
                dataset_kwargs_list,
                threads_per_dataset,
                reads_per_dataset,
            ):
            # Pass only accepted args to SingleOXECoTRldsDataset
            ds = SingleOXECoTRldsDataset(
                dataset_name=dataset_kwargs["name"],
                data_dir=dataset_kwargs["data_dir"],
                config=config,
                dataset_kwargs=dataset_kwargs,
                action_chunk_size=action_chunk_size,
                num_parallel_reads=reads,
                num_parallel_calls=threads,
                split_seed=split_seed,
                seed=seed,
                split=split,
            )
            datasets.append(ds.dataset)
            dataset_statistics = ds.dataset_statistics
            dataset_sizes.append(dataset_statistics["num_transitions"])
            all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

        # Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        # primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
        primary_dataset_indices = np.array(list(range(len(sample_weights))))

        # Balance and Normalize Weights
        if balance_weights:
            sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
        unnormalized_sample_weights = sample_weights.copy()
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)
        pprint_data_mixture(dataset_kwargs_list, sample_weights)

        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

        self.sample_weights = sample_weights
        self.unnormalized_sample_weights = unnormalized_sample_weights
        self.dataset_statistics = all_dataset_statistics
        self.dataset_length = dataset_len
        self.datasets = datasets

    # def apply_frame_transforms(
    #     self,
    #     train: bool,
    #     image_augment_kwargs: dict | dict[str, dict] = {},
    #     resize_size: tuple[int, int] | dict[str, tuple[int, int]] = {},
    #     depth_resize_size: tuple[int, int] | dict[str, tuple[int, int]] = {},
    #     num_parallel_calls: int = tf.data.AUTOTUNE,
    # ):
    #     # Convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    #     # it to the chunked "observation" dict as well as the non-chunked "task" dict
    #     def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
    #         frame["task"] = fn(frame["task"])
    #         frame["observation"] = dl.vmap(fn)(frame["observation"])
    #         return frame

    #     # Decode + resize images (and depth images)
    #     self.dataset = self.dataset.frame_map(
    #         partial(
    #             apply_obs_transform,
    #             partial(obs_transforms.decode_and_resize, resize_size=resize_size, depth_resize_size=depth_resize_size),
    #         ),
    #         num_parallel_calls,
    #     )

    #     if train:
    #         # Augment all images with the same seed, skipping padding images
    #         def aug(frame: dict):
    #             seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
    #             aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
    #             return apply_obs_transform(aug_fn, frame)

    #         self.dataset = self.dataset.frame_map(aug, num_parallel_calls)


class OXECoTRldsDatasets(OXECoTRldsDatasetsRaw):
    def __init__(self, **kwargs):
        # Filter kwargs using inspect
        top_kwargs, _ = _filter_kwargs_for(
            lambda *,
            batch_size,
            shuffle=True,
            max_samples=None,
            shuffle_buffer_size=250_000,
            seed=0,
            use_wrist_image=False,
            split="train": None,
            kwargs,
        )
        for req in ("batch_size",):
            if req not in top_kwargs:
                raise ValueError(f"Missing required arguments: ['{req}']")
        raw_kwargs, _ = _filter_kwargs_for(OXECoTRldsDatasetsRaw.__init__, kwargs)
        for req in ("config", "data_root_dir", "data_mix", "resize_resolution", "action_chunk_size", "seed", "split"):
            if req not in raw_kwargs:
                raise ValueError(f"Missing required arguments: ['{req}']")
        consumed_keys = set(top_kwargs.keys()) | set(raw_kwargs.keys())
        unexpected = set(kwargs.keys()) - consumed_keys
        if unexpected:
            raise ValueError(f"Unexpected arguments for OXECoTRldsDatasets: {sorted(unexpected)}")

        batch_size = int(top_kwargs["batch_size"])  # required
        shuffle = bool(top_kwargs.get("shuffle", True))
        max_samples = top_kwargs.get("max_samples")
        shuffle_buffer_size = int(top_kwargs.get("shuffle_buffer_size", 250_000))
        seed = int(top_kwargs.get("seed", 0))
        use_wrist_image = bool(top_kwargs.get("use_wrist_image", False))
        want_val = raw_kwargs.get("split", "train") == "val"

        super().__init__(**raw_kwargs)

        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        decode_fn = make_decode_images_fn(
            primary_key="image_primary",
            wrist_key="image_wrist",
            use_wrist_image=use_wrist_image,
        )
        self.dataset = self.dataset.frame_map(decode_fn, self.num_parallel_calls)

        self.dataset = batch_prefetch(self.dataset, batch_size)

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        return self.dataset_length


class CombinedCoTRldsDataset:
    def __init__(
        self,
        **kwargs,
    ):
        # Filter using inspect
        def _top_sig(
            *,
            batch_size,
            droid_weight,
            shuffle=True,
            max_samples=None,
            shuffle_buffer_size=250_000,
            seed=0,
            split="train",
            use_wrist_image=False,
        ):
            return None

        top_kwargs, _ = _filter_kwargs_for(_top_sig, kwargs)
        for req in ("batch_size", "droid_weight"):
            if req not in top_kwargs:
                raise ValueError(f"Missing required arguments: ['{req}']")

        droid_kwargs, _ = _filter_kwargs_for(DroidCoTRldsDatasetRaw.__init__, kwargs)
        for req in ("data_dir", "language_action_dir", "config"):
            if req not in droid_kwargs:
                raise ValueError(f"Missing required arguments: ['{req}']")

        oxe_kwargs, _ = _filter_kwargs_for(OXECoTRldsDatasetsRaw.__init__, kwargs)
        for req in ("config", "data_root_dir", "data_mix", "resize_resolution", "action_chunk_size", "seed", "split"):
            if req not in oxe_kwargs:
                raise ValueError(f"Missing required arguments: ['{req}']")

        consumed_keys = set(top_kwargs.keys()) | set(droid_kwargs.keys()) | set(oxe_kwargs.keys())
        unexpected = set(kwargs.keys()) - consumed_keys
        if unexpected:
            raise ValueError(f"Unexpected arguments for CombinedCoTRldsDataset: {sorted(unexpected)}")

        use_wrist_image = bool(top_kwargs.get("use_wrist_image", False))
        batch_size = int(top_kwargs["batch_size"])  # required
        shuffle = bool(top_kwargs.get("shuffle", True))
        max_samples = top_kwargs.get("max_samples")
        shuffle_buffer_size = int(top_kwargs.get("shuffle_buffer_size", 250_000))
        seed = int(top_kwargs.get("seed", 0))
        split = top_kwargs.get("split", oxe_kwargs.get("split", droid_kwargs.get("split", "train")))
        want_val = split == "val"
        droid_weight = float(top_kwargs["droid_weight"])  # required

        # Build sub-datasets with only their required args
        droid = DroidCoTRldsDatasetRaw(**droid_kwargs)
        oxe = OXECoTRldsDatasetsRaw(**oxe_kwargs)

        all_datasets = [*oxe.datasets, droid.dataset]
        sample_weights = [*oxe.unnormalized_sample_weights, droid_weight]

        logging.info("Interleaving datasets...")
        self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(all_datasets, sample_weights)

        # Apply common finalization
        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        # Apply frame transforms via shared helper
        logging.info("Applying frame transforms on dataset...")
        decode_fn = make_decode_images_fn(
            primary_key="image_primary",
            wrist_key="image_wrist",
            use_wrist_image=use_wrist_image,
        )
        self.dataset = self.dataset.frame_map(decode_fn, tf.data.AUTOTUNE)
        self.dataset.sample_weights = sample_weights
        self.dataset_length = oxe.dataset_length
        self.dataset_statistics = oxe.dataset_statistics

        self.dataset = batch_prefetch(self.dataset, batch_size)

    def apply_frame_transforms(self, use_wrist_image: bool):
        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            def _decode_single(img_bytes):
                return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)

            traj["observation"]["image_primary"] = _decode_single(traj["observation"]["image_primary"])
            if use_wrist_image:
                traj["observation"]["image_wrist"] = _decode_single(traj["observation"]["image_wrist"])
            return traj

        self.dataset = self.dataset.frame_map(decode_images, tf.data.AUTOTUNE)

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch
