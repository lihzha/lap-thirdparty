"""
RLDS-based data loader for DROID with CoT-style language actions.

This revision merges the standalone pre-processing logic from
`process_dataset.py` directly into the `DroidCoTRldsDataset` class so
that everything from raw RLDS → ready-to-train batches happens in one
place.  The main additions are:

*   **Episode calibration lookup** – we pre-compute, on initialisation,
    which exterior camera (ext 1 vs ext 2) should be used for every
    DROID episode based on the extrinsics files.  At run-time the correct
    image is selected without expensive Python branching inside the
    TensorFlow graph.
*   **Language-action loading** – natural-language low-level action
    strings are loaded from the `<episode_id>_language_action.json`
    files and stored in a `tf.lookup.StaticHashTable`, so they can be
    joined with the trajectory entirely on the TF side.  The final batch
    therefore contains a `language_actions` tensor of shape
    `(B, T_chunk)` aligned with the action chunk.
*   **Restructure pass rewritten** – now relies on the lookup tables for
    both the calibrated image key and language actions; the remaining
    logic stays in TF ops (no `py_function`) so the pipeline is fully
    traceable and fast.

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

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from enum import auto
import json
import logging
import os
from pathlib import Path

import jax
import numpy as np
import openpi.training.config as _config
import psutil

import openpi_cot.training.config as _config

METADATA_PATH = "/n/fs/robot-data/vlm-syn/droid"
IMAGE_LIST = [
    "exterior_image_1_left",
    "exterior_image_2_left",
]


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


def _maybe(builder_fn: Callable[[], object], fallback_fn: Callable[[], object]):
    try:
        return builder_fn()
    except Exception:
        return fallback_fn()


# Optional imports: OXE configs/mixes/transforms
try:
    from openpi_cot.training.oxe_utils.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS  # type: ignore
    from openpi_cot.training.oxe_utils.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES  # type: ignore
    from openpi_cot.training.oxe_utils.data.oxe.oxe_standardization_transforms import (
        OXE_STANDARDIZATION_TRANSFORMS,  # type: ignore
    )
except Exception:
    OXE_DATASET_CONFIGS = {}
    OXE_NAMED_MIXES = {}
    OXE_STANDARDIZATION_TRANSFORMS = {}


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_POSITION = auto()


class DroidCoTRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        language_action_dir: str,
        config: _config.DataConfig,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.CARTESIAN_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # Validation support
        split_seed: int = 0,
        # Overfitting support: cap number of flattened samples (after shuffle)
        max_samples: int | None = None,
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        assert action_space == DroidActionSpace.CARTESIAN_POSITION, "CoT only supports EEF actions for now"
        validation_mode = getattr(config, "validation_mode", "easy")
        summation_steps = getattr(config, "summation_steps", 15)
        val_fraction = getattr(config, "val_fraction", 0.02)
        vis_dataset = getattr(config, "vis_dataset", False)
        use_wrist_image = getattr(config, "use_wrist_image", False)
        apply_idle_filter = getattr(config, "apply_idle_filter", True)
        drop_gripper_oob = getattr(config, "drop_gripper_oob", False)

        logging.info(
            f"validation_mode: {validation_mode}, val_fraction: {val_fraction}, vis_dataset: {vis_dataset}, \
                use_wrist_image: {use_wrist_image}, summation_steps: {summation_steps}, max_samples: {max_samples}, \
                    sum_decimal: {config.sum_decimal}, left_pad: {config.left_pad}, include_decimal_point: {config.include_decimal_point}, \
                        batch_size: {batch_size}"
        )

        # ------------------------------------------------------------------
        # Global seeding for reproducibility across dataset ops
        # ------------------------------------------------------------------
        tf.random.set_seed(seed)
        # try:
        #     # TF 2.12+: enable deterministic kernels where available
        #     tf.config.experimental.enable_op_determinism()
        # except Exception:
        #     pass

        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        if "droid-lang-actions" in language_action_dir:
            METADATA_PATH = language_action_dir.replace("droid-lang-actions", "metadata")
        elif "droid-base-lang-actions" in language_action_dir:
            METADATA_PATH = language_action_dir.replace("droid-base-lang-actions", "metadata")
        else:
            raise ValueError(f"Unknown language action directory: {language_action_dir}")
        # ------------------------------------------------------------------
        # Validation difficulty levels
        #   - "easy": train/val do NOT share trajectories (split by episode_id); labs can overlap
        #   - "hard": train/val come from different labs (split by lab prefix in episode_id)
        # Aliases for backward compatibility: {"easier", "medium"} -> "easy"; {"harder"} -> "hard"
        # ------------------------------------------------------------------
        validation_mode = (validation_mode or "easy").lower()
        assert validation_mode in {"easy", "hard"}, (
            f"validation_mode must be one of 'easy', 'hard'; got: {validation_mode}"
        )

        # ---------------------------------------------------------------------
        # 1. TF-DS builder + base dataset
        # ---------------------------------------------------------------------
        # Resolve autotune sentinels now that TF is imported
        if num_parallel_reads == -1:
            num_parallel_reads = tf.data.AUTOTUNE
        if num_parallel_calls == -1:
            num_parallel_calls = tf.data.AUTOTUNE

        want_val = split == "val"

        builder = tfds.builder(config.repo_id, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=bool(want_val),  # shuffle at file/shard level for first-level randomness
            num_parallel_reads=num_parallel_reads,
        )

        dataset = dataset.shard(jax.process_count(), jax.process_index())
        # Enforce deterministic order for reproducibility and increase host-side parallelism
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True
        cpu_count = psutil.cpu_count(logical=True) or 16
        opts.experimental_threading.private_threadpool_size = int(max(16, cpu_count))
        dataset = dataset.with_options(opts)
        # Repeat early to increase interleaving across files/episodes
        if (not want_val) and (max_samples is None):
            dataset = dataset.repeat()

        # ---------------------------------------------------------------------
        # 2. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        FEATURES = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(record):
            ex = tf.io.parse_single_example(record, FEATURES)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        files = tf.io.gfile.glob(f"{language_action_dir}/tfds_language_actions-*.tfrecord.gz")
        ds = (
            tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
            .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        episodes, lang_serialized = [], []
        for ep_id, lang in ds:
            episodes.append(ep_id.numpy().decode())
            lang_serialized.append(tf.io.serialize_tensor(lang).numpy())

        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        default_lang_value = tf.constant(b"", dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_lang_value,
        )

        print_memory_usage("After building lang_table")

        # ---------------------------------------------------------------------
        # 4. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{METADATA_PATH}/episode_id_to_path.json", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        default_ep_value = tf.constant("", dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_ep_value,
        )
        print_memory_usage("After building ep_table")

        # ---------------------------------------------------------------------
        # 5. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{METADATA_PATH}/cam2base_extrinsics.json", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{METADATA_PATH}/camera_serials.json", "r") as fp:
            camera_serials = json.load(fp)
        need_calib = bool(vis_dataset or drop_gripper_oob)
        if need_calib:
            with tf.io.gfile.GFile(f"{METADATA_PATH}/intrinsics.json", "r") as fp:
                intrinsics_json = json.load(fp)
            eid_to_intr_vec = {}
            eid_to_extr_mat = {}

            def _euler_xyz_to_rot(rx, ry, rz):
                # Build rotation matrix from XYZ intrinsic rotations
                cx, sx = np.cos(rx), np.sin(rx)
                cy, sy = np.cos(ry), np.sin(ry)
                cz, sz = np.cos(rz), np.sin(rz)
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
                return Rz @ Ry @ Rx

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

            calib_image_idx = IMAGE_LIST.index(calib_image_name)
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
                    R = _euler_xyz_to_rot(rx, ry, rz)
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R
                    T[:3, 3] = [tx, ty, tz]
                    eid_to_extr_mat[eid] = T.reshape(-1)
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
            default_intr_ser = tf.io.serialize_tensor(tf.zeros([4], tf.float32))
            default_extr_ser = tf.io.serialize_tensor(tf.reshape(tf.eye(4, dtype=tf.float32), [-1]))
            intr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, intr_vals),
                default_value=default_intr_ser,
            )
            extr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, extr_vals),
                default_value=default_extr_ser,
            )

        # ---------------------------------------------------------------------
        # 6. Language-instruction table (merged; episode_id → serialized [K])
        # ---------------------------------------------------------------------
        instr_cache_path = f"{METADATA_PATH}/droid_instructions.json"
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
                    _instr_vals_ser.append(b"")
                else:
                    _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())
        else:
            with tf.io.gfile.GFile(f"{METADATA_PATH}/droid_language_annotations.json", "r") as fp:
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

        print_memory_usage("After building instr_table")

        # ------------------------------------------------------------------
        # Regex helpers for robust path/id extraction
        # ------------------------------------------------------------------
        def _extract_episode_path_from_file_path(file_path):
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

        def _episode_id_from_traj(traj):
            """Lookup episode_id from trajectory metadata using regex extraction."""
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            episode_path = _extract_episode_path_from_file_path(file_path)
            return ep_table.lookup(episode_path)

        def _lab_from_episode_id(episode_id):
            """Extract lab/environment name from an episode_id.

            Example episode_id: "AUTOLab+5d05c5aa+2023-07-07-10h-18m-41s" -> "AUTOLab".
            Uses regex to avoid RaggedTensor outputs from tf.strings.split.
            """
            return tf.strings.regex_replace(episode_id, r"\+.*$", "")

        def _id_ok(traj):
            episode_id = _episode_id_from_traj(traj)
            if tf.equal(episode_id, default_ep_value):
                return tf.constant(value=False, dtype=tf.bool)
            # Look up by episode_id (NOT episode_path). Using episode_path here would filter everything out.
            lang = lang_table.lookup(episode_id)
            if tf.equal(lang, default_lang_value):
                return tf.constant(value=False, dtype=tf.bool)
            return tf.logical_and(
                tf.not_equal(episode_id, default_ep_value),
                tf.not_equal(lang, default_lang_value),
            )

        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        # Prefer cheap regex path filter first, then id/lang checks
        dataset = dataset.filter(_path_ok).filter(_id_ok)

        def _split_filter(traj):
            episode_id = _episode_id_from_traj(traj)  # scalar tf.string

            # --- Deterministic hash split ---
            salt = tf.strings.as_string(split_seed)
            if validation_mode == "hard":
                # Environment-level split: hold out entire labs
                lab_name = _lab_from_episode_id(episode_id)
                key = tf.strings.join([salt, lab_name])
            else:  # "easy": per-trajectory split within seen labs
                key = tf.strings.join([salt, episode_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(val_fraction * 1000), tf.int64)
            is_val = bucket < thr

            return is_val if want_val else tf.logical_not(is_val)

        dataset = dataset.filter(_split_filter)

        # Set determinism for validation
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(want_val)
        dataset = dataset.with_options(opts)

        if apply_idle_filter:
            with tf.io.gfile.GFile(f"{METADATA_PATH}/keep_ranges_1_0_1.json", "r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in filter_dict.items():
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
                default_value=False,
            )
            logging.info("Filter hash table initialized")

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
            episode_id = _episode_id_from_traj(traj)
            lang_bytes = lang_table.lookup(episode_id)
            lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
            # Language actions may include an extra terminal step; crop to match action length
            lang_tensor = lang_tensor[:traj_len]
            # Sample instruction from merged table or fallback
            instr_bytes = instr_table.lookup(episode_id)
            fallback_index = tf.random.uniform(
                (),
                minval=0,
                maxval=tf.shape(fallback_instructions)[0],
                dtype=tf.int32,
                seed=seed,
            )
            fallback_instruction = fallback_instructions[fallback_index]

            def _sample_from_table():
                arr = tf.io.parse_tensor(instr_bytes, out_type=tf.string)
                return tf.random.shuffle(arr, seed=seed)[0]

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

            # TODO: use wrist camera image or not
            # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # exterior_img = tf.cond(
            #     tf.random.uniform(shape=[], seed=seed) > 0.5,
            #     lambda: traj["observation"]["exterior_image_1_left"],
            #     lambda: traj["observation"]["exterior_image_2_left"],
            # )
            episode_id_vec = tf.fill([traj_len], episode_id)

            # Deserialize intrinsics/extrinsics and broadcast across trajectory length
            if need_calib:
                intr = tf.io.parse_tensor(intr_table.lookup(episode_id), out_type=tf.float32)  # [4]
                extr = tf.reshape(
                    tf.io.parse_tensor(extr_table.lookup(episode_id), out_type=tf.float32),
                    [4, 4],
                )  # [4,4]
                intr_b = tf.broadcast_to(intr[None, :], [traj_len, 4])
                extr_b = tf.broadcast_to(extr[None, :, :], [traj_len, 4, 4])

            _return_dict = {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "cartesian_position": traj["observation"]["cartesian_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "episode_id": episode_id_vec,
            }

            if apply_idle_filter:
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + tf.as_string(tf.range(traj_len))
                )
                passes_filter = filter_table.lookup(step_id)
                _return_dict["passes_filter"] = passes_filter

            if need_calib:
                _return_dict["camera_intrinsics"] = intr_b  # [traj_len, 4]
                _return_dict["camera_extrinsics"] = extr_b  # [traj_len, 4, 4]
            if use_wrist_image:
                _return_dict["observation"]["wrist_image"] = traj["observation"]["wrist_image_left"]
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

        dataset = dataset.traj_map(restructure, num_parallel_calls)
        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

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
            if vis_dataset or drop_gripper_oob:
                grouped_cart = tf.gather(traj["observation"]["cartesian_position"], summation_indices)
                traj["observation"]["cartesian_position_window"] = grouped_cart

            # Also group broadcast calibration to align with the same windowed time dimension when needed
            if vis_dataset or drop_gripper_oob:
                # camera_intrinsics: [traj_len, 4] -> [traj_len, summation_steps, 4]
                assert "camera_intrinsics" in traj, "camera_intrinsics not found in traj"
                traj["camera_intrinsics"] = tf.gather(traj["camera_intrinsics"], summation_indices)
                assert "camera_extrinsics" in traj, "camera_extrinsics not found in traj"
                # camera_extrinsics: [traj_len, 4,4] -> [traj_len, summation_steps, 4,4]
                idx = summation_indices
                T = traj["camera_extrinsics"]
                traj["camera_extrinsics"] = tf.gather(T, idx)

            # Optional: compute in-view mask using calibration if requested
            if drop_gripper_oob:
                # Expect calibration present; if missing, mark as False to be safe
                def _project_in_bounds(xyz, intr4, extr44):
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

                # Use start and end positions per window
                cart = traj["observation"]["cartesian_position_window"]  # [traj_len, summation_steps, 6]
                start_xyz = cart[:, 0, :3]
                end_xyz = cart[:, -1, :3]

                intr = traj["camera_intrinsics"][:, 0, :]  # [traj_len, 4]
                extr = traj["camera_extrinsics"][:, 0, :, :]  # [traj_len, 4,4]

                start_ok = _project_in_bounds(start_xyz, intr, extr)
                end_ok = _project_in_bounds(end_xyz, intr, extr)
                keep_vec = tf.logical_and(start_ok, end_ok)  # [traj_len]
                traj["gripper_in_view"] = keep_vec

            return traj

        dataset = dataset.traj_map(group_language_actions, num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        if apply_idle_filter:

            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)

        else:

            def filter_idle(traj):
                """Filter out chunks with idle actions.
                --> we filter if at least first half of chunk does not move.
                """
                if action_space == DroidActionSpace.CARTESIAN_POSITION:
                    # Compute delta to first position in action chunk
                    return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

            dataset = dataset.filter(filter_idle)

        # Optional filter: drop samples where gripper projects out of the view
        if drop_gripper_oob:

            def _filter_in_view(frame):
                return frame["gripper_in_view"]

            dataset = dataset.filter(_filter_in_view)

            def _remove_in_view(frame):
                frame.pop("gripper_in_view")
                return frame

            dataset = dataset.map(_remove_in_view)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            def _decode_single(img_bytes):
                return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)

            # if vis_dataset:
            #     traj["observation"]["image"] = tf.map_fn(
            #         _decode_single,
            #         traj["observation"]["image"],
            #         fn_output_signature=tf.uint8,
            #     )
            #     if use_wrist_image:
            #         traj["observation"]["wrist_image"] = tf.map_fn(
            #             _decode_single,
            #             traj["observation"]["wrist_image"],
            #             fn_output_signature=tf.uint8,
            #         )
            # else:
            traj["observation"]["image"] = _decode_single(traj["observation"]["image"])
            if use_wrist_image:
                traj["observation"]["wrist_image"] = _decode_single(traj["observation"]["wrist_image"])
            return traj

        # Only shuffle during training; validation should be deterministic and cheaper
        if (not want_val) and shuffle and max_samples is None:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

        # If requested, cap the number of flattened samples for overfitting tests.
        # We cache the capped set so repeating yields the same fixed subset.
        if max_samples is not None:
            dataset = dataset.take(int(max_samples)).cache().repeat()

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Overlap input pipeline with consumers; let TF fill a buffer per host.
        try:
            dataset = dataset.prefetch_to_device(2)
        except Exception:
            dataset = dataset.prefetch(2)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset

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
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000


if __name__ == "__main__":
    import numpy as np

    ds = DroidCoTRldsDataset(
        data_dir="/n/fs/vla-mi/datasets/OXE/",
        language_action_dir="/n/fs/robot-data/vlm-syn/posed_droid",
        batch_size=32,
        shuffle_buffer_size=200,
    )
    ds = iter(ds)
    all_eids = []
    for f in Path("/n/fs/robot-data/vlm-syn/posed_droid").glob("*_language_action.json"):
        eid = f.stem.replace("_language_action", "")
        all_eids.append(eid)

    with open(f"{METADATA_PATH}/droid_language_annotations.json") as f:
        language_annotations = json.load(f)
        all_lang_eids = list(language_annotations.keys())
    total_empty = 0
    for i, batch in enumerate(ds):
        if np.any(batch["prompt"] == b"Do something useful."):
            # count the number of "Do something useful." prompts
            total_empty += np.sum(batch["prompt"] == b"Do something useful.")
            propotion = total_empty / (i + 1) / 32
            print(f"Iter {i}, Total empty prompts: {total_empty}, Propotion: {propotion:.2f}")
        for raw_eid in batch["episode_id"]:
            eid = raw_eid.decode()
            assert eid in all_eids, f"Episode ID {eid} not found in the list of valid episode IDs."
            # assert eid in all_lang_eids, f"Episode ID {eid} not found in the language annotations."
            if eid not in all_lang_eids:
                print(f"Episode ID {eid} not found in the language annotations.")


"""
OXE CoT RLDS dataset.

This dataset mirrors the core functionality found in `oxe_utils/` but
streamlines the implementation and adds CoT features inspired by
`DroidCoTRldsDataset` without any DROID-specific camera/extrinsics logic.

Key features
------------
- RLDS trajectory loading via TFDS builder (with a fallback to builder_from_directory)
- Deterministic train/val splitting using a hash of a stable per-trajectory key
- Per-host sharding (JAX process_count/process_index)
- Language-action lookup (record_id -> serialized tf.string tensor) from TFRecords
- Optional instruction lookup with fallback heuristic phrases
- Action chunking with future-horizon padding at sequence end
- Language-action grouping into future windows (`summation_steps`)
- Optional idle-chunk filtering
- Image decoding (single image key) and optional wrist decoding
- Efficient shuffle/batch/prefetch

Notes
-----
- We intentionally avoid any DROID-specific elements: no episode-ID ↔ path mapping,
  no camera selection, and no calibration/extrinsics handling.
- We keep the output schema minimal and model-friendly:
    {
      "actions": [T, action_chunk_size, D],
      "language_actions": [T, summation_steps],
      "prompt": [T],
      "episode_id": [T]  # actually a per-trajectory record key; retained for debugging
      "observation": {
          "image":  (decoded) [H, W, 3], or a grouped variant if you extend this class
          "wrist_image": optional
      }
    }
  You can add more keys upstream via repack/transforms as desired.
"""


class OxeCoTRldsDataset:
    def __init__(
        self,
        *,
        data_dir: str,
        repo_id: str,
        batch_size: int,
        config: _config.CoTDataConfig,
        language_action_dir: str | None = None,
        # Observation mappings (mirroring oxe_utils)
        image_obs_keys: Mapping[str, str | None] | None = None,  # {new_key: old_key or None}
        depth_obs_keys: Mapping[str, str | None] | None = None,
        proprio_obs_key: str | None = None,
        # Backwards-compat convenience: single-key selection
        image_obs_key: str | None = None,
        wrist_obs_key: str | None = None,
        action_key: str = "action",
        split: str = "train",
        shuffle: bool = True,
        action_chunk_size: int = 16,
        action_horizon: int | None = None,  # alias for action_chunk_size
        window_size: int = 1,
        summation_steps: int | None = None,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 => AUTOTUNE (set after TF import)
        num_parallel_calls: int = -1,  # -1 => AUTOTUNE (set after TF import)
        split_seed: int = 0,
        val_fraction: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        apply_idle_filter: bool | None = None,
        skip_unlabeled: bool = False,
        # Optional filters and padding (akin to oxe_utils)
        max_action: float | None = None,
        max_proprio: float | None = None,
        max_action_dim: int | None = None,
        max_proprio_dim: int | None = None,
        subsample_length: int | None = None,
        post_chunk_transforms: Sequence[Callable] | None = None,
        # Frame-level transforms
        resize_size: Mapping[str, tuple[int, int]] | None = None,
        depth_resize_size: Mapping[str, tuple[int, int]] | None = None,
        image_dropout_prob: float = 0.0,
        image_dropout_keep_key: str | None = None,
        image_augment_kwargs: Mapping[str, dict] | dict | None = None,
        do_imgaug: bool = True,
        # Optional chunk-level filter applied after flatten
        chunk_filter_fn: Callable | None = None,
    ):
        """Construct an iterable OXE CoT RLDS dataset.

        Args:
            data_dir: TFDS data directory.
            repo_id: TFDS dataset name (e.g., "oxe_..." or similar). When not present,
                     a builder_from_directory fallback is attempted at data_dir/repo_id/*.
            batch_size: Global batch size. Will be split per-host automatically.
            config: CoT data configuration (used to pull a few defaults like sum formatting flags).
            language_action_dir: Directory with TFRecords containing per-trajectory language actions.
                                 Filenames are expected to match: tfds_language_actions-*.tfrecord.gz
                                 Each record must contain features: {"record_id": string, "lang_ser": tf.string}
            image_obs_key: Optional key inside RLDS observation to use as the primary image. If None,
                           a reasonable default will be auto-detected.
            wrist_obs_key: Optional key for wrist image (if present in RLDS observation).
            action_key: Name of the action tensor in the raw RLDS trajectory.
            split: "train" or "val". If the TFDS builder doesn't expose a val split, a deterministic
                   hash-based split will be applied using `val_fraction`.
            shuffle: If True, enables shuffling (only in training).
            action_chunk_size: Number of future actions to include per sample.
            summation_steps: Window size for grouping language actions. Defaults to `config.summation_steps`.
            shuffle_buffer_size: TF shuffle buffer size for frame-level shuffling.
            num_parallel_reads: Parallel reads (TF AUTOTUNE if -1).
            num_parallel_calls: Parallel map calls (TF AUTOTUNE if -1).
            split_seed: Seed for deterministic hash-based splitting.
            val_fraction: Fraction for validation when builder lacks an explicit val split. If None, uses
                          `config.val_fraction` if available, else 0.02.
            max_samples: Optional cap on number of flattened samples. If provided, a fixed subset is cached
                         and repeated.
            seed: Global TF seed.
            apply_idle_filter: If True, filters chunks with near-zero initial motion.
                               Defaults to `config.apply_idle_filter`.
            skip_unlabeled: If True, skip trajectories with no language-action annotations.
        """
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Resolve defaults from config
        if summation_steps is None:
            summation_steps = int(getattr(config, "summation_steps", 15))
        if val_fraction is None:
            val_fraction = float(getattr(config, "val_fraction", 0.02) or 0.02)
        if apply_idle_filter is None:
            apply_idle_filter = bool(getattr(config, "apply_idle_filter", True))
        if action_horizon is not None:
            action_chunk_size = int(action_horizon)

        # Normalize defaults for mapping-like params
        if image_obs_keys is None:
            image_obs_keys = {}
        if depth_obs_keys is None:
            depth_obs_keys = {}
        if post_chunk_transforms is None:
            post_chunk_transforms = ()
        if resize_size is None:
            resize_size = {}
        if depth_resize_size is None:
            depth_resize_size = {}
        if image_augment_kwargs is None:
            image_augment_kwargs = {}

        # Setup TF determinism and devices
        tf.random.set_seed(seed)
        tf.config.set_visible_devices([], "GPU")
        from contextlib import suppress

        with suppress(Exception):
            tf.config.set_visible_devices([], "TPU")

        # Enable AUTOTUNE sentinels
        if num_parallel_reads == -1:
            num_parallel_reads = tf.data.AUTOTUNE
        if num_parallel_calls == -1:
            num_parallel_calls = tf.data.AUTOTUNE

        # Helper: create a TFDS builder for a dataset name
        def _make_builder_for(name: str):
            def _mk():
                return tfds.builder(name, data_dir=data_dir)

            def _mk_dir():
                import os as _os

                ds_dir = _os.path.join(data_dir, name)
                subdirs = [d for d in _os.listdir(ds_dir) if _os.path.isdir(_os.path.join(ds_dir, d))]
                if len(subdirs) != 1:
                    raise FileNotFoundError(f"Expected one subdir under {ds_dir}, found: {subdirs}")
                return tfds.builder_from_directory(_os.path.join(ds_dir, subdirs[0]))

            return _maybe(_mk, _mk_dir)

        want_val = split == "val"

        # Determine whether to load a named mixture (concatenate) or a single dataset
        names_and_weights = [(repo_id, 1.0)]
        if repo_id in OXE_NAMED_MIXES:
            names_and_weights = OXE_NAMED_MIXES[repo_id]

        datasets_to_concat = []

        # Common TF options
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True

        # Build per-dataset pipelines up to trajectory-level grouping
        for ds_name, _w in names_and_weights:
            builder = _make_builder_for(ds_name)
            if (not want_val) and (max_samples is None):
                ds = dl.DLataset.from_rlds(builder, split="train", shuffle=True, num_parallel_reads=num_parallel_reads)
            else:
                ds = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=num_parallel_reads)
            ds = ds.shard(jax.process_count(), jax.process_index())
            ds = ds.with_options(opts)

            # Per-dataset default mappings from OXE configs if not provided explicitly
            image_keys_i = (
                image_obs_keys if image_obs_keys else OXE_DATASET_CONFIGS.get(ds_name, {}).get("image_obs_keys", {})
            )
            depth_keys_i = (
                depth_obs_keys if depth_obs_keys else OXE_DATASET_CONFIGS.get(ds_name, {}).get("depth_obs_keys", {})
            )
            proprio_key_i = proprio_obs_key
            # Standardization transform (if available) – run first on trajectories
            standardize_fn = OXE_STANDARDIZATION_TRANSFORMS.get(ds_name)

            # Helper: record-id and split filter, parameterized by dataset name for fallback uniqueness
            def _record_id_from_traj_local(traj):
                try:
                    return traj["traj_metadata"]["episode_metadata"]["file_path"][0]
                except Exception:
                    try:
                        return traj["traj_metadata"]["record_metadata"]["file_path"][0]
                    except Exception:
                        base = tf.constant(ds_name, tf.string)
                        idx = tf.as_string(traj.get("_traj_index", tf.constant([0], tf.int64))[0])
                        return tf.strings.join([base, idx], separator=":")

            def _split_filter_local(traj):
                rid = _record_id_from_traj_local(traj)
                salt = tf.strings.as_string(split_seed)
                key = tf.strings.join([salt, rid])
                bucket = tf.strings.to_hash_bucket_fast(key, 1000)
                thr = tf.cast(int(val_fraction * 1000), tf.int64)
                is_val = bucket < thr
                return is_val if want_val else tf.logical_not(is_val)

            ds = ds.filter(_split_filter_local)
            if skip_unlabeled and lang_table is not None:

                def _has_lang_local(traj):
                    rid = _record_id_from_traj_local(traj)
                    return tf.not_equal(lang_table.lookup(rid), default_lang_value)

                ds = ds.filter(_has_lang_local)

            # Build restructure specialized for this dataset
            def _restructure_local(traj):
                # Apply dataset-specific standardization first
                if standardize_fn is not None:
                    traj = standardize_fn(traj)

                # Extract actions
                if action_key in traj:
                    actions = tf.cast(traj[action_key], tf.float32)
                elif "action" in traj:
                    actions = tf.cast(traj["action"], tf.float32)
                else:
                    ad = traj.get("action_dict", {})
                    actions = tf.cast(ad.get("cartesian_position", traj.get("actions", None)), tf.float32)
                traj_len = tf.shape(actions)[0]

                # Optional action padding
                if max_action_dim is not None:
                    cur = tf.shape(actions)[-1]
                    pad = tf.maximum(0, int(max_action_dim) - cur)
                    actions = tf.pad(actions, [[0, 0], [0, pad]])

                rid = _record_id_from_traj_local(traj)

                # Language actions
                if lang_table is not None:
                    lang_bytes = lang_table.lookup(rid)
                    lang_tensor = tf.cond(
                        tf.greater(tf.strings.length(lang_bytes), 0),
                        lambda: tf.io.parse_tensor(lang_bytes, out_type=tf.string),
                        lambda: tf.fill([traj_len], tf.constant("", tf.string)),
                    )
                else:
                    lang_tensor = tf.fill([traj_len], tf.constant("", tf.string))
                lang_tensor = lang_tensor[:traj_len]

                # Instruction
                if instr_table is not None:
                    instr_bytes = instr_table.lookup(rid)

                    def _sample_from_table():
                        arr = tf.io.parse_tensor(instr_bytes, out_type=tf.string)
                        return tf.random.shuffle(arr, seed=seed)[0]

                    instruction = tf.cond(
                        tf.greater(tf.strings.length(instr_bytes), 0),
                        _sample_from_table,
                        lambda: fallback_instructions[
                            tf.random.uniform((), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32)
                        ],
                    )
                else:
                    instruction = fallback_instructions[
                        tf.random.uniform((), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32)
                    ]
                instruction_vec = tf.fill([traj_len], instruction)

                # Observation mapping
                obs = traj.get("observation", traj.get("obs", {}))
                new_obs = {}
                if image_keys_i:
                    for new, old in image_keys_i.items():
                        new_key = f"image_{new}"
                        if old is None:
                            new_obs[new_key] = tf.repeat("", traj_len)
                        else:
                            new_obs[new_key] = tf.convert_to_tensor(obs[old])
                if depth_keys_i:
                    for new, old in depth_keys_i.items():
                        new_key = f"depth_{new}"
                        if old is None:
                            new_obs[new_key] = tf.repeat("", traj_len)
                        else:
                            new_obs[new_key] = tf.convert_to_tensor(obs[old])
                if proprio_key_i is not None and proprio_key_i in obs:
                    new_obs["proprio"] = tf.cast(obs[proprio_key_i], tf.float32)

                out = {
                    "actions": actions,
                    "language_actions": lang_tensor,
                    "prompt": instruction_vec,
                    "episode_id": tf.fill([traj_len], rid),
                    "observation": new_obs,
                    "dataset_name": tf.repeat(tf.constant(ds_name, tf.string), traj_len),
                    "traj_index": tf.repeat(traj.get("_traj_index", tf.constant([0]))[0], traj_len),
                }
                if wrist_obs_key and wrist_obs_key in obs:
                    out["observation"]["wrist_image"] = obs[wrist_obs_key]
                return out

            ds = ds.traj_map(_restructure_local, num_parallel_calls)

            # Optional filters
            if max_action is not None:
                ds = ds.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["actions"]) <= max_action))
            if max_proprio is not None:

                def _has_ok_prop(x):
                    prop = x.get("observation", {}).get("proprio", None)
                    return tf.constant(True) if prop is None else tf.math.reduce_all(tf.math.abs(prop) <= max_proprio)

                ds = ds.filter(_has_ok_prop)

            # Chunking
            def _chunk(traj):
                traj_len = tf.shape(traj["actions"])[0]
                idx = tf.broadcast_to(
                    tf.range(action_chunk_size)[None], [traj_len, action_chunk_size]
                ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_chunk_size])
                idx = tf.minimum(idx, traj_len - 1)
                traj["actions"] = tf.gather(traj["actions"], idx)
                return traj

            ds = ds.traj_map(_chunk, num_parallel_calls)

            # Subsample
            if subsample_length is not None:

                def _subsample_local(traj):
                    t_len = tf.shape(traj["actions"])[0]
                    l_len = tf.minimum(tf.cast(subsample_length, tf.int32), t_len)
                    for k in list(traj.keys()):
                        v = traj[k]
                        if isinstance(v, dict):
                            for sk in list(v.keys()):
                                vv = v[sk]
                                if tf.rank(vv) >= 1 and tf.shape(vv)[0] == t_len:
                                    v[sk] = vv[:l_len]
                        elif tf.rank(v) >= 1 and tf.shape(v)[0] == t_len:
                            traj[k] = v[:l_len]
                    return traj

                ds = ds.traj_map(_subsample_local, num_parallel_calls)

            # Window observations
            if int(window_size) > 1:
                ws = int(window_size)

                def _win(traj):
                    t_len = tf.shape(traj["language_actions"])[0]
                    idx = tf.broadcast_to(tf.range(ws)[None], [t_len, ws]) + tf.broadcast_to(
                        tf.range(t_len)[:, None], [t_len, ws]
                    )
                    idx = tf.minimum(idx, t_len - 1)
                    for k, v in list(traj["observation"].items()):
                        if tf.rank(v) >= 1 and tf.shape(v)[0] == t_len:
                            traj["observation"][k] = tf.gather(v, idx)
                    return traj

                ds = ds.traj_map(_win, num_parallel_calls)

            # Group language actions windows
            def _group(traj):
                traj_len = tf.shape(traj["language_actions"])[0]
                idx = tf.broadcast_to(tf.range(summation_steps)[None], [traj_len, summation_steps]) + tf.broadcast_to(
                    tf.range(traj_len)[:, None], [traj_len, summation_steps]
                )
                idx = tf.minimum(idx, traj_len - 1)
                traj["language_actions"] = tf.gather(traj["language_actions"], idx)
                return traj

            ds = ds.traj_map(_group, num_parallel_calls)

            # Post-chunk transforms
            for fn in post_chunk_transforms:
                ds = ds.traj_map(fn, num_parallel_calls)

            datasets_to_concat.append(ds)

        # Concatenate datasets according to descending weights order
        if len(datasets_to_concat) == 0:
            raise ValueError("No datasets constructed for OXE CoT dataset")
        if len(datasets_to_concat) == 1:
            dataset = datasets_to_concat[0]
        else:
            # Sort by weights (descending)
            order = sorted(range(len(names_and_weights)), key=lambda i: names_and_weights[i][1], reverse=True)
            dataset = datasets_to_concat[order[0]]
            for i in order[1:]:
                dataset = dataset.concatenate(datasets_to_concat[i])

        # ---- Language actions lookup (record_id -> serialized tf.string tensor [T(+1)]) ----
        lang_table = None
        default_lang_value = tf.constant(b"", dtype=tf.string)
        if language_action_dir is not None:
            features = {
                "record_id": tf.io.FixedLenFeature([], tf.string),
                "lang_ser": tf.io.FixedLenFeature([], tf.string),
            }

            def _parse(record):
                ex = tf.io.parse_single_example(record, features)
                # Note: we store serialized tf.string tensor to avoid ragged/shape issues in lookup
                return ex["record_id"], ex["lang_ser"]

            files = tf.io.gfile.glob(f"{language_action_dir}/tfds_language_actions-*.tfrecord.gz")
            if not files:
                logging.warning("No language-action TFRecords found at %s", language_action_dir)
            ds_lang = (
                tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
                .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
            keys_py, vals_py = [], []
            for k, v in ds_lang:
                keys_py.append(k.numpy())
                vals_py.append(v.numpy())
            if keys_py:
                lang_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(tf.constant(keys_py), tf.constant(vals_py)),
                    default_value=default_lang_value,
                )

        # ---- Instruction lookup (optional): record_id -> serialized [K] (strings) ----
        # If an instructions.json exists alongside language actions, load it. Otherwise, sample fallback.
        instr_table = None
        _instr_default = tf.constant(b"", dtype=tf.string)
        if language_action_dir is not None:
            instr_json = os.path.join(language_action_dir, "instructions.json")
            if tf.io.gfile.exists(instr_json):
                try:
                    with tf.io.gfile.GFile(instr_json, "r") as f:
                        instr_index = json.load(f)
                    _keys = list(instr_index.keys())
                    _vals = []
                    for rid in _keys:
                        arr = instr_index.get(rid, [])
                        arr = [s for s in arr if isinstance(s, str) and len(s) > 0]
                        if not arr:
                            _vals.append(b"")
                        else:
                            _vals.append(tf.io.serialize_tensor(tf.constant(arr, dtype=tf.string)).numpy())
                    instr_table = tf.lookup.StaticHashTable(
                        tf.lookup.KeyValueTensorInitializer(
                            tf.constant(_keys, dtype=tf.string), tf.constant(_vals, dtype=tf.string)
                        ),
                        default_value=_instr_default,
                    )
                except Exception as e:
                    logging.warning("Failed loading instructions.json: %s", e)

        fallback_instructions = tf.constant(
            [
                "Do something useful.",
                "Complete the task.",
                "Perform the task.",
                "Carry out the objective.",
                "Proceed with the task.",
                "Accomplish the goal.",
                "Handle the task at hand.",
                "Continue the operation.",
                "Fulfill the task.",
                "Take meaningful steps.",
                "Demonstrate useful behavior.",
            ],
            dtype=tf.string,
        )

        # ---- Helpers to extract IDs and keys ----
        def _record_id_from_traj(traj):
            # Prefer a file_path or any present identifier in RLDS metadata
            try:
                return traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            except Exception:
                try:
                    return traj["traj_metadata"]["record_metadata"]["file_path"][0]
                except Exception:
                    # deterministic fallback: combine data_dir hint with index if present
                    base = tf.constant(repo_id, tf.string)
                    idx = tf.as_string(traj.get("_traj_index", tf.constant([0], tf.int64))[0])
                    return tf.strings.join([base, idx], separator=":")

        def _has_lang(traj):
            if lang_table is None:
                return tf.constant(True, tf.bool)  # nothing to check
            rid = _record_id_from_traj(traj)
            val = lang_table.lookup(rid)
            return tf.not_equal(val, default_lang_value)

        # ---- Split filter: deterministic per-trajectory hash using record_id ----
        def _split_filter(traj):
            rid = _record_id_from_traj(traj)
            salt = tf.strings.as_string(split_seed)
            key = tf.strings.join([salt, rid])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if want_val else tf.logical_not(is_val)

        dataset = dataset.filter(_split_filter)

        if skip_unlabeled and lang_table is not None:
            dataset = dataset.filter(_has_lang)

        # ---- Restructure: standardize without dataset-specific assumptions ----
        def _pad_last_dim_2d(x: tf.Tensor, target: int) -> tf.Tensor:
            x = tf.convert_to_tensor(x)
            cur = tf.shape(x)[-1]
            pad = tf.maximum(0, target - cur)
            return tf.pad(x, [[0, 0], [0, pad]])

        def restructure(traj):
            # Actions (float32)
            if action_key in traj:
                actions = tf.cast(traj[action_key], tf.float32)
            elif "action" in traj:
                actions = tf.cast(traj["action"], tf.float32)
            else:
                # Last resort: attempt action_dict with cartesian + gripper
                ad = traj.get("action_dict", {})
                try:
                    actions = tf.concat((ad["cartesian_position"], ad["gripper_position"]), axis=-1)
                    actions = tf.cast(actions, tf.float32)
                except Exception:
                    raise KeyError("Could not locate action tensor in trajectory.")

            traj_len = tf.shape(actions)[0]
            # Optional action dim padding
            if max_action_dim is not None:
                actions = _pad_last_dim_2d(actions, int(max_action_dim))

            rid = _record_id_from_traj(traj)

            # Language actions per-step (serialized tf.string tensor)
            if lang_table is not None:
                lang_bytes = lang_table.lookup(rid)
                lang_tensor = tf.cond(
                    tf.greater(tf.strings.length(lang_bytes), 0),
                    lambda: tf.io.parse_tensor(lang_bytes, out_type=tf.string),
                    lambda: tf.fill([traj_len], tf.constant("", tf.string)),
                )
            else:
                lang_tensor = tf.fill([traj_len], tf.constant("", tf.string))
            lang_tensor = lang_tensor[:traj_len]

            # Instruction sampling from optional table; fallback otherwise
            if instr_table is not None:
                instr_bytes = instr_table.lookup(rid)

                def _sample_from_table():
                    arr = tf.io.parse_tensor(instr_bytes, out_type=tf.string)
                    return tf.random.shuffle(arr, seed=seed)[0]

                instruction = tf.cond(
                    tf.greater(tf.strings.length(instr_bytes), 0),
                    _sample_from_table,
                    lambda: fallback_instructions[
                        tf.random.uniform((), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32)
                    ],
                )
            else:
                instruction = fallback_instructions[
                    tf.random.uniform((), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32)
                ]

            instruction_vec = tf.fill([traj_len], instruction)

            # Observation image(s) and depth(s)
            obs = traj.get("observation", traj.get("obs", {}))
            new_obs: dict[str, tf.Tensor] = {}

            if image_obs_keys:
                for new, old in image_obs_keys.items():
                    if old is None:
                        new_obs[f"image_{new}"] = tf.repeat("", traj_len)
                    else:
                        new_obs[f"image_{new}"] = tf.convert_to_tensor(obs[old])
            else:
                preferred_image_keys = (
                    image_obs_key,
                    "image_primary",
                    "image",
                    "exterior_image_1_left",
                    "rgb",
                    "cam0",
                )

                def _select_image(_obs: dict):
                    for k in preferred_image_keys:
                        if k and (k in _obs):
                            return _obs[k]
                    for k, v in _obs.items():
                        if v.dtype == tf.string:
                            return v
                    raise KeyError("No image-like key found in observation.")

                new_obs["image_primary"] = _select_image(obs)

            if depth_obs_keys:
                for new, old in depth_obs_keys.items():
                    if old is None:
                        new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
                    else:
                        new_obs[f"depth_{new}"] = tf.convert_to_tensor(obs[old])

            # Optional proprio
            if proprio_obs_key is not None and proprio_obs_key in obs:
                proprio = tf.cast(obs[proprio_obs_key], tf.float32)
                if max_proprio_dim is not None:
                    proprio = _pad_last_dim_2d(proprio, int(max_proprio_dim))
                new_obs["proprio"] = proprio

            out = {
                "actions": actions,
                "language_actions": lang_tensor,
                "prompt": instruction_vec,
                "episode_id": tf.fill([traj_len], rid),  # Keep for debugging/traceability
                "observation": new_obs,
                "dataset_name": tf.repeat(tf.constant(repo_id, tf.string), traj_len),
                "traj_index": tf.repeat(traj.get("_traj_index", tf.constant([0]))[0], traj_len),
            }
            if wrist_obs_key and wrist_obs_key in obs:
                out["observation"]["wrist_image"] = obs[wrist_obs_key]
            return out

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        # Filters similar to oxe_utils
        if max_action is not None:
            dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["actions"]) <= max_action))
        if max_proprio is not None:

            def _has_ok_proprio(x):
                prop = x.get("observation", {}).get("proprio", None)
                if prop is None:
                    return tf.constant(True)
                return tf.math.reduce_all(tf.math.abs(prop) <= max_proprio)

            dataset = dataset.filter(_has_ok_proprio)

        # ---- Chunk actions (pad with last step at sequence end) ----
        def chunk_actions(traj):
            traj_len = tf.shape(traj["actions"])[0]
            idx = tf.broadcast_to(tf.range(action_chunk_size)[None], [traj_len, action_chunk_size]) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, action_chunk_size]
            )
            idx = tf.minimum(idx, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], idx)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Optional subsample after chunking
        if subsample_length is not None:

            def _subsample(traj):
                t_len = tf.shape(traj["actions"])[0]
                l_len = tf.minimum(tf.cast(subsample_length, tf.int32), t_len)
                for k in list(traj.keys()):
                    v = traj[k]
                    if isinstance(v, dict):
                        for sk in list(v.keys()):
                            vv = v[sk]
                            if tf.rank(vv) >= 1 and tf.shape(vv)[0] == t_len:
                                v[sk] = vv[:l_len]
                    elif tf.rank(v) >= 1 and tf.shape(v)[0] == t_len:
                        traj[k] = v[:l_len]
                return traj

            dataset = dataset.traj_map(_subsample, num_parallel_calls)

        # Window observations if requested
        if int(window_size) > 1:
            ws = int(window_size)

            def _window_obs(traj):
                t_len = tf.shape(traj["language_actions"])[0]
                idx = tf.broadcast_to(tf.range(ws)[None], [t_len, ws]) + tf.broadcast_to(
                    tf.range(t_len)[:, None], [t_len, ws]
                )
                idx = tf.minimum(idx, t_len - 1)
                for k, v in list(traj["observation"].items()):
                    if tf.rank(v) >= 1 and tf.shape(v)[0] == t_len:
                        traj["observation"][k] = tf.gather(v, idx)
                return traj

            dataset = dataset.traj_map(_window_obs, num_parallel_calls)

        # ---- Group language actions: [T] -> [T, summation_steps] future window ----
        def group_language_actions(traj):
            traj_len = tf.shape(traj["language_actions"])[0]
            idx = tf.broadcast_to(tf.range(summation_steps)[None], [traj_len, summation_steps]) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, summation_steps]
            )
            idx = tf.minimum(idx, traj_len - 1)
            traj["language_actions"] = tf.gather(traj["language_actions"], idx)
            return traj

        dataset = dataset.traj_map(group_language_actions, num_parallel_calls)

        # Post-chunk transforms (trajectory level)
        for fn in post_chunk_transforms:
            dataset = dataset.traj_map(fn, num_parallel_calls)

        # ---- Flatten to frame-level (each element is one action-chunk) ----
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # ---- Optional idle filtering ----
        if apply_idle_filter:

            def _filter_idle(frame):
                # Filter if first half of the action chunk shows no movement beyond a small threshold
                first = frame["actions"][: action_chunk_size // 2]
                ref = frame["actions"][:1]
                return tf.reduce_any(tf.abs(first - ref) > 1e-3)

            dataset = dataset.filter(_filter_idle)

        # Optional chunk-level filter
        if chunk_filter_fn is not None:
            dataset = dataset.filter(chunk_filter_fn)

        # Decode/resize/augment/dropout at frame level
        is_train = split == "train"

        def _decode_one(img_bytes):
            return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)

        def _resize_if_needed(img: tf.Tensor, size: tuple[int, int] | None):
            if size is None:
                return img
            x = tf.image.resize(tf.cast(img, tf.float32), size, method=tf.image.ResizeMethod.BILINEAR)
            return tf.cast(tf.clip_by_value(tf.round(x), 0.0, 255.0), tf.uint8)

        def _maybe_augment(img: tf.Tensor) -> tf.Tensor:
            if not (is_train and do_imgaug):
                return img
            # Chain simple deterministic ops for linter clarity
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            return img

        def _decode_and_resize_frame(frame: dict) -> dict:
            obs = frame["observation"]

            def _decode_value(key: str, val: tf.Tensor, is_depth: bool) -> tf.Tensor:
                if val.dtype != tf.string:
                    return val

                def _decode_vec(vec):
                    return tf.map_fn(_decode_one, vec, fn_output_signature=tf.uint8)

                decoded = tf.cond(
                    tf.rank(val) == 0,
                    lambda: _decode_one(val),
                    lambda: _decode_vec(val),
                )
                size_map = depth_resize_size if is_depth else resize_size
                size = None
                name = key.replace("image_", "").replace("depth_", "")
                if isinstance(size_map, dict) and name in size_map:
                    size = size_map[name]
                if tf.rank(decoded) == 3:  # [H,W,C]
                    decoded = _resize_if_needed(decoded, size)
                    decoded = _maybe_augment(decoded)
                elif tf.rank(decoded) == 4:  # [W,H,W,C]

                    def _rz(img):
                        img = _resize_if_needed(img, size)
                        img = _maybe_augment(img)
                        return img

                    decoded = tf.map_fn(_rz, decoded, fn_output_signature=tf.uint8)
                return decoded

            for k, v in list(obs.items()):
                if k.startswith("image_"):
                    obs[k] = _decode_value(k, v, is_depth=False)
                elif k.startswith("depth_"):
                    obs[k] = _decode_value(k, v, is_depth=True)
                elif k == "wrist_image":
                    obs[k] = _decode_value(k, v, is_depth=False)

            # Image dropout per key (after decode) — keep at least one
            if is_train and image_dropout_prob > 0.0:
                image_keys = [k for k in obs if k.startswith("image_")]
                if image_keys:
                    rng = tf.random.uniform([len(image_keys)])
                    keep_mask = rng > image_dropout_prob
                    if image_dropout_keep_key is not None:
                        keep_any = tf.reduce_any(keep_mask)

                        def _ensure_keep():
                            idx = tf.argmax(tf.cast(tf.equal(image_keys, f"image_{image_dropout_keep_key}"), tf.int32))
                            return tf.tensor_scatter_nd_update(keep_mask, [[idx]], [True])

                        keep_mask = tf.cond(keep_any, lambda: keep_mask, _ensure_keep)

                    for i, k in enumerate(image_keys):
                        zeros = tf.zeros_like(obs[k])
                        obs[k] = tf.cond(keep_mask[i], lambda: obs[k], lambda: zeros)

            return frame

        dataset = dataset.frame_map(_decode_and_resize_frame, num_parallel_calls)

        # ---- Shuffle/batch/prefetch ----
        if (not want_val) and shuffle and (max_samples is None):
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

        if max_samples is not None:
            dataset = dataset.take(int(max_samples)).cache().repeat()

        # Per-host batching
        local_bsz = max(1, batch_size // jax.process_count())
        dataset = dataset.batch(local_bsz, drop_remainder=True)
        try:
            dataset = dataset.prefetch_to_device(2)
        except Exception:
            dataset = dataset.prefetch(2)
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset

    # ---- Iterable protocol ----
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
        # Unknown a priori without scanning; return a large sentinel as in DROID impl.
        return 20_000_000


if __name__ == "__main__":
    # Minimal smoke test stub (assumes a local TFDS directory structure)
    test_repo = os.environ.get("OXE_REPO_ID", "")
    test_data_dir = os.environ.get("OXE_DATA_DIR", "")
    test_lang_dir = os.environ.get("OXE_LANG_DIR", "")
    if test_repo and test_data_dir:
        ds = OxeCoTRldsDataset(
            data_dir=test_data_dir,
            repo_id=test_repo,
            language_action_dir=(test_lang_dir or None),
            batch_size=8,
            split="train",
            config=_config.CoTDataConfig(),
        )
        it = iter(ds)
        for i in range(2):
            batch = next(it)
            print("batch keys:", batch.keys())
            print("image shape:", batch["observation"]["image"].shape)
            print("actions shape:", batch["actions"].shape)
            print("language_actions shape:", batch["language_actions"].shape)
            print("prompt example:", batch["prompt"][0])
