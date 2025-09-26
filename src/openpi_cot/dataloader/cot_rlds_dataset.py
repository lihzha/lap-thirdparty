from collections.abc import Callable
import contextlib
from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.helpers import StateEncoding
from openpi_cot.dataloader.helpers import convert_action_encoding
from openpi_cot.dataloader.helpers import convert_state_encoding
from openpi_cot.dataloader.helpers import euler_xyz_to_rot
from openpi_cot.dataloader.helpers import extract_episode_path_from_file_path
from openpi_cot.dataloader.helpers import project_in_bounds
from openpi_cot.dataloader.oxe_utils.data_utils import NormalizationType
from openpi_cot.dataloader.oxe_utils.data_utils import allocate_threads
from openpi_cot.dataloader.oxe_utils.data_utils import pprint_data_mixture
from openpi_cot.dataloader.oxe_utils.materialize import get_oxe_dataset_kwargs_and_weights
from openpi_cot.dataloader.oxe_utils.mixtures import OXE_NAMED_MIXTURES
from openpi_cot.shared.adapters.normalize_adapter import check_dataset_statistics
from openpi_cot.shared.adapters.normalize_adapter import get_dataset_statistics
from openpi_cot.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


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
        return dataset.repeat().shuffle(shuffle_buffer_size, seed=seed)
    if max_samples is not None:
        return dataset.take(int(max_samples)).cache().repeat()
    return dataset


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    use_wrist_image: bool,
    resize_to: tuple[int, int] | None = (224, 224),
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors.
    Preserves aspect ratio, pads symmetrically, and returns the original dtype semantics
    (uint8 clamped 0-255, float32 clamped to [-1, 1]).
    """

    def _tf_resize_with_pad(image: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        # Compute resized dimensions preserving aspect ratio
        in_h = tf.shape(image)[0]
        in_w = tf.shape(image)[1]
        orig_dtype = image.dtype

        h_f = tf.cast(in_h, tf.float32)
        w_f = tf.cast(in_w, tf.float32)
        th_f = tf.cast(target_h, tf.float32)
        tw_f = tf.cast(target_w, tf.float32)

        ratio = tf.maximum(w_f / tw_f, h_f / th_f)
        resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
        resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)

        # Resize in float32
        img_f32 = tf.cast(image, tf.float32)
        resized_f32 = tf.image.resize(img_f32, [resized_h, resized_w], method=tf.image.ResizeMethod.BILINEAR)

        # Dtype-specific postprocess (python conditional on static dtype)
        if orig_dtype == tf.uint8:
            resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
            const_val = tf.constant(0, dtype=resized.dtype)
        else:
            resized = tf.clip_by_value(resized_f32, -1.0, 1.0)
            const_val = tf.constant(-1.0, dtype=resized.dtype)

        # Compute symmetric padding
        pad_h_total = target_h - resized_h
        pad_w_total = target_w - resized_w
        pad_h0 = pad_h_total // 2
        pad_h1 = pad_h_total - pad_h0
        pad_w0 = pad_w_total // 2
        pad_w1 = pad_w_total - pad_w0

        padded = tf.pad(resized, [[pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], constant_values=const_val)
        return padded

    def _decode_single(img_bytes):
        # If already numeric, cast to uint8 and return
        if img_bytes.dtype != tf.string:
            img = tf.cast(img_bytes, tf.uint8)
        else:
            # Guard against empty placeholders (e.g., padding "")
            has_data = tf.greater(tf.strings.length(img_bytes), 0)
            img = tf.cond(
                has_data,
                lambda: tf.io.decode_image(
                    img_bytes,
                    channels=3,
                    expand_animations=False,
                    dtype=tf.uint8,
                ),
                lambda: tf.zeros([1, 1, 3], dtype=tf.uint8),
            )
        # Optional resize-with-pad to ensure batching shape compatibility
        if resize_to is not None:
            h, w = resize_to
            img = _tf_resize_with_pad(img, h, w)
        return img

    def _decode_frame(traj: dict) -> dict:
        traj["observation"][primary_key] = _decode_single(traj["observation"][primary_key])
        if use_wrist_image and wrist_key is not None:
            traj["observation"][wrist_key] = _decode_single(traj["observation"][wrist_key])
        # traj["observation"][primary_key] = tf.map_fn(
        #     _decode_single,
        #     traj["observation"][primary_key],
        #     fn_output_signature=tf.uint8,
        # )
        # traj["observation"][wrist_key] = tf.map_fn(
        #     _decode_single,
        #     traj["observation"][wrist_key],
        #     fn_output_signature=tf.uint8,
        # )
        return traj

    return _decode_frame


def compute_window_indices(sequence_length: tf.Tensor, window_size: int) -> tf.Tensor:
    """Return [T, window] indices for gathering sliding windows with end padding.

    Builds indices for each timestep t to gather the next `window_size` steps,
    clamped to the final index so the tail windows repeat the last element.
    """
    # Shape: [T, window]
    base = tf.broadcast_to(tf.range(window_size)[None], [sequence_length, window_size])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size])
    indices = base + offsets
    # Cap to the last valid index to repeat the final element
    return tf.minimum(indices, sequence_length - 1)


# Helper: try cardinality; fall back to counting if UNKNOWN/INFINITE
def dataset_size(ds: tf.data.Dataset) -> int:
    c = ds.cardinality().numpy()  # returns int64 or negative sentinel
    if c >= 0:
        return int(c)
    # Count explicitly (works after .filter/.flat_map, etc.)
    return int(ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())


@dataclass(frozen=True)
class DroidDatasetSpec:
    lang_action_tfrecord_pattern: str = "tfds_language_actions-*.tfrecord.gz"
    lang_action_dir_name: str = "droid-lang-actions"
    lang_action_dir_name_base: str = "droid-base-lang-actions"
    metadata_path_name: str = "metadata"
    episode_id_to_path_file: str = "episode_id_to_path.json"
    cam2base_extrinsics_file: str = "cam2base_extrinsics.json"
    camera_serials_file: str = "camera_serials.json"
    intrinsics_file: str = "intrinsics.json"
    droid_instructions_file: str = "droid_instructions.json"
    droid_language_annotations_file: str = "droid_language_annotations.json"
    keep_ranges_file: str = "keep_ranges_1_0_1.json"
    images_list: tuple[str, str] = ("exterior_image_1_left", "exterior_image_2_left")
    default_lang_value: tf.Tensor = field(
        default_factory=lambda: tf.io.serialize_tensor(tf.constant([], dtype=tf.string))
    )
    default_ep_value: tf.Tensor = field(default_factory=lambda: tf.constant("", dtype=tf.string))
    default_intr_ser: tf.Tensor = field(default_factory=lambda: tf.io.serialize_tensor(tf.zeros([4], tf.float32)))
    default_extr_ser: tf.Tensor = field(
        default_factory=lambda: tf.io.serialize_tensor(tf.reshape(tf.eye(4, dtype=tf.float32), [-1]))
    )
    default_state_encoding: StateEncoding = StateEncoding.POS_EULER
    default_action_encoding: ActionEncoding = ActionEncoding.ABS_EEF_POS


class _SingleCoTRldsDatasetRaw:
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
        config: "CoTDataConfig",
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
        if ds_name == "fmb":
            ds_name = "fmb:0.0.1"
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

    # Template hook: subclasses must provide a stable per-trajectory anchor used for splitting
    def get_split_anchor(self, traj):
        """Return a tf.string key used to deterministically split train/val."""
        raise NotImplementedError

    # Shared split implementation using the anchor returned by get_split_anchor
    def split_val(self, split_seed):
        def _split_filter(traj):
            salt = tf.strings.as_string(split_seed)
            anchor = self.get_split_anchor(traj)
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)


class _DroidCoTRldsDatasetRaw(_SingleCoTRldsDatasetRaw):
    spec: ClassVar[DroidDatasetSpec] = DroidDatasetSpec()

    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lang_action_table(self, language_action_dir):
        # ---------------------------------------------------------------------
        # 1. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        print_memory_usage("Before building lang_action_table")
        features = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(record):
            ex = tf.io.parse_single_example(record, features)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        files = tf.io.gfile.glob(f"{language_action_dir}/{self.spec.lang_action_tfrecord_pattern}")
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
            default_value=self.spec.default_lang_value,
        )
        print_memory_usage("After building lang_table")
        return lang_table

    def build_lookup_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 2. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.episode_id_to_path_file}", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.spec.default_ep_value,
        )
        print_memory_usage("After building ep_table")
        return ep_table

    def build_cam_tables(self, metadata_path, need_calib: bool):
        # ---------------------------------------------------------------------
        # 3. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.cam2base_extrinsics_file}", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.camera_serials_file}", "r") as fp:
            camera_serials = json.load(fp)
        if need_calib:
            with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.intrinsics_file}", "r") as fp:
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
            if eid not in intrinsics_json:
                continue

            calib_camera_name = serial_to_name[camera_serial]
            if calib_camera_name == "ext1_cam_serial":
                calib_image_name = "exterior_image_1_left"
            elif calib_camera_name == "ext2_cam_serial":
                calib_image_name = "exterior_image_2_left"
            else:
                raise ValueError(f"Unknown camera name: {calib_camera_name}")

            calib_image_idx = self.spec.images_list.index(calib_image_name)
            eid_to_cam_dict[eid] = calib_image_idx

            if need_calib:
                # Camera intrinsics as [fx, fy, cx, cy]
                fx, cx, fy, cy = intrinsics_json[eid][camera_serial]["cameraMatrix"]
                eid_to_intr_vec[eid] = [fx, fy, cx, cy]

                # Camera extrinsics 4x4 from [tx,ty,tz,rx,ry,rz]
                tx, ty, tz, rx, ry, rz = extr[camera_serial]
                rot_matrix = euler_xyz_to_rot(rx, ry, rz)
                transform_matrix = np.eye(4, dtype=np.float32)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [tx, ty, tz]
                eid_to_extr_mat[eid] = transform_matrix.reshape(-1)

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
                default_value=self.spec.default_intr_ser,
            )
            extr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, extr_vals),
                default_value=self.spec.default_extr_ser,
            )
            print_memory_usage("After building intr_table and extr_table")
        return cam_table, intr_table, extr_table

    def build_instr_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 6. Language-instruction table (merged; episode_id → serialized [K])
        # ---------------------------------------------------------------------
        instr_cache_path = f"{metadata_path}/{self.spec.droid_instructions_file}"
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
                    # Use truly empty bytes for missing-instruction episodes so we can detect and fallback later
                    _instr_vals_ser.append(b"")
                else:
                    _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())
        else:
            with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.droid_language_annotations_file}", "r") as fp:
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
            # Store per-trajectory ranges, not per-step flags
            with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.keep_ranges_file}", "r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            if self.use_per_traj_filter:
                ep_keys = []
                ranges_ser = []
                for episode_key, ranges in filter_dict.items():
                    # Ensure serialized ranges are always 2D [M, 2]
                    # Some entries may be a single [start, end] list → shape (2,)
                    arr = np.array(ranges, dtype=np.int32).reshape(-1, 2)
                    ep_keys.append(episode_key)
                    ranges_ser.append(tf.io.serialize_tensor(tf.constant(arr)).numpy())

                keys = tf.constant(ep_keys, dtype=tf.string)
                vals = tf.constant(ranges_ser, dtype=tf.string)
                filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys, vals),
                    default_value=tf.constant(b"", dtype=tf.string),
                )
                print_memory_usage("After building filter_table (per-trajectory)")

            else:
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
                print_memory_usage("After building filter_table (per-step)")
        logging.info("Filter hash table initialized")

        return filter_table

    def get_split_anchor(self, traj):
        episode_id = self._episode_id_from_traj(traj, self.ep_table)
        if self.validation_mode == "hard":
            # Environment-level split: hold out entire labs
            lab_name = tf.strings.regex_replace(episode_id, r"\+.*$", "")
            return lab_name
        return episode_id

    def apply_restructure(self):
        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    traj["observation"]["cartesian_position"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            actions = convert_action_encoding(
                action=actions,
                from_encoding=self.spec.default_action_encoding,
                to_encoding=self.config.action_encoding,
                to_delta_cartesian_pose=True,
            )
            # Align lengths across modalities
            traj_len = tf.shape(actions)[0]
            episode_id = self._episode_id_from_traj(traj, self.ep_table)
            if not self.use_base_actions:
                lang_bytes = self.lang_table.lookup(episode_id)
                lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
                # Language actions may include an extra terminal step; crop to match action length
                lang_tensor = lang_tensor[:traj_len]
            else:
                lang_tensor = tf.fill([traj_len], tf.constant(""))
            # Sample instruction from merged table or fallback
            instr_bytes = self.instr_table.lookup(episode_id)
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
                # Guard against empty instruction arrays
                return tf.cond(
                    tf.greater(tf.shape(arr)[0], 0),
                    lambda: tf.random.shuffle(arr, seed=self.seed)[0],
                    lambda: fallback_instruction,
                )

            instruction = tf.cond(
                tf.greater(tf.strings.length(instr_bytes), 0),
                _sample_from_table,
                lambda: fallback_instruction,
            )

            instruction_vec = tf.fill([tf.shape(actions)[0]], instruction)

            if not self.use_base_actions:
                cam_idx = self.cam_table.lookup(episode_id)
                cam_images = [
                    traj["observation"]["exterior_image_1_left"],
                    traj["observation"]["exterior_image_2_left"],
                ]
                cam_images = tf.stack(cam_images, axis=0)  # shape (2, H, W, C)
                cam_idx_clamped = tf.clip_by_value(cam_idx, 0, tf.shape(cam_images)[0] - 1)
                exterior_img = tf.gather(cam_images, cam_idx_clamped)
            else:
                # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[], seed=self.seed) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )

            episode_id_vec = tf.fill([traj_len], episode_id)

            cartesian = traj["observation"]["cartesian_position"]
            gripper = traj["observation"]["gripper_position"]

            gripper = tf.cond(
                tf.equal(tf.rank(cartesian), tf.rank(gripper)),
                lambda: gripper,  # same rank → no change
                lambda: tf.expand_dims(gripper, axis=-1),  # add new axis if rank differs
            )

            state = tf.concat([cartesian, gripper], axis=-1)
            state = convert_state_encoding(
                state, from_encoding=self.spec.default_state_encoding, to_encoding=self.config.state_encoding
            )

            _return_dict = {
                "actions": tf.cast(actions, tf.float32),
                "observation": {
                    "exterior_image_1_left": exterior_img,
                    "state": tf.cast(state, tf.float32),
                    "cartesian_position": traj["observation"]["cartesian_position"],  # for need_calib
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "episode_id": episode_id_vec,
                "traj_metadata": traj["traj_metadata"],
                "raw_action": tf.cast(actions, tf.float32),
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
                # Attach control_frequency per step for downstream windowing/summarization
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
            }

            if self.use_idle_filter:
                if self.use_per_traj_filter:
                    # episode-level ranges lookup; compute per-step mask
                    # Ensure scalar episode key (metadata fields may be length-1 vectors)
                    rec_path = traj["traj_metadata"]["episode_metadata"]["recording_folderpath"][0]
                    file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
                    episode_key = rec_path + "--" + file_path
                    ranges_bytes = self.filter_table.lookup(episode_key)

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
                else:
                    step_id = (
                        traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                        + "--"
                        + traj["traj_metadata"]["episode_metadata"]["file_path"]
                        + "--"
                        + tf.as_string(tf.range(traj_len))
                    )
                    passes_filter = self.filter_table.lookup(step_id)
                    _return_dict["passes_filter"] = passes_filter

            if self.need_calib:
                intr = tf.io.parse_tensor(self.intr_table.lookup(episode_id), out_type=tf.float32)  # [4]
                extr = tf.reshape(
                    tf.io.parse_tensor(self.extr_table.lookup(episode_id), out_type=tf.float32),
                    [4, 4],
                )  # [4,4]
                intr_b = tf.broadcast_to(intr[None, :], [traj_len, 4])
                extr_b = tf.broadcast_to(extr[None, :, :], [traj_len, 4, 4])
                _return_dict["camera_intrinsics"] = intr_b  # [traj_len, 4]
                _return_dict["camera_extrinsics"] = extr_b  # [traj_len, 4, 4]

            if self.use_wrist_image:
                _return_dict["observation"]["wrist_image_left"] = traj["observation"]["wrist_image_left"]
            return _return_dict

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def apply_traj_transforms(
        self,
        action_chunk_size: int,
        summation_steps: int,
    ):
        self.dataset = self.dataset.traj_map(
            NormalizeActionAndProprio(
                norm_stats=self.dataset_statistics,
                normalization_type=self.action_proprio_normalization_type,
                action_key="actions",
                state_key="state",
            ),
            self.num_parallel_calls,
        )

        def pad_action_state(traj):
            # pad actions to action_dim
            traj["actions"] = tf.pad(traj["actions"], [[0, 0], [0, self.action_dim - tf.shape(traj["actions"])[-1]]])
            # pad state to action_dim
            traj["observation"]["state"] = tf.pad(
                traj["observation"]["state"],
                [[0, 0], [0, self.action_dim - tf.shape(traj["observation"]["state"])[-1]]],
            )
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks using shared indexing utility."""
            traj_len = tf.shape(traj["actions"])[0]
            action_chunk_indices = compute_window_indices(traj_len, action_chunk_size)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

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
            summation_indices = compute_window_indices(traj_len, summation_steps)

            # Trim window to control_frequency and pad to fixed length (summation_steps)
            trimmed_len = tf.minimum(tf.cast(self.control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))
            if not self.use_base_actions:
                la_window = tf.gather(traj["language_actions"], summation_indices[:, :trimmed_len])
                pad_len = summation_steps - trimmed_len

                def _pad_text():
                    pad = tf.fill([tf.shape(la_window)[0], pad_len], tf.constant("", dtype=tf.string))
                    return tf.concat([la_window, pad], axis=1)

                traj["language_actions"] = tf.cond(pad_len > 0, _pad_text, lambda: la_window)
            else:
                # Gather numeric actions for the future window: [T, trimmed_len, A]
                actions_window_trim = tf.gather(traj["raw_action"], summation_indices[:, :trimmed_len])
                pad_len = summation_steps - trimmed_len

                def _pad_numeric():
                    zeros_pad = tf.zeros(
                        [tf.shape(actions_window_trim)[0], pad_len, tf.shape(actions_window_trim)[-1]],
                        dtype=actions_window_trim.dtype,
                    )
                    return tf.concat([actions_window_trim, zeros_pad], axis=1)

                actions_window = tf.cond(pad_len > 0, _pad_numeric, lambda: actions_window_trim)

                # Convert per-step numeric rows to tf.string via serialization -> [T, summation_steps]
                flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
                serialized_flat = tf.map_fn(
                    lambda v: tf.io.serialize_tensor(v),
                    flat_rows,
                    fn_output_signature=tf.string,
                )
                traj["language_actions"] = tf.reshape(serialized_flat, [tf.shape(actions_window)[0], summation_steps])

            if self.vis_dataset:
                grouped_images = tf.gather(traj["observation"]["exterior_image_1_left"], summation_indices)
                traj["observation"]["exterior_image_1_left"] = grouped_images

                if self.use_wrist_image:
                    grouped_wrist_images = tf.gather(traj["observation"]["wrist_image_left"], summation_indices)
                    traj["observation"]["wrist_image_left"] = grouped_wrist_images

            # Group cartesian positions for start/end projection when needed
            if self.need_calib:
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
            if self.drop_gripper_oob:
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

    def apply_frame_filters(self):
        if self.use_idle_filter:

            def filter_from_dict(frame):
                return frame["passes_filter"]

            self.dataset = self.dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            self.dataset = self.dataset.map(remove_passes_filter)

        # Optional filter: drop samples where gripper projects out of the view
        if self.drop_gripper_oob:

            def _filter_in_view(frame):
                return frame["gripper_in_view"]

            self.dataset = self.dataset.filter(_filter_in_view)

            def _remove_in_view(frame):
                frame.pop("gripper_in_view")
                return frame

            self.dataset = self.dataset.map(_remove_in_view)

        def _remove_raw_action(frame):
            frame.pop("raw_action")
            return frame

        self.dataset = self.dataset.map(_remove_raw_action)

    def apply_traj_filters(self):
        # ------------------------------------------------------------------
        # Regex helpers for robust path/id extraction
        # ------------------------------------------------------------------

        def _id_ok(traj):
            episode_id = self._episode_id_from_traj(traj, self.ep_table)
            if tf.equal(episode_id, self.spec.default_ep_value):
                return tf.constant(value=False, dtype=tf.bool)
            # Look up by episode_id (NOT episode_path). Using episode_path here would filter everything out.
            lang = self.lang_table.lookup(episode_id)
            if tf.equal(lang, self.spec.default_lang_value):
                return tf.constant(value=False, dtype=tf.bool)
            return tf.logical_and(
                tf.not_equal(episode_id, self.spec.default_ep_value),
                tf.not_equal(lang, self.spec.default_lang_value),
            )

        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        def _has_instruction(traj):
            episode_id = self._episode_id_from_traj(traj, self.ep_table)
            instr_bytes = self.instr_table.lookup(episode_id)
            return tf.not_equal(instr_bytes, tf.constant(b"", dtype=tf.string))

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        # Prefer cheap regex path filter first, then id/lang checks
        self.dataset = self.dataset.filter(_path_ok)
        self.dataset = self.dataset.filter(_has_instruction)
        if not self.use_base_actions:
            self.dataset = self.dataset.filter(_id_ok)

    def apply_align_oxe_fmt(self):
        def _to_oxe_spec(traj):
            traj.pop("traj_metadata")
            traj.pop("episode_id")
            traj["observation"].pop("cartesian_position")

            return traj

        self.dataset = self.dataset.traj_map(_to_oxe_spec, self.num_parallel_calls)

    def __init__(
        self,
        data_dir: str,
        language_action_dir: str,
        config: "CoTDataConfig",
        *,  # Force keyword-only arguments
        action_chunk_size: int = 16,
        action_dim: int = 32,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # Validation support
        split_seed: int = 0,
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        align_oxe_fmt: bool = False,
        train_dataset=None,
        use_base_actions=True,
        **kwargs,
    ):
        super().__init__(
            dataset_name=config.droid_dataset_name,
            data_dir=data_dir,
            config=config,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
        )
        self.action_dim = action_dim
        self.use_wrist_image = bool(config.use_wrist_image)
        self.vis_dataset = bool(config.vis_dataset)
        self.use_idle_filter = bool(config.use_idle_filter)
        self.drop_gripper_oob = bool(config.drop_gripper_oob)
        self.need_calib = bool(config.vis_dataset or self.drop_gripper_oob)
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.use_per_traj_filter = bool(config.use_per_traj_filter)
        self.use_base_actions = use_base_actions
        # Persist per-dataset control frequency for later padding and policy use
        # Default to 15 for DROID datasets
        self.control_frequency: int = 15

        if train_dataset is not None:
            self.lang_table = train_dataset.lang_table
            self.ep_table = train_dataset.ep_table
            self.cam_table = train_dataset.cam_table
            self.intr_table = train_dataset.intr_table
            self.extr_table = train_dataset.extr_table
            self.instr_table = train_dataset.instr_table
            self.filter_table = train_dataset.filter_table

        else:
            if self.spec.lang_action_dir_name in language_action_dir:
                metadata_path = language_action_dir.replace(
                    self.spec.lang_action_dir_name, self.spec.metadata_path_name
                )
            elif self.spec.lang_action_dir_name_base in language_action_dir:
                metadata_path = language_action_dir.replace(
                    self.spec.lang_action_dir_name_base, self.spec.metadata_path_name
                )
            else:
                raise ValueError(f"Unknown language action directory: {language_action_dir}")

            self.lang_table = self.build_lang_action_table(language_action_dir)
            self.ep_table = self.build_lookup_table(metadata_path)
            self.cam_table, self.intr_table, self.extr_table = self.build_cam_tables(
                metadata_path, need_calib=self.need_calib
            )
            self.instr_table = self.build_instr_table(metadata_path)
            self.filter_table = self.build_filter_table(metadata_path, use_idle_filter=self.use_idle_filter)

        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)
        if cached_stats is not None:
            # Prefer early filtering when stats are already available to reduce downstream work.
            self.apply_traj_filters()
            self.split_val(split_seed=split_seed)
            self.apply_restructure()
            self.dataset_statistics = cached_stats
        else:
            # Build required fields first, compute stats on cardinality-preserving pipeline, then filter.
            self.apply_restructure()
            self.dataset_statistics = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="actions",
                state_key="state",
            )
            self.apply_traj_filters()
            self.split_val(split_seed=split_seed)

        self.apply_traj_transforms(
            action_chunk_size=action_chunk_size,
            summation_steps=30,
        )

        if align_oxe_fmt:
            self.apply_align_oxe_fmt()

        self.apply_flatten()

        self.apply_frame_filters()


class _SingleOXECoTRldsDatasetRaw(_SingleCoTRldsDatasetRaw):
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
        config: "CoTDataConfig",
        dataset_kwargs: dict,
        *,  # Force keyword-only arguments
        action_chunk_size: int = 16,
        action_dim: int = 32,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        split_seed: int = 0,
        seed: int = 0,
        split: str = "train",
        global_state_encoding: StateEncoding | None = None,
        global_action_encoding: ActionEncoding | None = None,
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
        self.action_dim = action_dim
        self.standardize_fn = dataset_kwargs["standardize_fn"]
        self.image_obs_keys = dataset_kwargs["image_obs_keys"]
        self.state_obs_keys = dataset_kwargs["state_obs_keys"]
        self.language_key = dataset_kwargs["language_key"]
        self.action_proprio_normalization_type = dataset_kwargs["action_proprio_normalization_type"]
        # self.action_normalization_mask = dataset_kwargs["action_normalization_mask"]
        self.state_encoding = dataset_kwargs["state_encoding"]
        self.action_encoding = dataset_kwargs["action_encoding"]
        self.global_state_encoding = global_state_encoding
        self.global_action_encoding = global_action_encoding
        dataset_frame_transform_kwargs = dataset_kwargs.get("dataset_frame_transform_kwargs", {})
        assert "primary" in self.image_obs_keys, "primary image is required"
        assert "wrist" in self.image_obs_keys, "wrist image is required"
        if self.language_key is not None:
            self.REQUIRED_KEYS.add(self.language_key)

        # Persist per-dataset control frequency for later padding and policy use
        self.control_frequency: int = int(dataset_kwargs["control_frequency"])  # constant for this dataset

        logging.info(f"Dataset kwargs: {dataset_kwargs}")

        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)
        if cached_stats is not None:
            # Prefer early filtering when stats are already available to reduce downstream work.
            self.apply_traj_filters()
            self.apply_restructure(use_wrist_image=config.use_wrist_image)
            self.split_val(split_seed=split_seed)
            self.dataset_statistics = cached_stats
        else:
            # Build required fields first, compute stats on cardinality-preserving pipeline, then filter.
            self.apply_restructure(use_wrist_image=config.use_wrist_image)
            self.dataset_statistics = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="action",
                state_key="proprio",
            )
            self.apply_traj_filters()
            self.split_val(split_seed=split_seed)

        # dataset_statistics = tree_map(np.array, dataset_statistics)
        # if self.action_normalization_mask is not None and "ego" not in self.dataset_name:
        #     if len(self.action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
        #         raise ValueError(
        #             f"Length of skip_normalization_mask ({len(self.action_normalization_mask)}) "
        #             f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
        #         )
        #     dataset_statistics["action"]["mask"] = np.array(self.action_normalization_mask)
        # self.apply_traj_transforms(action_chunk_size=action_chunk_size, summation_steps=config.summation_steps)
        # Use a fixed summation window across datasets to enable interleaving
        self.apply_traj_transforms(action_chunk_size=action_chunk_size, summation_steps=30)
        self.apply_repack_transforms(use_wrist_image=config.use_wrist_image)
        self.apply_flatten()
        self.apply_frame_filters(**dataset_frame_transform_kwargs)

    def apply_restructure(self, use_wrist_image: bool):
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

            traj["action"] = convert_action_encoding(
                action=traj["action"],
                from_encoding=self.action_encoding,
                to_encoding=self.global_action_encoding,
            )

            for new, old in self.image_obs_keys.items():
                img_key = f"image_{new}"
                if not use_wrist_image and new == "wrist":
                    continue
                if old is None:
                    new_obs[img_key] = tf.repeat("", traj_len)  # padding
                else:
                    new_obs[img_key] = old_obs[old]

            if self.state_obs_keys:
                # Note: instead of padding with zeros, we drop the key if it is None
                new_obs["proprio"] = tf.concat(
                    [tf.cast(old_obs[key], tf.float32) for key in self.state_obs_keys if key is not None],
                    axis=1,
                )
                new_obs["proprio"] = convert_state_encoding(
                    new_obs["proprio"],
                    from_encoding=self.state_encoding,
                    to_encoding=self.global_state_encoding,
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
                    arr = tf.reshape(raw_language_instruction, [-1])
                    return tf.random.shuffle(arr, seed=self.seed)[0]

                has_any_instruction = tf.reduce_any(
                    tf.greater(tf.strings.length(tf.reshape(raw_language_instruction, [-1])), 0)
                )

                instruction = tf.cond(has_any_instruction, _sample_from_table, lambda: fallback_instruction)

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

            assert "depth_gripper" not in new_obs, "depth images are not supported"

            traj = {
                "observation": new_obs,
                "task": task,
                "action": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": tf.repeat(traj_uid, traj_len),
                "raw_action": tf.cast(traj["action"], tf.float32),
                # Attach control_frequency per step for downstream windowing/summarization
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
            }

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def apply_traj_filters(self):
        # def is_nonzero_length(traj):
        #     return tf.shape(traj["action"])[0] > 0

        # # self.dataset = self.dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

        # self.dataset = self.dataset.filter(is_nonzero_length)
        return

    def get_split_anchor(self, traj):
        # Use the per-trajectory identifier (constant along time) to split deterministically
        return traj["trajectory_id"][0]

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
            NormalizeActionAndProprio(
                norm_stats=self.dataset_statistics,
                normalization_type=self.action_proprio_normalization_type,
                action_key="action",
                state_key="proprio",
            ),
            self.num_parallel_calls,
        )

        def pad_action_state(traj):
            # pad actions to action_dim
            traj["action"] = tf.pad(traj["action"], [[0, 0], [0, self.action_dim - tf.shape(traj["action"])[-1]]])
            # pad state to action_dim
            traj["observation"]["proprio"] = tf.pad(
                traj["observation"]["proprio"],
                [[0, 0], [0, self.action_dim - tf.shape(traj["observation"]["proprio"])[-1]]],
            )
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks using shared indexing utility."""
            traj_len = tf.shape(traj["action"])[0]

            action_chunk_indices = compute_window_indices(traj_len, action_chunk_size)
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
            traj_len = tf.shape(traj["raw_action"])[0]
            # First, create indices for summation (current + future steps)
            summation_indices = compute_window_indices(traj_len, summation_steps)

            # Trim to dataset control frequency and pad to fixed window length (summation_steps)
            # Note: self.control_frequency is a Python int constant per dataset instance
            trimmed_len = min(self.control_frequency, int(summation_steps))
            # Gather numeric actions for the future window up to control frequency: [T, trimmed_len, A]
            actions_window_trim = tf.gather(traj["raw_action"], summation_indices[:, :trimmed_len])
            pad_len = int(summation_steps) - trimmed_len
            if pad_len > 0:
                zeros_pad = tf.zeros(
                    [
                        tf.shape(actions_window_trim)[0],
                        pad_len,
                        tf.shape(actions_window_trim)[-1],
                    ],
                    dtype=actions_window_trim.dtype,
                )
                actions_window = tf.concat([actions_window_trim, zeros_pad], axis=1)
            else:
                actions_window = actions_window_trim

            # Unify spec with DROID by converting per-step numeric rows to tf.string via serialization.
            # Result shape: [T, summation_steps] tf.string (each element is a serialized [A] float32 tensor)
            flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
            serialized_flat = tf.map_fn(
                lambda v: tf.io.serialize_tensor(v),
                flat_rows,
                fn_output_signature=tf.string,
            )
            traj["language_actions"] = tf.reshape(
                serialized_flat,
                [tf.shape(actions_window)[0], int(summation_steps)],
            )

            return traj

        self.dataset = self.dataset.traj_map(group_language_actions, self.num_parallel_calls)

    def apply_frame_filters(
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

    def apply_repack_transforms(self, use_wrist_image: bool):
        def repack_transforms(traj):
            return_dict = {
                "actions": traj["action"],
                "observation": {
                    "exterior_image_1_left": traj["observation"]["image_primary"],
                    "state": traj["observation"]["proprio"],
                },
                "prompt": traj["task"]["language_instruction"],
                "language_actions": traj["language_actions"],
                "dataset_name": traj["dataset_name"],
                "control_frequency": traj["control_frequency"],
            }
            if use_wrist_image:
                return_dict["observation"]["wrist_image_left"] = traj["observation"]["image_wrist"]
            return return_dict

        self.dataset = self.dataset.traj_map(repack_transforms, self.num_parallel_calls)


class _OXECoTRldsDatasetsRaw:
    def __init__(
        self,
        config: "CoTDataConfig",
        data_dir: str,
        data_mix: str,
        action_dim: int = 32,
        action_chunk_size: int = 16,
        split_seed: int = 0,
        seed: int = 0,
        split: str = "train",
        balance_weights: bool = True,  # noqa: FBT001, FBT002
        traj_transform_threads: int | None = None,
        traj_read_threads: int | None = None,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
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
            data_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
        self.dataset_names = [dataset_kwargs["name"] for dataset_kwargs in dataset_kwargs_list]


        if len(sample_weights) != len(dataset_kwargs_list):
            raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")


        # Allocate Threads based on Weights
        threads_per_dataset = allocate_threads(traj_transform_threads, np.array(sample_weights))
        reads_per_dataset = allocate_threads(traj_read_threads, np.array(sample_weights))

        logging.info("Threads per Dataset: %s", threads_per_dataset)
        logging.info("Reads per Dataset: %s", reads_per_dataset)

        datasets, dataset_sizes, all_dataset_statistics = [], [], {}
        logging.info("Constructing datasets...")
        for dataset_kwargs, threads, reads in zip(  # noqa: B905
                dataset_kwargs_list,
                threads_per_dataset,
                reads_per_dataset,
            ):
            assert threads != tf.data.AUTOTUNE, "threads should not be AUTOTUNE"
            assert reads != tf.data.AUTOTUNE, "reads should not be AUTOTUNE"
            # Pass only accepted args to SingleOXECoTRldsDataset
            ds = _SingleOXECoTRldsDatasetRaw(
                dataset_name=dataset_kwargs["name"],
                data_dir=dataset_kwargs["data_dir"],
                config=config,
                dataset_kwargs=dataset_kwargs,
                action_chunk_size=action_chunk_size,
                num_parallel_reads=threads,
                num_parallel_calls=threads,
                split_seed=split_seed,
                seed=seed,
                split=split,
                global_state_encoding=config.state_encoding,
                global_action_encoding=config.action_encoding,
                action_dim=action_dim,
            )
            datasets.append(ds.dataset.with_ram_budget(1))
            dataset_statistics = ds.dataset_statistics
            dataset_sizes.append(dataset_statistics["state"].num_transitions)
            all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

        # Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        # primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
        primary_dataset_indices = np.array(list(range(len(sample_weights))))

        # Balance and Normalize Weights
        if balance_weights:
            sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
        unnormalized_sample_weights = sample_weights.copy()
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)

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


class DroidCoTRldsDataset(_DroidCoTRldsDatasetRaw):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool,  # noqa: FBT001
        max_samples: int | None,
        seed: int,
        config: "CoTDataConfig",
        split: str,
        action_horizon: int,
        action_dim: int,
        action_normalization_type: NormalizationType = NormalizationType.NORMAL,
        train_dataset: _DroidCoTRldsDatasetRaw | None = None,
    ):
        # Initialize the base class with only the raw kwargs
        super().__init__(
            data_dir=data_dir,
            language_action_dir=config.language_action_dir,
            config=config,
            split_seed=seed,
            seed=seed,
            split=split,
            action_normalization_type=action_normalization_type,
            action_chunk_size=action_horizon,
            action_dim=action_dim,
            train_dataset=train_dataset,
        )

        # Apply common shuffling/take/cache behavior
        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=self.want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        # Apply frame transforms via shared decode helper
        self.apply_frame_transforms(use_wrist_image=config.use_wrist_image, resize_resolution=config.resize_resolution)

        # Batch and prefetch
        self.dataset = batch_prefetch(self.dataset, batch_size)

    def apply_frame_transforms(self, use_wrist_image: bool, resize_resolution: tuple[int, int]):  # noqa: FBT001
        # Retained for compatibility; use shared helper
        decode_fn = make_decode_images_fn(
            primary_key="exterior_image_1_left",
            wrist_key="wrist_image_left",
            use_wrist_image=use_wrist_image,
            resize_to=resize_resolution,
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
        return self.dataset_statistics["state"].num_transitions


class OXECoTRldsDatasets(_OXECoTRldsDatasetsRaw):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool,  # noqa: FBT001
        max_samples: int | None,
        seed: int,
        config: "CoTDataConfig",
        split: str,
        action_horizon: int,
        action_dim: int = 32,
        balance_weights: bool = True,  # noqa: FBT001, FBT002
    ):
        total_threads = len(os.sched_getaffinity(0))
        want_val = split == "val"
        super().__init__(
            config=config,
            data_dir=data_dir,
            data_mix=config.data_mix,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            action_dim=action_dim,
            split_seed=seed,
            seed=seed,
            split=split,
            balance_weights=balance_weights,
            traj_transform_threads=int(total_threads * 0.3) if not want_val else int(total_threads * 0.1),
            traj_read_threads=int(total_threads * 0.3) if not want_val else int(total_threads * 0.1),
        )
        pprint_data_mixture(self.dataset_names, self.sample_weights)

        self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(self.datasets, self.sample_weights)

        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        decode_fn = make_decode_images_fn(
            primary_key="exterior_image_1_left",
            wrist_key="wrist_image_left",
            use_wrist_image=config.use_wrist_image,
            resize_to=config.resize_resolution,
        )
        self.dataset = self.dataset.frame_map(decode_fn, tf.data.AUTOTUNE)

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
        data_dir: str,
        batch_size: int,
        shuffle: bool,  # noqa: FBT001
        max_samples: int | None,
        seed: int,
        config: "CoTDataConfig",
        split: str,
        action_horizon: int,
        action_dim: int = 32,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        balance_weights: bool = True,  # noqa: FBT001, FBT002
    ):
        # Build sub-datasets with only their required args
        total_threads = len(os.sched_getaffinity(0))
        want_val = split == "val"

        oxe = _OXECoTRldsDatasetsRaw(
            config=config,
            data_dir=data_dir,
            data_mix=config.data_mix,
            action_chunk_size=action_horizon,
            action_dim=action_dim,
            split_seed=seed,
            seed=seed,
            split=split,
            balance_weights=balance_weights,
            # TODO: support different normalization type within combined dataset
            # action_proprio_normalization_type=action_proprio_normalization_type,
            traj_transform_threads=int(total_threads * 0.3) if not want_val else int(total_threads * 0.1),
            traj_read_threads=int(total_threads * 0.3) if not want_val else int(total_threads * 0.1),
        )

        droid = _DroidCoTRldsDatasetRaw(
            data_dir=config.droid_rlds_data_dir if config.droid_rlds_data_dir is not None else data_dir,
            language_action_dir=config.language_action_dir,
            config=config,
            action_chunk_size=action_horizon,
            action_dim=action_dim,
            split_seed=seed,
            seed=seed,
            split=split,
            action_proprio_normalization_type=action_proprio_normalization_type,
            align_oxe_fmt=True,
            num_parallel_reads=int(total_threads * 0.2) if not want_val else int(total_threads * 0.1),
            num_parallel_calls=int(total_threads * 0.2) if not want_val else int(total_threads * 0.1),
        )

        use_wrist_image = config.use_wrist_image
        all_datasets = [*oxe.datasets, droid.dataset]
        unnormalized_sample_weights = [
            *oxe.unnormalized_sample_weights,
            config.droid_weight * droid.dataset_statistics["state"].num_transitions,
        ]
        sample_weights = unnormalized_sample_weights / np.sum(unnormalized_sample_weights)
        pprint_data_mixture([*oxe.dataset_names, "droid"], sample_weights)

        logging.info("Interleaving datasets...")
        self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(all_datasets, sample_weights)

        # Apply common finalization
        self.dataset = maybe_shuffle_and_take(
            self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
        )

        # Apply frame transforms via shared helper
        logging.info("Applying frame transforms on dataset...")
        decode_fn = make_decode_images_fn(
            primary_key="exterior_image_1_left",
            wrist_key="wrist_image_left",
            use_wrist_image=use_wrist_image,
            resize_to=config.resize_resolution,
        )
        self.dataset = self.dataset.frame_map(
            decode_fn, int(total_threads * 0.5) if not want_val else int(total_threads * 0.1)
        )

        self.dataset.sample_weights = sample_weights
        self.dataset_length = oxe.dataset_length
        self.dataset_statistics = oxe.dataset_statistics

        self.dataset = batch_prefetch(self.dataset, batch_size)

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
