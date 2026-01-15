from collections.abc import Iterable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from dataclasses import field
import logging
import pickle
import re
import textwrap
from typing import Any

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import jax
from jax.experimental import multihost_utils as mh
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
import wandb

from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
from openpi_cot.training import utils as _utils
import openpi_cot.training.utils as training_utils

AXIS_PERM = np.array([0, 2, 1], dtype=np.int32)
AXIS_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def infer_local_batch_size(obs_local: CoTObservation | None) -> int:
    if obs_local is None:
        return 0
    candidate_sizes: list[int] = []
    state_local = getattr(obs_local, "state", None)
    if state_local is not None:
        candidate_sizes.append(int(np.shape(state_local)[0]))
    prompt_local = getattr(obs_local, "tokenized_prompt", None)
    if prompt_local is not None:
        candidate_sizes.append(int(np.shape(prompt_local)[0]))
    image_values = list(getattr(obs_local, "images", {}).values())
    for img in image_values:
        if img is not None:
            candidate_sizes.append(int(np.shape(img)[0]))
            break
    return candidate_sizes[0] if candidate_sizes else 0


@dataclass
class HardExampleTracker:
    tokenizer: PaligemmaCoTTokenizer
    hard_quantile: float = 0.99
    buffer_ratio: float = 0.07
    buffer_min: int = 32
    buffer_slack: int = 32
    max_hard_examples: int = 50
    resize_hw: tuple[int, int] | None = (128, 128)
    _interval_losses: list[np.ndarray] = field(default_factory=list, init=False)
    _interval_total_samples: int = field(default=0, init=False)
    _hard_example_buffer: list[dict[str, Any]] = field(default_factory=list, init=False)
    _hard_example_keys: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    # Track batch references for lazy image extraction
    _batch_cache: dict[tuple[int, int], tuple[CoTObservation, _model.Actions]] = field(default_factory=dict, init=False)
    _extraction_failures: int = field(default=0, init=False)

    def update(self, per_sample_losses: np.ndarray | None) -> None:
        if per_sample_losses is None:
            return
        arr = np.asarray(per_sample_losses, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._interval_losses.append(arr)
        self._interval_total_samples += int(arr.size)

    def add_local_examples(
        self,
        step_idx: int,
        host_batch_local: tuple[CoTObservation, _model.Actions] | None,
        local_losses: np.ndarray | None,
        global_idx_base: int,
        *,
        process_idx: int,
    ) -> None:
        """Add examples with high losses to the buffer.

        Strategy: Store loss metadata immediately, cache batch reference for lazy image extraction.
        This ensures we never miss high-loss samples due to image extraction failures.
        """
        if host_batch_local is None or local_losses is None:
            return
        losses = np.asarray(local_losses, dtype=np.float32).reshape(-1)
        if losses.size == 0:
            return
        capacity = self._compute_buffer_capacity()
        assert capacity > 0
        buffer_len = len(self._hard_example_buffer)
        if buffer_len < capacity:
            # To guarantee global top-k correctness across the interval,
            # consider up to the top-`capacity` samples from this step, not just the remaining slots.
            top_n = int(min(losses.size, capacity))
            if top_n <= 0:
                candidate_indices = []
            elif top_n >= losses.size:
                candidate_indices = list(range(losses.size))
            else:
                candidate_indices = np.argpartition(losses, -top_n)[-top_n:].tolist()
        else:
            # Buffer full: only candidates that can enter the global top-`capacity` are those
            # with loss >= current minimum in the buffer.
            threshold_loss = self._hard_example_buffer[-1]["loss"]
            candidate_indices = np.flatnonzero(losses >= threshold_loss).tolist()
        if not candidate_indices:
            return
        candidate_indices = sorted({int(idx) for idx in candidate_indices})
        new_indices = [
            idx
            for idx in candidate_indices
            if (process_idx, step_idx, int(global_idx_base + idx)) not in self._hard_example_keys
        ]
        if not new_indices:
            return

        # Cache batch for lazy image extraction
        batch_key = (process_idx, step_idx)
        self._batch_cache[batch_key] = host_batch_local

        # Store metadata immediately WITHOUT requiring image extraction
        # This ensures we never lose high-loss samples due to image failures
        added_count = 0
        for local_idx in new_indices:
            loss_val = float(losses[local_idx])
            global_idx = int(local_idx + global_idx_base)

            entry = {
                "loss": loss_val,
                "step": step_idx,
                "local_idx": int(local_idx),
                "global_idx": global_idx,
                "process_index": int(process_idx),
                # Store None placeholders - will extract lazily in log_if_ready()
                "image": None,
                "language_action": None,
                "dataset_name": None,
                "prompt": None,
            }
            self._hard_example_buffer.append(entry)
            self._hard_example_keys.add((process_idx, step_idx, global_idx))
            added_count += 1

        # Sort and trim to capacity
        self._hard_example_buffer.sort(key=lambda e: e["loss"], reverse=True)
        capacity = self._compute_buffer_capacity()
        if len(self._hard_example_buffer) > capacity:
            for removed in self._hard_example_buffer[capacity:]:
                self._hard_example_keys.discard((removed["process_index"], removed["step"], removed["global_idx"]))
            del self._hard_example_buffer[capacity:]

        if added_count > 0:
            logging.debug(
                f"[HardExampleTracker] Added {added_count} candidates from step {step_idx} "
                f"(loss range: {min(losses[new_indices]):.4f}-{max(losses[new_indices]):.4f}), "
                f"buffer size: {len(self._hard_example_buffer)}/{capacity}"
            )

    def log_if_ready(self, step_idx: int) -> dict[str, Any] | None:
        """Prepare payload with top-K hard examples, extracting images lazily."""
        if not self._interval_losses:
            self.reset()
            return None
        interval_all = np.concatenate(self._interval_losses, axis=0)
        total_samples = int(interval_all.size)
        hard_to_log = sorted(self._hard_example_buffer, key=lambda e: e["loss"], reverse=True)[: self.max_hard_examples]
        quantile_threshold = float("nan")
        if total_samples > 0 and hard_to_log:
            target = min(self.max_hard_examples, total_samples)
            quantile_prob = 1.0 - target / total_samples
            quantile_prob = float(np.clip(quantile_prob, 0.0, 1.0))
            quantile_threshold = float(np.quantile(interval_all, quantile_prob))
            min_logged_loss = float(hard_to_log[-1]["loss"])
            if not np.isfinite(quantile_threshold) or quantile_threshold < min_logged_loss:
                quantile_threshold = min_logged_loss

        # LAZY IMAGE EXTRACTION: Only extract images for the top-K samples that will be logged
        # This avoids expensive extraction for samples that won't make the cut
        extraction_successes = 0
        extraction_failures = 0

        for entry in hard_to_log:
            if entry["image"] is not None:
                # Already extracted
                continue

            # Extract image lazily from cached batch
            batch_key = (entry["process_index"], entry["step"])
            batch = self._batch_cache.get(batch_key)

            if batch is None:
                logging.warning(
                    f"[HardExampleTracker] Batch cache miss for step={entry['step']}, "
                    f"process={entry['process_index']}, global_idx={entry['global_idx']}"
                )
                extraction_failures += 1
                continue

            try:
                visuals = visualize_language_actions(
                    batch,
                    self.tokenizer,
                    indices=[entry["local_idx"]],
                    max_examples=1,
                    resize_hw=self.resize_hw,
                )

                if visuals and len(visuals) > 0:
                    vis = visuals[0]
                    entry["image"] = vis["image"]
                    entry["language_action"] = vis.get("language_action", "") or ""
                    entry["dataset_name"] = vis.get("dataset_name", "") or ""
                    entry["prompt"] = vis.get("prompt", "") or ""
                    extraction_successes += 1
                else:
                    logging.warning(
                        f"[HardExampleTracker] Image extraction failed for "
                        f"step={entry['step']}, global_idx={entry['global_idx']}, loss={entry['loss']:.4f}"
                    )
                    extraction_failures += 1
            except Exception as e:
                logging.warning(
                    f"[HardExampleTracker] Exception during image extraction for global_idx={entry['global_idx']}: {e}"
                )
                extraction_failures += 1

        # Log extraction statistics
        if extraction_successes > 0 or extraction_failures > 0:
            logging.info(
                f"[HardExampleTracker] Lazy extraction: {extraction_successes} success, "
                f"{extraction_failures} failures out of {len(hard_to_log)} top samples"
            )

        # Filter out entries without images for logging
        entries_with_images = [e for e in hard_to_log if e["image"] is not None]

        payload = {
            "entries": entries_with_images,
            "quantile_threshold": quantile_threshold,
            "total_samples": total_samples,
            "max_hard_examples": self.max_hard_examples,
            "step": step_idx,
        }

        # Log buffer statistics before reset
        if hard_to_log:
            buffer_losses = [e["loss"] for e in hard_to_log]
            logging.info(
                f"[HardExampleTracker] Top-{len(hard_to_log)} losses: "
                f"max={max(buffer_losses):.4f}, min={min(buffer_losses):.4f}, "
                f"logged_with_images={len(entries_with_images)}"
            )

        self.reset()
        if not entries_with_images and total_samples == 0:
            return None
        return payload

    def reset(self) -> None:
        self._interval_losses.clear()
        self._interval_total_samples = 0
        self._hard_example_buffer.clear()
        self._hard_example_keys.clear()
        self._batch_cache.clear()  # Clear cached batches

    def _compute_buffer_capacity(self) -> int:
        # Always maintain capacity equal to the maximum number of hard examples to log.
        # This ensures we can always retain the top-k (descending by loss) across the interval.
        return int(self.max_hard_examples)


def _decode_reasoning_strings(obs: CoTObservation, tokenizer) -> list[str]:
    """Extract and decode the reasoning (language action) tokens per example.

    Returns one decoded string per example. If reasoning fields are absent, returns [].
    """
    tokens = _utils.to_local_array(obs.tokenized_prompt)
    rmask = _utils.to_local_array(obs.tokenized_prompt_mask)
    langact_mask = _utils.to_local_array(obs.tokenized_langact_mask)
    texts: list[str] = []
    lang_acts: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        text = tokenizer.decode(sel.astype(np.int32))
        if langact_mask is None:
            lang_acts.append("")
        else:
            lang_act = tokens[i][langact_mask[i].astype(bool)]
            lang_act = tokenizer.decode(lang_act.astype(np.int32))
            lang_acts.append(lang_act)
        texts.append(text)
    return texts, lang_acts


def get_language_actions(batch, tok):
    texts, _ = _decode_reasoning_strings(batch[0], tok)
    return texts


def visualize_language_actions(
    batch: tuple[CoTObservation, _model.Actions],
    tok: PaligemmaCoTTokenizer,
    *,
    indices: Sequence[int] | None = None,
    max_examples: int | None = 5,
    image_keys: Iterable[str] | None = None,
    resize_hw: tuple[int, int] | None = None,
) -> list[Mapping[str, object]]:
    """Return combined RGB images and decoded language actions for selected examples.

    Args:
        batch: A tuple of (`CoTObservation`, actions) as produced by the dataloader.
        tok: Tokenizer used for decoding language tokens.
        indices: Optional iterable of batch indices to visualize.
        max_examples: Maximum number of examples to return.
        image_keys: Optional iterable specifying the order of image keys to concatenate.

    Returns:
        A list of dictionaries with keys:
            ``image`` (np.ndarray uint8 HxWx3), ``language_action`` (str), ``index`` (int).
    """

    obs, _ = batch
    images = {key: _utils.to_local_array(value) for key, value in obs.images.items() if value is not None}
    if not images:
        raise ValueError("No images found")

    order = list(image_keys) if image_keys is not None else sorted(images.keys())

    batch_sizes = [arr.shape[0] for arr in images.values() if arr is not None and arr.ndim >= 1]
    if not batch_sizes:
        raise ValueError("No images found")
    batch_size = min(batch_sizes)

    texts = get_language_actions(batch, tok)

    if indices is None:
        indices_list = list(range(batch_size))
    else:
        indices_list = [i for i in indices if 0 <= i < batch_size]

    if max_examples is not None:
        indices_list = indices_list[:max_examples]

    visuals: list[Mapping[str, object]] = []
    for idx in indices_list:
        per_cam: list[np.ndarray] = []
        for key in order:
            arr = images.get(key)
            if arr is None or idx >= arr.shape[0]:
                continue
            frame = np.asarray(arr[idx])
            if frame.ndim > 3:
                frame = frame[0]
            if np.issubdtype(frame.dtype, np.floating):
                frame = ((frame + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if resize_hw is not None and frame.shape[:2] != resize_hw:
                if HAS_CV2:
                    frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
                else:
                    # Fallback: simple nearest-neighbor resize using numpy indexing
                    h_old, w_old = frame.shape[:2]
                    h_new, w_new = resize_hw
                    row_idx = (np.arange(h_new) * h_old // h_new).astype(np.int32)
                    col_idx = (np.arange(w_new) * w_old // w_new).astype(np.int32)
                    frame = frame[row_idx[:, None], col_idx[None, :]]
            per_cam.append(frame)

        if not per_cam:
            continue

        if len(per_cam) == 1:
            combined = per_cam[0]
        else:
            try:
                combined = np.concatenate(per_cam, axis=1)
            except ValueError:
                # Pad images to match the maximum height before concatenation
                max_h = max(img.shape[0] for img in per_cam)
                padded: list[np.ndarray] = []
                for img in per_cam:
                    if img.shape[0] == max_h:
                        padded.append(img)
                        continue
                    pad_total = max_h - img.shape[0]
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                    pad_spec = ((pad_top, pad_bottom), (0, 0), (0, 0))
                    padded_img = np.pad(img, pad_spec, mode="constant")
                    padded.append(padded_img)
                try:
                    combined = np.concatenate(padded, axis=1)
                except ValueError:
                    logging.warning("Failed to concatenate images for index %d due to incompatible shapes", idx)
                    combined = per_cam[0]

        text = texts[idx] if idx < len(texts) else ""
        visuals.append({"image": combined, "language_action": text, "index": idx})

    return visuals


def _normalize_image_for_viz(image: np.ndarray | jax.Array) -> np.ndarray:
    """Convert JAX/NumPy image tensors with varying dtypes/shapes to uint8 RGB."""
    arr = np.asarray(image)
    # Remove singleton dimensions that occasionally show up (e.g., [1, H, W, C]).
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype == np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    arr = arr.astype(np.float32)
    arr_min = float(np.min(arr)) if arr.size else 0.0
    arr_max = float(np.max(arr)) if arr.size else 1.0
    if arr_max <= 1.05 and arr_min >= -0.05:
        # Already scaled between [0, 1]
        arr = arr.clip(0.0, 1.0) * 255.0
    elif arr_max <= 1.05 and arr_min >= -1.05:
        # Likely in [-1, 1]
        arr = ((arr + 1.0) * 0.5).clip(0.0, 1.0) * 255.0
    else:
        denom = max(arr_max - arr_min, 1e-6)
        arr = ((arr - arr_min) / denom).clip(0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    max_chars = max(1, max_chars)
    return textwrap.wrap(text, width=max_chars, break_long_words=False, replace_whitespace=True) or [""]


def create_rollout_visualization(image, gt_text: str, pred_text: str) -> np.ndarray:
    """Create a visualization with predicted/ground-truth text under the image."""
    img = _normalize_image_for_viz(image)
    h, w = img.shape[:2]

    max_chars = max(24, w // 12)
    gt_lines = _wrap_lines(f"GT: {gt_text or ''}", max_chars)
    pred_lines = _wrap_lines(f"Pred: {pred_text or ''}", max_chars)
    all_lines = gt_lines + [""] + pred_lines
    line_height = max(18, int(round(h * 0.03)))
    pad_height = max(60, (len(all_lines) + 1) * (line_height + 4))
    pad = np.zeros((pad_height, w, 3), dtype=np.uint8)

    if HAS_CV2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(1.2, w / 640.0))
        thickness = max(1, int(round(scale * 2)))
        y = line_height
        for line in all_lines:
            if not line.strip():
                y += line_height
                continue
            cv2.putText(
                pad,
                line,
                (10, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += line_height + 4
    else:
        try:
            from PIL import Image
            from PIL import ImageDraw
            from PIL import ImageFont

            pad_img = Image.fromarray(pad)
            draw = ImageDraw.Draw(pad_img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            y = line_height
            for line in all_lines:
                if not line.strip():
                    y += line_height
                    continue
                draw.text((10, y - line_height // 3), line, fill=(255, 255, 255), font=font)
                y += line_height + 4
            pad = np.asarray(pad_img)
        except Exception:
            # Leave pad black if PIL is unavailable.
            pass

    combined = np.concatenate([img, pad], axis=0)
    return combined


@dataclasses.dataclass
class HostBatchCache:
    host_batch: tuple[CoTObservation, _model.Actions] | None = None
    local_batch_size: int = 0
    step: int | None = None

    def ensure(
        self,
        *,
        step: int,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> tuple[tuple[CoTObservation, _model.Actions] | None, int]:
        if self.step != step:
            self.host_batch = jax.tree.map(training_utils.to_local_array, batch)
            obs_local = self.host_batch[0] if self.host_batch else None
            self.local_batch_size = infer_local_batch_size(obs_local)
            self.step = step
        return self.host_batch, self.local_batch_size


@dataclass
class DatasetLogTracker:
    """Tracks per-dataset logging counts to ensure uniform sample distribution across datasets."""

    tokenizer: PaligemmaCoTTokenizer
    _dataset_log_counts: dict[str, int] = field(default_factory=dict, init=False)

    def get_dataset_names_from_batch(self, batch: tuple[CoTObservation, _model.Actions]) -> list[str]:
        """Extract dataset names from batch."""
        obs = batch[0]
        if not hasattr(obs, "tokenized_dataset_name"):
            return []

        tokenized_names = _utils.to_local_array(obs.tokenized_dataset_name)
        if tokenized_names is None:
            return []

        dataset_names = []
        for i in range(tokenized_names.shape[0]):
            try:
                name = self.tokenizer.decode(tokenized_names[i])
                name = name.strip()
                dataset_names.append(name)
            except Exception as e:
                logging.warning(f"Failed to decode dataset name for sample {i}: {e}")
                dataset_names.append("unknown")

        return dataset_names

    def select_indices_uniform(
        self,
        dataset_names: list[str],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Select indices to ensure uniform sampling across datasets.

        Strategy: Prioritize datasets with fewer logged samples, then randomly sample
        within each dataset.
        """
        if not dataset_names or num_to_select <= 0:
            return []

        # Group indices by dataset
        dataset_to_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(dataset_names):
            dataset_to_indices.setdefault(name, []).append(idx)

        # Sort datasets by their current log count (ascending)
        datasets_sorted = sorted(dataset_to_indices.keys(), key=lambda d: self._dataset_log_counts.get(d, 0))

        selected_indices = []
        round_idx = 0

        # Round-robin selection: prioritize datasets with fewer samples logged
        while len(selected_indices) < num_to_select:
            added_any = False
            for dataset_name in datasets_sorted:
                if len(selected_indices) >= num_to_select:
                    break

                indices_for_dataset = dataset_to_indices[dataset_name]

                # Shuffle indices for this dataset to get random samples
                if round_idx == 0:
                    rng.shuffle(indices_for_dataset)

                if round_idx < len(indices_for_dataset):
                    selected_indices.append(indices_for_dataset[round_idx])
                    added_any = True

            if not added_any:
                break
            round_idx += 1

        return selected_indices

    def update_counts(self, dataset_names: list[str], selected_indices: list[int]) -> None:
        """Update logging counts for selected datasets."""
        for idx in selected_indices:
            if idx < len(dataset_names):
                dataset_name = dataset_names[idx]
                self._dataset_log_counts[dataset_name] = self._dataset_log_counts.get(dataset_name, 0) + 1

    def get_stats(self) -> dict[str, int]:
        """Get current logging statistics."""
        return dict(self._dataset_log_counts)


def log_random_examples(
    step: int,
    host_batch: tuple[CoTObservation, _model.Actions] | None,
    tokenizer: PaligemmaCoTTokenizer,
    *,
    local_batch_size: int,
    num_random: int = 5,
    dataset_log_tracker: DatasetLogTracker | None = None,
    prefix: str = "train",
) -> None:
    if host_batch is None or local_batch_size <= 0:
        return
    count = min(num_random, local_batch_size)
    if count <= 0:
        return
    process_idx = getattr(jax, "process_index", lambda: 0)()
    rng_seed = int(step + 997 * process_idx)
    rng_local = np.random.default_rng(rng_seed)

    # If tracker is provided, use uniform sampling across datasets
    if dataset_log_tracker is not None:
        dataset_names = dataset_log_tracker.get_dataset_names_from_batch(host_batch)
        if dataset_names:
            rand_idx = dataset_log_tracker.select_indices_uniform(dataset_names, count, rng_local)
            dataset_log_tracker.update_counts(dataset_names, rand_idx)
        else:
            # Fallback to random sampling if dataset names not available
            rand_idx = rng_local.choice(local_batch_size, size=count, replace=False).tolist()
    else:
        # Original random sampling
        rand_idx = rng_local.choice(local_batch_size, size=count, replace=False).tolist()

    random_visuals = visualize_language_actions(
        host_batch,
        tokenizer,
        indices=rand_idx,
        max_examples=count,
    )
    if not random_visuals:
        return
    images_to_log = []
    for vis in random_visuals:
        caption_text = vis.get("prompt", "") or ""
        caption_text += vis.get("language_action", "") or ""
        caption_text += vis.get("dataset_name", "") or ""
        caption = f"{caption_text}"
        images_to_log.append(wandb.Image(vis["image"], caption=caption))
    if images_to_log:
        wandb.log({f"{prefix}/random_examples": images_to_log}, step=step)


def _log_entries(entries: list[dict[str, Any]], *, step: int, quantile_threshold: float, total_samples: int) -> None:
    if not entries:
        return
    log_images = []
    for entry in entries:
        caption_text = entry.get("prompt", "") or ""
        caption_text += entry.get("language_action", "") or ""
        caption_text += entry.get("dataset_name", "") or ""
        host_idx = entry.get("process_index")
        host_tag = f"host={host_idx}" if host_idx is not None else "host=?"
        caption = f"{host_tag} loss={entry['loss']:.4f}"
        log_images.append(wandb.Image(entry["image"], caption=f"{caption} | {caption_text}"))
    wandb_payload = {
        "train/hard_examples": log_images,
        "train/hard_examples_loss_quantile": quantile_threshold,
        "train/hard_examples_count": len(log_images),
        "train/hard_examples_interval_samples": total_samples,
    }
    wandb.log(wandb_payload, step=step)


def log_hard_examples_payload(payload: dict[str, Any]) -> None:
    if not payload:
        return
    entries: list[dict[str, Any]] = payload.get("entries", [])
    max_examples = int(payload.get("max_hard_examples", len(entries) or 0))
    quantile_threshold = float(payload.get("quantile_threshold", float("nan")))
    total_samples = int(payload.get("total_samples", 0))
    step = int(payload.get("step", 0))

    process_count = getattr(jax, "process_count", lambda: 1)()
    process_idx = getattr(jax, "process_index", lambda: 0)()

    if process_count == 1:
        _log_entries(
            entries[:max_examples], step=step, quantile_threshold=quantile_threshold, total_samples=total_samples
        )
        return

    num_slots = int(max_examples)
    if num_slots <= 0:
        return

    local_payload = {
        "process_index": process_idx,
        "entries": entries,
    }
    try:
        local_blob = pickle.dumps(local_payload)
    except Exception as exc:
        logging.warning("Failed to serialize hard example payload on host %d: %s", process_idx, exc)
        return

    blob_arr = np.frombuffer(local_blob, dtype=np.uint8)
    blob_len = np.array([blob_arr.size], dtype=np.int32)

    gathered_lengths = mh.process_allgather(blob_len, tiled=False)
    lengths_np = np.asarray(gathered_lengths, dtype=object)
    max_len = 0
    if lengths_np.size > 0:
        max_len = int(max(int(np.asarray(x)[0]) for x in gathered_lengths))
    if max_len <= 0:
        max_len = int(blob_arr.size)

    padded = np.zeros((max_len,), dtype=np.uint8)
    if blob_arr.size > 0:
        padded[: blob_arr.size] = blob_arr

    gathered_blobs = mh.process_allgather(padded, tiled=False)

    if process_idx != 0:
        return

    length_list = []
    for x in gathered_lengths:
        arr = np.asarray(x)
        if arr.size == 0:
            length_list.append(0)
        elif arr.ndim == 0:
            length_list.append(int(arr))
        else:
            length_list.append(int(arr.flat[0]))

    all_entries: list[dict[str, Any]] = []
    for arr, length in zip(gathered_blobs, length_list):
        if length <= 0:
            continue
        try:
            host_payload = pickle.loads(arr[:length].tobytes())
        except Exception as exc:
            logging.warning("Failed to deserialize hard example payload on process 0: %s", exc)
            continue
        host_idx = host_payload.get("process_index", 0)
        for entry in host_payload.get("entries", []):
            entry = dict(entry)
            entry.setdefault("process_index", host_idx)
            all_entries.append(entry)

    if not all_entries:
        return

    # Sort per host so that we can sample fairly across processes.
    entries_by_host: dict[int, list[dict[str, Any]]] = {}
    for entry in all_entries:
        host_idx = int(entry.get("process_index", 0))
        entries_by_host.setdefault(host_idx, []).append(entry)

    for host_list in entries_by_host.values():
        host_list.sort(key=lambda e: e.get("loss", float("-inf")), reverse=True)

    selected: list[dict[str, Any]] = []
    round_idx = 0
    host_ids = sorted(entries_by_host.keys())
    while len(selected) < max_examples:
        added_any = False
        for host_id in host_ids:
            host_list = entries_by_host.get(host_id, [])
            if round_idx < len(host_list):
                selected.append(host_list[round_idx])
                added_any = True
                if len(selected) >= max_examples:
                    break
        if not added_any:
            break
        round_idx += 1

    # If we still have room (e.g., fewer hosts than slots), fill with remaining highest-loss entries.
    if len(selected) < max_examples:
        remaining = []
        for host_id in host_ids:
            host_list = entries_by_host[host_id]
            remaining.extend(host_list[round_idx:])
        remaining.sort(key=lambda e: e.get("loss", float("-inf")), reverse=True)
        for entry in remaining:
            if len(selected) >= max_examples:
                break
            selected.append(entry)

    # Keep overall ordering by loss for readability.
    selected.sort(key=lambda e: e.get("loss", float("-inf")), reverse=True)
    _log_entries(selected, step=step, quantile_threshold=quantile_threshold, total_samples=total_samples)


def prepare_eval_batch(batch, *, global_max_cut_idx: int | None = None):
    # Process the batch to remove reasoning and update masks
    obs, actions = batch

    # Find the first True index for each sample in the batch
    # obs.tokenized_langact_mask has shape [batch_size, seq_len]
    batch_size = obs.tokenized_langact_mask.shape[0]

    # For each sample, find the first index where tokenized_langact_mask is True
    cut_indices = []
    for i in range(batch_size):
        mask = obs.tokenized_langact_mask[i]
        # Find first True index
        true_indices = jnp.where(mask)[0]
        if true_indices.shape[0] > 0:
            cut_idx = int(true_indices[0])
        else:
            # If no True values, keep the entire sequence
            cut_idx = int(mask.shape[0])
        cut_indices.append(cut_idx)

    # Find the maximum cut index (longest sequence after cutting)
    max_cut_idx = max(cut_indices)
    if global_max_cut_idx is not None:
        max_cut_idx = max(max_cut_idx, int(global_max_cut_idx))

    # Cut and pad each sample
    new_tokenized_prompt_list = []
    new_tokenized_prompt_mask_list = []
    new_tokenized_langact_mask_list = []

    for i in range(batch_size):
        cut_idx = cut_indices[i]

        # Cut to the first True index (keep everything before reasoning)
        prompt_cut = obs.tokenized_prompt[i, :cut_idx]
        prompt_mask_cut = obs.tokenized_prompt_mask[i, :cut_idx]
        # Since we're removing reasoning, langact_mask should be all False
        langact_mask_cut = jnp.zeros(cut_idx, dtype=jnp.bool_)

        # Right-pad to max_cut_idx
        pad_len = max_cut_idx - cut_idx
        if pad_len > 0:
            prompt_padded = jnp.concatenate([prompt_cut, jnp.zeros(pad_len, dtype=prompt_cut.dtype)])
            prompt_mask_padded = jnp.concatenate([prompt_mask_cut, jnp.zeros(pad_len, dtype=jnp.bool_)])
            langact_mask_padded = jnp.concatenate([langact_mask_cut, jnp.zeros(pad_len, dtype=jnp.bool_)])
        else:
            prompt_padded = prompt_cut
            prompt_mask_padded = prompt_mask_cut
            langact_mask_padded = langact_mask_cut

        new_tokenized_prompt_list.append(prompt_padded)
        new_tokenized_prompt_mask_list.append(prompt_mask_padded)
        new_tokenized_langact_mask_list.append(langact_mask_padded)

    # Stack back into batch
    new_tokenized_prompt = jnp.stack(new_tokenized_prompt_list)
    new_tokenized_prompt_mask = jnp.stack(new_tokenized_prompt_mask_list)
    new_tokenized_langact_mask = jnp.stack(new_tokenized_langact_mask_list) * False

    new_obs = dataclasses.asdict(obs)
    new_obs["tokenized_prompt"] = new_tokenized_prompt
    new_obs["tokenized_prompt_mask"] = new_tokenized_prompt_mask
    new_obs["tokenized_langact_mask"] = new_tokenized_langact_mask
    new_obs["token_loss_mask"] = None
    new_obs["critical_token_mask"] = None
    new_obs["number_token_mask"] = None
    new_obs["direction_token_mask"] = None
    new_obs = CoTObservation(**new_obs)
    return (new_obs, actions)


def subsample_batch(
    batch: tuple[CoTObservation, _model.Actions],
    idx: jax.Array,
) -> tuple[CoTObservation, _model.Actions]:
    obs, acts = batch

    def take0(x):
        return jnp.take(x, idx, axis=0)

    obs_k = jax.tree.map(take0, obs)
    acts_k = jax.tree.map(take0, acts)
    return obs_k, acts_k


def _parse_language_delta_cm(text: str) -> np.ndarray | None:
    """Parse summed language action text -> net [right, forward, down] in cm.

    Accepts parts joined by " and ", like: "move right 10.3cm and move up 1.2cm and move forward 1.35cm".
    Recognized directions: left/right, forward/backward, up/down. Units: mm, cm, m.
    Returns None if no valid movements found.
    """
    totals = {
        "left": 0.0,
        "right": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "up": 0.0,
        "down": 0.0,
    }
    any_valid = False
    for part in filter(None, [p.strip() for p in text.split(" and ")]):
        m = re.match(r"move\s+(\w+)\s+([-+]?\d*\.?\d+)\s*(\w+)", part, flags=re.IGNORECASE)
        if not m:
            continue
        direction = m.group(1).lower()
        value = float(m.group(2))
        unit = m.group(3).lower()
        # Normalize to cm
        if unit.startswith("mm"):
            value = value / 10.0
        elif unit == "m" or (unit.startswith("m") and not unit.startswith("mm")):
            value = value * 100.0
        totals[direction] = totals.get(direction, 0.0) + value
        any_valid = True
    if "set gripper to" in text:
        any_valid = True
    if not any_valid:
        return None
    right = totals["right"] - totals["left"]
    forward = totals["forward"] - totals["backward"]
    down = totals["down"] - totals["up"]
    return np.array([right, forward, down], dtype=np.float32)


def _invert_camera_axis_map(v_cm: np.ndarray) -> np.ndarray:
    """Invert AXIS_PERM mapping to camera-frame delta in metres.

    Mirrors scripts/visualization/train_vis_gripper.py logic.
    """

    t_cam = np.zeros(3, dtype=np.float32)
    t_cam[AXIS_PERM] = (v_cm / 100.0) / AXIS_SIGN
    return t_cam


def _project_point(
    base_xyz: np.ndarray,
    cam_T_base: np.ndarray,
    intr: np.ndarray,
    out_hw: tuple[int, int],
) -> tuple[int, int] | None:
    """Project base-frame 3D point to pixel coordinates respecting resize_with_pad letterboxing.

    intr: [fx, fy, cx, cy] measured at calibration resolution (Wc≈2*cx, Hc≈2*cy).
    """
    if base_xyz is None or intr is None or cam_T_base is None:
        return None
    if intr.shape[-1] != 4 or cam_T_base.shape[-2:] != (4, 4):
        return None
    fx, fy, cx, cy = intr.tolist()
    if fx == 0 or fy == 0:
        return None
    base_to_cam = np.linalg.inv(cam_T_base)
    p_base_h = np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=np.float32)
    p_cam = base_to_cam @ p_base_h
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    # Calibration pixel coordinates (before resize/pad)
    u = fx * (p_cam[0] / z) + cx
    v = fy * (p_cam[1] / z) + cy
    # Derive calibration resolution from principal point
    Ht, Wt = int(out_hw[0]), int(out_hw[1])
    Wc = max(1.0, 2.0 * cx)
    Hc = max(1.0, 2.0 * cy)
    # Compute resized (letterboxed) dimensions identical to resize_with_pad
    ratio = max(Wc / Wt, Hc / Ht)
    resized_w = int(Wc / ratio)
    resized_h = int(Hc / ratio)
    pad_w0 = (Wt - resized_w) // 2
    pad_h0 = (Ht - resized_h) // 2
    # Scale and offset
    x = int(np.round(u * (resized_w / Wc) + pad_w0))
    y = int(np.round(v * (resized_h / Hc) + pad_h0))
    x = int(np.clip(x, 0, Wt - 1))
    y = int(np.clip(y, 0, Ht - 1))
    return x, y


def _draw_dot(img_u8: np.ndarray, xy: tuple[int, int] | None, color: tuple[int, int, int]) -> np.ndarray:
    out = img_u8.copy()
    if xy is None:
        return out
    x, y = xy
    H, W = out.shape[:2]
    rr = 4
    y0, y1 = max(0, y - rr), min(H, y + rr + 1)
    x0, x1 = max(0, x - rr), min(W, x + rr + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (yy - y) ** 2 + (xx - x) ** 2 <= rr * rr:
                out[yy, xx] = color
    return out


def _draw_line(
    img_u8: np.ndarray,
    xy0: tuple[int, int] | None,
    xy1: tuple[int, int] | None,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw a simple line between two points. Uses OpenCV if available, otherwise a fallback.

    Args:
        img_u8: Image array uint8 [H,W,3]
        xy0: Start point (x, y) or None
        xy1: End point (x, y) or None
        color: BGR color tuple
    """
    out = img_u8.copy()
    if xy0 is None or xy1 is None:
        return out
    try:
        import cv2

        cv2.line(out, xy0, xy1, color=color, thickness=2, lineType=cv2.LINE_AA)
        return out
    except Exception:
        pass

    # Fallback: simple DDA interpolation
    x0, y0 = xy0
    x1, y1 = xy1
    H, W = out.shape[:2]
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps <= 0:
        if 0 <= y0 < H and 0 <= x0 < W:
            out[y0, x0] = color
        return out
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + t * dx))
        y = int(round(y0 + t * dy))
        if 0 <= y < H and 0 <= x < W:
            out[y, x] = color
    return out


def eval_step(
    gt_batch: tuple[CoTObservation, _model.Actions],
    output_tokens: jax.Array,
    tok: PaligemmaCoTTokenizer,
    k_local: int,
):
    # Always run reasoning sampling across all processes; restrict decoding/logging to process 0.
    # Bound to local batch size to avoid indexing errors
    _, gt_texts = _decode_reasoning_strings(gt_batch[0], tok)
    # Decode sampled reasoning tokens
    ids = _utils.to_local_array(output_tokens)
    # Derive safe local loop bound across all sources
    k_decode = int(min(k_local, ids.shape[0], len(gt_texts)))
    pred_texts: list[str] = []
    for bi in range(k_decode):
        seq = ids[bi, :].astype(np.int32)
        pred_texts.append(tok.decode(seq))

    # # Compute L2 metric over parsed movement vectors (in cm)
    # for bi in range(k_decode):
    #     logging.info(f"GT text: {gt_texts[bi]}")
    #     logging.info(f"Pred text: {pred_texts[bi]}")

    return gt_texts, pred_texts


def vis_batch(batch, tok=None, step=None):
    """Visualize a training batch for debugging purposes.

    Args:
        batch: Tuple of (observation, actions)
        tok: Tokenizer for decoding tokenized prompts (optional)
        step: Training step number for wandb logging (optional)
    """
    obs = batch[0]
    actions = batch[1]

    logging.info("=" * 80)
    logging.info("BATCH VISUALIZATION")
    logging.info("=" * 80)

    # 1. Visualize images: print shape and log to wandb
    logging.info("\n--- IMAGES ---")
    wandb_images = {}
    for key, img in obs.images.items():
        logging.info(f"{key}: shape={img.shape}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")

        num_samples = img.shape[0]
        sample_images = []
        for t in range(min(num_samples, 4)):  # Log up to 4 samples
            sample_img = img[t]  # [H, W, C]

            # Convert from [-1, 1] to [0, 255]
            sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

            # Convert to numpy if it's a JAX array
            sample_img_uint8 = np.asarray(sample_img_uint8)

            # Add to wandb images list
            sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_t{t}"))
            logging.info(
                f"  Prepared image [{key}] timestep {t} for wandb "
                f"(range: [{sample_img_uint8.min()}, {sample_img_uint8.max()}])"
            )

        if sample_images:
            wandb_images[f"batch_vis/{key}"] = sample_images

    # 2. Visualize image_masks: print shape
    logging.info("\n--- IMAGE MASKS ---")
    for key, mask in obs.image_masks.items():
        logging.info(f"{key}: shape={mask.shape}, dtype={mask.dtype}, true_count={mask.sum()}/{mask.size}")

    # 3. Visualize state: print shape and min/max for each dimension
    logging.info("\n--- STATE ---")
    state = obs.state
    logging.info(f"state: shape={state.shape}, dtype={state.dtype}")
    if len(state.shape) >= 2:
        for dim_idx in range(state.shape[-1]):
            dim_data = state[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    # 4. Visualize tokenized_prompt with tokenizer
    logging.info("\n--- TOKENIZED PROMPTS ---")
    tokenized_prompt = obs.tokenized_prompt
    tokenized_prompt_mask = obs.tokenized_prompt_mask

    logging.info(f"tokenized_prompt: shape={tokenized_prompt.shape}, dtype={tokenized_prompt.dtype}")
    logging.info(f"tokenized_prompt_mask: shape={tokenized_prompt_mask.shape}, dtype={tokenized_prompt_mask.dtype}")
    # logging.info(f"token_ar_mask: shape={token_ar_mask.shape}, dtype={token_ar_mask.dtype}")
    # logging.info(f"token_loss_mask: shape={token_loss_mask.shape}, dtype={token_loss_mask.dtype}")

    if tok is not None:
        # Decode first sample in batch
        sample_idx = 0
        if tokenized_prompt.shape[0] > 0:
            # Full tokenized prompt
            tokens_full = tokenized_prompt[sample_idx]
            decoded_full = tok.decode(tokens_full)
            logging.info(f"\n[Sample {sample_idx}] Full tokenized_prompt:")
            logging.info(f"  Decoded: {decoded_full}")

            # Tokenized prompt with prompt mask applied
            tokens_masked = tokenized_prompt[sample_idx] * tokenized_prompt_mask[sample_idx]
            decoded_masked = tok.decode(tokens_masked)
            logging.info(f"\n[Sample {sample_idx}] tokenized_prompt * tokenized_prompt_mask:")
            logging.info(f"  Decoded: {decoded_masked}")

            # # Tokenized prompt with AR mask applied
            # tokens_ar = tokenized_prompt[sample_idx] * token_ar_mask[sample_idx]
            # decoded_ar = tok.decode(tokens_ar)
            # logging.info(f"\n[Sample {sample_idx}] tokenized_prompt * token_ar_mask:")
            # logging.info(f"  Decoded: {decoded_ar[:500]}...")
    else:
        logging.info("  (Tokenizer not provided - skipping decode)")

    # 5. Print token_loss_mask statistics
    # logging.info(f"\ntoken_loss_mask: sum={token_loss_mask.sum()}, mean={token_loss_mask.mean():.4f}")

    # 6. Visualize actions
    logging.info("\n--- ACTIONS ---")
    logging.info(f"actions: shape={actions.shape}, dtype={actions.dtype}")
    if len(actions.shape) >= 2:
        for dim_idx in range(actions.shape[-1]):
            dim_data = actions[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    logging.info("=" * 80)

    # Log images to wandb
    if wandb_images and jax.process_index() == 0 and step is not None:
        wandb.log(wandb_images, step=step)
        logging.info(f"Logged {len(wandb_images)} image groups to wandb")


def vis_augmented_images(
    augmented_images: dict[str, jax.Array] | None,
    step: int,
    prefix: str = "train",
    num_samples: int = 4,
) -> None:
    """Visualize augmented images from training step info.

    Args:
        augmented_images: Dictionary of augmented image arrays {key: [B, H, W, C]} in [-1, 1] range
        step: Training step number for wandb logging
        prefix: Prefix for wandb log keys (e.g., "train" or "val")
        num_samples: Number of samples to visualize per image key
    """
    if augmented_images is None:
        return

    if jax.process_index() != 0:
        return

    wandb_images = {}
    for key, img in augmented_images.items():
        img = _utils.to_local_array(img)
        if img is None or img.ndim < 4:
            continue

        num_to_show = min(img.shape[0], num_samples)
        sample_images = []
        for t in range(num_to_show):
            sample_img = img[t]  # [H, W, C]

            # Convert from [-1, 1] to [0, 255]
            sample_img = np.asarray(sample_img)
            sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

            # Add to wandb images list
            sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_sample{t}"))

        if sample_images:
            wandb_images[f"{prefix}/augmented_{key}"] = sample_images

    if wandb_images:
        wandb.log(wandb_images, step=step)
        logging.info(f"Logged {len(wandb_images)} augmented image groups to wandb at step {step}")
