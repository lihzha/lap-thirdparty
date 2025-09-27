from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
import logging
import re
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
import wandb

from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
from openpi_cot.training import utils as _utils

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
    _hard_example_keys: set[tuple[int, int]] = field(default_factory=set, init=False)

    def update(self, per_sample_losses: np.ndarray | None) -> None:
        if per_sample_losses is None:
            return
        arr = np.asarray(per_sample_losses, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._interval_losses.append(arr)
        self._interval_total_samples += int(arr.size)

    def add_hard_examples(
        self,
        step_idx: int,
        host_batch_local: tuple[CoTObservation, _model.Actions] | None,
        local_losses: np.ndarray | None,
        idx_offset: int,
    ) -> None:
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
            idx for idx in candidate_indices if (step_idx, int(idx_offset + idx)) not in self._hard_example_keys
        ]
        if not new_indices:
            return
        visuals = visualize_language_actions(
            host_batch_local,
            self.tokenizer,
            indices=new_indices,
            max_examples=len(new_indices),
            resize_hw=self.resize_hw,
        )
        if not visuals:
            return
        vis_by_index = {int(vis.get("index", idx)): vis for vis, idx in zip(visuals, new_indices)}
        for local_idx in new_indices:
            vis = vis_by_index.get(local_idx)
            if vis is None:
                continue
            loss_val = float(losses[local_idx])
            entry = {
                "loss": loss_val,
                "step": step_idx,
                "local_idx": int(local_idx),
                "global_idx": int(local_idx + idx_offset),
                "image": vis["image"],
                "language_action": vis.get("language_action", "") or "",
                "dataset_name": vis.get("dataset_name", "") or "",
                "prompt": vis.get("prompt", "") or "",
            }
            self._hard_example_buffer.append(entry)
            self._hard_example_keys.add((step_idx, entry["global_idx"]))
        self._hard_example_buffer.sort(key=lambda e: e["loss"], reverse=True)
        capacity = self._compute_buffer_capacity()
        if len(self._hard_example_buffer) > capacity:
            for removed in self._hard_example_buffer[capacity:]:
                self._hard_example_keys.discard((removed["step"], removed["global_idx"]))
            del self._hard_example_buffer[capacity:]

    def log_if_ready(self, step_idx: int) -> None:
        if not self._interval_losses:
            self.reset()
            return
        interval_all = np.concatenate(self._interval_losses, axis=0)
        total_samples = int(interval_all.size)
        hard_to_log = sorted(self._hard_example_buffer, key=lambda e: e["loss"], reverse=True)[: self.max_hard_examples]
        if total_samples == 0 or not hard_to_log:
            quantile_threshold = float("nan")
        else:
            target = min(self.max_hard_examples, total_samples)
            quantile_prob = 1.0 - target / total_samples
            quantile_prob = float(np.clip(quantile_prob, 0.0, 1.0))
            quantile_threshold = float(np.quantile(interval_all, quantile_prob))
            min_logged_loss = float(hard_to_log[-1]["loss"])
            if not np.isfinite(quantile_threshold) or quantile_threshold < min_logged_loss:
                quantile_threshold = min_logged_loss
        if hard_to_log:
            log_images = []
            for entry in hard_to_log:
                caption_text = entry.get("prompt", "") or ""
                caption_text += entry.get("language_action", "") or ""
                caption_text += entry.get("dataset_name", "") or ""
                caption = f"loss={entry['loss']:.4f}"
                panel_caption = caption
                log_images.append(
                    wandb.Image(
                        entry["image"],
                        caption=f"{panel_caption} | {caption_text}",
                    )
                )
            wandb_payload = {
                "train/hard_examples": log_images,
                "train/hard_examples_loss_quantile": quantile_threshold,
                "train/hard_examples_count": len(log_images),
            }
            wandb.log(wandb_payload, step=step_idx)
        self.reset()

    def reset(self) -> None:
        self._interval_losses.clear()
        self._interval_total_samples = 0
        self._hard_example_buffer.clear()
        self._hard_example_keys.clear()

    def _compute_buffer_capacity(self) -> int:
        # Always maintain capacity equal to the maximum number of hard examples to log.
        # This ensures we can always retain the top-k (descending by loss) across the interval.
        return int(self.max_hard_examples)


def _decode_reasoning_strings(obs: CoTObservation, tokenizer) -> list[str]:
    """Extract and decode the reasoning (language action) tokens per example.

    Returns one decoded string per example. If reasoning fields are absent, returns [].
    """
    tokens = _utils.to_local_array(obs.tokenized_prompt)
    rmask = _utils.to_local_array(obs.tokenized_reasoning_mask)
    reasonings: list[str] = []
    prompts: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        text = tokenizer.decode(sel.astype(np.int32))
        sel2 = tokens[i][(1 - rmask[i]).astype(bool)]
        prompt = tokenizer.decode(sel2.astype(np.int32))
        reasonings.append(text)
        prompts.append(prompt)
    return reasonings, prompts


def get_language_actions(batch, tok):
    texts, prompts = _decode_reasoning_strings(batch[0], tok)
    return texts, prompts


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

    texts, prompts = get_language_actions(batch, tok)

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
                frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
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
        prompt = prompts[idx] if idx < len(prompts) else ""
        visuals.append({"image": combined, "language_action": text, "index": idx, "prompt": prompt})

    return visuals


def log_random_examples(
    step: int,
    host_batch: tuple[CoTObservation, _model.Actions] | None,
    tokenizer: PaligemmaCoTTokenizer,
    *,
    local_batch_size: int,
    num_random: int = 5,
) -> None:
    if host_batch is None or local_batch_size <= 0:
        return
    count = min(num_random, local_batch_size)
    if count <= 0:
        return
    process_idx = getattr(jax, "process_index", lambda: 0)()
    rng_seed = int(step + 997 * process_idx)
    rng_local = np.random.default_rng(rng_seed)
    rand_idx = rng_local.choice(local_batch_size, size=count, replace=False)
    random_visuals = visualize_language_actions(
        host_batch,
        tokenizer,
        indices=rand_idx.tolist(),
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
        wandb.log({"train/random_examples": images_to_log}, step=step)


def prepare_eval_batch(batch):
    # Process the batch to remove reasoning and update masks
    obs, actions = batch

    # Find position 108 (start of reasoning) in each batch item
    batch_size = obs.tokenized_prompt.shape[0]
    new_tokenized_prompts = []
    new_tokenized_prompt_masks = []
    new_tokenized_reasoning_masks = []

    for i in range(batch_size):
        prompt_tokens = obs.tokenized_prompt[i]

        # Find position of token 108 (start of reasoning)
        # Ensure prompt_tokens is int32 for the comparison
        prompt_tokens_int32 = prompt_tokens.astype(jnp.int32)
        pos_108 = jnp.where(prompt_tokens_int32 == 108, size=1, fill_value=-1)[0]

        if pos_108[0] >= 0:
            # Remove everything after token 108 (inclusive)
            prompt_without_reasoning = prompt_tokens[: pos_108[0] + 1]
            original_length = prompt_tokens.shape[0]

            # Left pad to maintain the same length
            padding_length = original_length - prompt_without_reasoning.shape[0]
            # Ensure consistent dtype for concatenation
            padding_zeros = jnp.zeros(padding_length, dtype=prompt_tokens.dtype)
            prompt_without_reasoning = prompt_without_reasoning.astype(prompt_tokens.dtype)
            padded_prompt = jnp.concatenate([padding_zeros, prompt_without_reasoning])

            # Create new mask: True for non-zero tokens, False for padding
            new_mask = (padded_prompt != 0).astype(jnp.bool_)

            # Create reasoning mask: all False - ensure consistent dtype
            reasoning_mask = jnp.zeros(original_length, dtype=jnp.bool_)

        else:
            # No token 108 found, keep original
            padded_prompt = prompt_tokens
            # Ensure consistent dtype for the mask
            if obs.tokenized_prompt_mask is not None:
                new_mask = obs.tokenized_prompt_mask[i].astype(jnp.bool_)
            else:
                # Create a boolean mask of the same length as prompt_tokens
                new_mask = jnp.ones(prompt_tokens.shape[0], dtype=jnp.bool_)
            # Create reasoning mask with consistent dtype - use original length instead of zeros_like
            reasoning_mask = jnp.zeros(prompt_tokens.shape[0], dtype=jnp.bool_)

            logging.info(f"Batch {i}: No token 108 found, keeping original prompt")

        new_tokenized_prompts.append(padded_prompt)
        new_tokenized_prompt_masks.append(new_mask)
        new_tokenized_reasoning_masks.append(reasoning_mask)

    # Ensure all tensors have consistent types before stacking
    # All masks should be boolean, all prompts should be int32
    new_tokenized_prompts = [p.astype(jnp.int32) for p in new_tokenized_prompts]
    new_tokenized_prompt_masks = [m.astype(jnp.bool_) for m in new_tokenized_prompt_masks]
    new_tokenized_reasoning_masks = [r.astype(jnp.bool_) for r in new_tokenized_reasoning_masks]

    # Stack the processed tensors
    new_tokenized_prompt = jnp.stack(new_tokenized_prompts)
    new_tokenized_prompt_mask = jnp.stack(new_tokenized_prompt_masks)
    new_tokenized_reasoning_mask = jnp.stack(new_tokenized_reasoning_masks)

    # Create new observation with modified prompts and masks
    new_obs = CoTObservation(
        images=obs.images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=new_tokenized_prompt,
        tokenized_prompt_mask=new_tokenized_prompt_mask,
        tokenized_reasoning_mask=new_tokenized_reasoning_mask,
        token_ar_mask=obs.token_ar_mask,
        token_loss_mask=obs.token_loss_mask,
        example_mask=obs.example_mask,
    )

    # Create new batch with modified observation
    new_batch = (new_obs, actions)
    return new_batch


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
    id_buf: jax.Array,
    t_final: jax.Array,
    tok: PaligemmaCoTTokenizer,
    k_local: int,
):
    l2_cm_values: list[float] = []
    to_log: list[np.ndarray] = []
    # Always run reasoning sampling across all processes; restrict decoding/logging to process 0.
    # Bound to local batch size to avoid indexing errors
    if jax.process_index() == 0:
        gt_texts, _ = _decode_reasoning_strings(gt_batch[0], tok)
        # Decode sampled reasoning tokens
        ids = _utils.to_local_array(id_buf)
        # Be robust to bounds: clamp final index
        t_host = int(np.clip(_utils.to_local_scalar(t_final), 0, ids.shape[1] - 1))
        # Prepare images now to compute consistent local count
        first_cam_key = next(iter(gt_batch[0].images))
        imgs = _utils.to_local_array(gt_batch[0].images[first_cam_key])
        imgs_u8 = ((np.asarray(imgs) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        # Derive safe local loop bound across all sources
        k_decode = int(min(k_local, ids.shape[0], imgs_u8.shape[0], len(gt_texts)))
        pred_texts: list[str] = []
        for bi in range(k_decode):
            seq = ids[bi, : t_host + 1, 0].astype(np.int32)
            pred_texts.append(tok.decode(seq))

        # Compute L2 metric over parsed movement vectors (in cm)
        for bi in range(k_decode):
            logging.info(f"GT text: {gt_texts[bi]}")
            logging.info(f"Pred text: {pred_texts[bi]}")
            gt_vec = _parse_language_delta_cm(gt_texts[bi])
            pred_vec = _parse_language_delta_cm(pred_texts[bi])
            if pred_vec is None:
                continue
            l2_cm = float(np.linalg.norm(gt_vec - pred_vec))
            l2_cm_values.append(l2_cm)

        if not l2_cm_values:
            return None, None

        # Prepare annotated images for a subset
        # Choose a camera to display
        # Optional 3D->2D projection inputs
        cart = gt_batch[0].cartesian_position_window
        intr_all = gt_batch[0].camera_intrinsics
        extr_all = gt_batch[0].camera_extrinsics
        cart_np = _utils.to_local_array(cart)
        intr_np = _utils.to_local_array(intr_all)
        extr_np = _utils.to_local_array(extr_all)
        if cart_np is None or intr_np is None or extr_np is None:
            logging.info("No extrinsics/intrinsics/cartesian position available. Try vis_dataset=True.")
            return None, None
        for bi in range(k_decode):
            vis = imgs_u8[bi]
            H, W = vis.shape[:2]
            if cart_np.shape[1] >= 1:
                # [T,6]
                seq = np.asarray(cart_np[bi])
                if seq.ndim == 2 and seq.shape[-1] >= 3:
                    start_xyz = seq[0, :3]
                    end_xyz = seq[-1, :3]
            ci = np.asarray(intr_np[bi])
            intr = ci[0] if ci.ndim == 2 else ci
            ce = np.asarray(extr_np[bi])
            extr = ce[0] if ce.ndim == 3 else ce
            start_xy = _project_point(start_xyz, extr, intr, (H, W))
            end_true_xy = _project_point(end_xyz, extr, intr, (H, W))
            # Predicted end via language delta
            v_cm = _parse_language_delta_cm(pred_texts[bi])
            if v_cm is None:
                continue
            t_cam = _invert_camera_axis_map(v_cm)
            R_cb = extr[:3, :3]
            t_base = R_cb @ t_cam
            pred_xyz = start_xyz + t_base
            pred_end_xy = _project_point(pred_xyz, extr, intr, (H, W))
            # Draw dots
            vis2 = vis
            vis2 = _draw_dot(vis2, start_xy, (0, 255, 255))  # GT start (yellow)
            vis2 = _draw_dot(vis2, pred_end_xy, (0, 0, 255))  # Pred end (red)
            vis2 = _draw_dot(vis2, end_true_xy, (0, 255, 0))  # GT end (green)
            vis2 = _draw_line(vis2, start_xy, end_true_xy, (0, 255, 0))
            vis2 = _draw_line(vis2, start_xy, pred_end_xy, (0, 0, 255))
            to_log.append(vis2)
    return l2_cm_values, to_log
