import dataclasses
import re
from typing import TypeAlias

import numpy as np
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
from openpi.transforms import DataTransformFn
from openpi.transforms import _assert_quantile_stats
from openpi.transforms import apply_tree
from openpi.transforms import flatten_dict
from openpi.transforms import unflatten_dict

from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer

# Optional TF import: used to ensure ops run inside tf.data pipelines
try:  # pragma: no cover - optional dependency in some environments
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None

DataDict: TypeAlias = at.PyTree


@dataclasses.dataclass(frozen=True)
class TokenizePromptAndReasoning(DataTransformFn):
    tokenizer: PaligemmaCoTTokenizer
    discrete_state_input: bool = False
    prediction_prompt: str = "What is the robot's movement between two frames?"

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        # Always tokenize regular reasoning (prompt + language_actions)
        language_actions = data.pop("language_actions", None)  # if None, inference
        if language_actions is not None and not isinstance(language_actions, str):
            language_actions = language_actions.item()

        # Idle check: mark examples whose summed language action is effectively zero on all axes
        def _is_idle_language_action(s: str | None) -> bool:
            if s is None:
                return False
            s = s.strip()
            if s == "":
                return True
            # Robust parse: accept patterns like "move forward 0.00 cm" joined by " and "
            parts = [p.strip() for p in s.split(" and ") if p.strip()]
            if not parts:
                return True
            any_nonzero = False
            for p in parts:
                m = re.match(r"move\s+(\w+)\s+([-+]?\d*\.?\d+)\s*(\w+)", p)
                if not m:
                    continue
                val = float(m.group(2))
                if abs(val) > 1e-6:
                    any_nonzero = True
                    break
            return not any_nonzero

        is_idle = _is_idle_language_action(language_actions)
        sample_mask = not is_idle

        # Tokenize regular reasoning
        tokens, pad_mask, reasoning_mask, numeric_mask = self.tokenizer.tokenize_cot(prompt, language_actions, state)

        result = {
            **data,
            "tokenized_prompt": tokens,  # kept for compatibility with upstream
            "tokenized_prompt_mask": pad_mask,  # kept for compatibility with upstream
            "tokenized_langact_mask": reasoning_mask,
            # Expose example-level mask so loaders/models can skip or mask (True = keep, False = idle)
            "crictical_token_mask": numeric_mask,
            "sample_mask": np.asarray(sample_mask, dtype=bool),
        }

        # Additionally tokenize prediction if prediction_language_action is present
        prediction_lang = data.pop("prediction_language_action", None)
        prediction_prompt_str = data.pop("prediction_prompt", self.prediction_prompt)
        if prediction_lang is not None:
            # Parse prediction language action
            if isinstance(prediction_lang, bytes):
                prediction_lang = prediction_lang.decode("utf-8")
            elif not isinstance(prediction_lang, str):
                prediction_lang = (
                    str(prediction_lang) if hasattr(prediction_lang, "__str__") else prediction_lang.item()
                )

            # Skip empty prediction language actions
            if prediction_lang and prediction_lang.strip():
                # Use prediction-specific tokenization
                pred_tokens, pred_pad_mask, pred_reasoning_mask, pred_numeric_mask = self.tokenizer.tokenize_prediction(
                    prediction_prompt_str, prediction_lang
                )

                # Add prediction-specific fields
                result["tokenized_prediction"] = pred_tokens
                result["tokenized_prediction_mask"] = pred_pad_mask
                result["tokenized_prediction_langact_mask"] = pred_reasoning_mask
                result["prediction_crictical_token_mask"] = pred_numeric_mask

        return result


@dataclasses.dataclass(frozen=True)
class DetokenizeReasoning(DataTransformFn):
    tokenizer: PaligemmaCoTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        text = self.tokenizer.decode(data["reasoning_logits"].squeeze()[: data["final_length"]].astype(np.int32))
        return {**data, "reasoning": text}


@dataclasses.dataclass(frozen=True)
class SafeRepackTransform:
    structure: at.PyTree
    strict: bool = False  # when True, error if any source path is missing

    def __call__(self, data: DataDict) -> DataDict:
        flat_data = flatten_dict(data)
        flat_struct = flatten_dict(self.structure)  # maps out_key -> source_path or list of candidates
        out = {}
        missing = []
        for out_key, src_spec in flat_struct.items():
            # Allow a single source path or a list/tuple of fallback source paths.
            candidates = src_spec if isinstance(src_spec, (list, tuple)) else [src_spec]
            found = False
            for src_path in candidates:
                if src_path in flat_data:
                    out[out_key] = flat_data[src_path]
                    found = True
                    break
            if not found:
                missing.append((out_key, tuple(candidates)))
        if self.strict and missing:
            raise KeyError(f"Missing source paths: {missing}")
        return unflatten_dict(out)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class NormalizeActionAndProprio(DataTransformFn):
    """Normalize `action` and `proprio`-like fields using dataset statistics.

    This class adapts the behavior of `normalize_action_and_proprio` from
    `openpi_cot.dataloader.oxe_utils.data_utils` into a DataTransformFn so it can
    be used directly in dataset pipelines.

    It supports both NumPy arrays and TensorFlow tensors. When TensorFlow tensors
    are encountered, TensorFlow math/clip/where ops are used to preserve graph
    execution; otherwise NumPy is used.
    """

    norm_stats: at.PyTree | None
    normalization_type: NormalizationType | str = NormalizationType.NORMAL
    action_key: str = "action"
    state_key: str = "proprio"

    def __call__(self, traj: DataDict) -> DataDict:
        if self.norm_stats is None:
            return traj

        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        def _is_tf_tensor(x):
            return tf is not None and isinstance(x, tf.Tensor)

        def _to_tensor(value, like):
            if value is None:
                return None
            if _is_tf_tensor(like):
                if _is_tf_tensor(value):
                    return tf.cast(value, tf.float32)
                return tf.convert_to_tensor(value, dtype=tf.float32)
            # NumPy path
            if isinstance(value, np.ndarray):
                return value.astype(np.float32, copy=False)
            return np.asarray(value, dtype=np.float32)

        def _clip(x, lo, hi):
            if _is_tf_tensor(x):
                return tf.clip_by_value(x, lo, hi)
            return np.clip(x, lo, hi)

        def _where(mask, a, b):
            if _is_tf_tensor(a) or _is_tf_tensor(b):
                return tf.where(mask, a, b)
            return np.where(mask, a, b)

        def _equal(a, b):
            if _is_tf_tensor(a) or _is_tf_tensor(b):
                return tf.equal(a, b)
            return np.equal(a, b)

        def _get_group(stats_root, group_name: str):
            if isinstance(stats_root, dict):
                group = stats_root.get(group_name)
                if group is None and group_name.endswith("s"):
                    group = stats_root.get(group_name[:-1])
                return group
            return None

        def _get_value(group_stats, key: str, like_arr):
            if group_stats is None:
                return None
            if isinstance(group_stats, dict):
                value = group_stats.get(key)
            else:
                value = getattr(group_stats, key, None)
            if value is None:
                return None
            return _to_tensor(value, like_arr)

        # Resolve stats groups
        actions_stats = _get_group(self.norm_stats, "actions")
        state_stats = _get_group(self.norm_stats, "state")

        # Fetch action/state tensors from the trajectory and harmonize dtypes
        action_arr = traj[self.action_key]
        state_container = traj.get("observation", {})
        state_arr = state_container.get(self.state_key)

        # Ensure numeric dtype consistency (float32) before arithmetic to avoid TF float64 vs float32 issues
        if _is_tf_tensor(action_arr):
            action_arr = tf.cast(action_arr, tf.float32)
        else:
            action_arr = np.asarray(action_arr, dtype=np.float32)

        if state_arr is not None:
            if _is_tf_tensor(state_arr):
                state_arr = tf.cast(state_arr, tf.float32)
            else:
                state_arr = np.asarray(state_arr, dtype=np.float32)
            # Update the container to keep downstream consistency
            traj.setdefault("observation", {})[self.state_key] = state_arr
        # Update the action in traj to the standardized dtype pre-normalization
        traj[self.action_key] = action_arr

        if normalization_type == NormalizationType.NORMAL:
            a_mean = _get_value(actions_stats, "mean", action_arr)
            a_std = _get_value(actions_stats, "std", action_arr)
            s_mean = _get_value(state_stats, "mean", state_arr) if state_arr is not None else None
            s_std = _get_value(state_stats, "std", state_arr) if state_arr is not None else None

            if a_mean is not None and a_std is not None:
                if _is_tf_tensor(action_arr):
                    traj[self.action_key] = (action_arr - a_mean) / (a_std + 1e-6)
                else:
                    traj[self.action_key] = (action_arr - a_mean) / (a_std + 1e-6)

            if state_arr is not None and s_mean is not None and s_std is not None:
                normed = (state_arr - s_mean) / (s_std + 1e-6)
                traj["observation"][self.state_key] = normed

        elif normalization_type in (NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99):
            low_key = "min" if normalization_type == NormalizationType.BOUNDS else "q01"
            high_key = "max" if normalization_type == NormalizationType.BOUNDS else "q99"

            a_low = _get_value(actions_stats, low_key, action_arr)
            a_high = _get_value(actions_stats, high_key, action_arr)
            s_low = _get_value(state_stats, low_key, state_arr) if state_arr is not None else None
            s_high = _get_value(state_stats, high_key, state_arr) if state_arr is not None else None

            if a_low is not None and a_high is not None:
                # scale to [-1, 1] and clip
                scaled = 2.0 * (action_arr - a_low) / (a_high - a_low + (1e-8)) - 1.0
                traj[self.action_key] = _clip(scaled, -1.0, 1.0)
                # zero-out dimensions with zero range
                zeros_mask = _equal(a_low, a_high)  # shape [..., D]
                # Broadcast mask over leading dims of action tensor
                if _is_tf_tensor(action_arr):
                    while len(zeros_mask.shape) < len(action_arr.shape):
                        zeros_mask = zeros_mask[None, ...]
                else:
                    while zeros_mask.ndim < action_arr.ndim:
                        zeros_mask = zeros_mask[None, ...]
                traj[self.action_key] = _where(zeros_mask, 0.0, traj[self.action_key])

            if state_arr is not None and s_low is not None and s_high is not None:
                scaled = 2.0 * (state_arr - s_low) / (s_high - s_low + (1e-8)) - 1.0
                state_normed = _clip(scaled, -1.0, 1.0)
                zeros_mask = _equal(s_low, s_high)
                if _is_tf_tensor(state_normed):
                    while len(zeros_mask.shape) < len(state_normed.shape):
                        zeros_mask = zeros_mask[None, ...]
                else:
                    while zeros_mask.ndim < state_normed.ndim:
                        zeros_mask = zeros_mask[None, ...]
                traj["observation"][self.state_key] = _where(zeros_mask, 0.0, state_normed)

        return traj
