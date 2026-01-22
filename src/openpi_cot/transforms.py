import dataclasses
from typing import TypeAlias

import numpy as np
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
from openpi.transforms import DataTransformFn
from openpi.transforms import _assert_quantile_stats
from openpi.transforms import apply_tree
from openpi.transforms import flatten_dict
from openpi.transforms import unflatten_dict

from openpi_cot.models.tokenizer import FASTTokenizer
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer

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
    dataset_name_pad_len: int = 100
    verbose_mode: bool = False
    state_dropout: float = 0.0

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
        dataset_name = data.pop("dataset_name", None)  # if None, inference
        frame_description = data.pop("frame_description", "robot base frame")
        if dataset_name is not None:
            tokenized_dataset_name = self.tokenizer._tokenizer.encode(dataset_name)
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * (
                self.dataset_name_pad_len - len(tokenized_dataset_name)
            ) + tokenized_dataset_name
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)
        else:
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * self.dataset_name_pad_len
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)


        # Tokenize regular reasoning
        (
            tokens,
            pad_mask,
            reasoning_mask,
            numeric_mask,
            direction_mask,
            token_loss_mask,
        ) = self.tokenizer.tokenize_cot(
            prompt,
            language_actions,
            state,
            frame_description=frame_description,
            state_dropout=self.state_dropout,
        )

        # Combine number_mask and direction_mask for critical tokens
        critical_mask = np.logical_or(numeric_mask, direction_mask)

        result = {
            **data,
            "tokenized_prompt": tokens,  # kept for compatibility with upstream
            "tokenized_prompt_mask": pad_mask,  # kept for compatibility with upstream
            "tokenized_langact_mask": reasoning_mask,
            "token_loss_mask": token_loss_mask,
            "tokenized_dataset_name": tokenized_dataset_name,
            
        }

        if self.verbose_mode:
            result.update(
                {  # Critical tokens are both numbers AND directional indicators
                    "critical_token_mask": critical_mask,
                    "number_token_mask": numeric_mask,
                    "direction_token_mask": direction_mask,
                }
            )

        return result


@dataclasses.dataclass(frozen=True)
class DetokenizeReasoning(DataTransformFn):
    tokenizer: PaligemmaCoTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if "tokens" in data:
            text = self.tokenizer.decode(data["tokens"].squeeze().astype(np.int32))
            return {**data, "reasoning": text}
        return data


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
    # Normalization type to use (NORMAL, BOUNDS, or BOUNDS_Q99)
    normalization_type: str = "normal"
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False


    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        normalization_type = self.normalization_type
        if normalization_type == "normal":
            normalize_fn = self._normalize
        elif normalization_type == "bounds":
            normalize_fn = self._normalize_bounds
        elif normalization_type == "bounds_q99":
            normalize_fn = self._normalize_quantile
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        return apply_tree(
            data,
            self.norm_stats,
            normalize_fn,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_bounds(self, x, stats: NormStats):
        assert stats.min is not None
        assert stats.max is not None
        min_val, max_val = stats.min[..., : x.shape[-1]], stats.max[..., : x.shape[-1]]
        # Scale to [-1, 1] and clip
        scaled = 2.0 * (x - min_val) / (max_val - min_val + 1e-8) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)
        # Zero-out dimensions with zero range
        zeros_mask = np.equal(min_val, max_val)
        while zeros_mask.ndim < x.ndim:
            zeros_mask = zeros_mask[None, ...]
        return np.where(zeros_mask, 0.0, scaled)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        # Scale to [-1, 1] and clip
        scaled = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        # scaled = np.clip(scaled, -1.0, 1.0)
        # Zero-out dimensions with zero range
        zeros_mask = np.equal(q01, q99)
        while zeros_mask.ndim < x.ndim:
            zeros_mask = zeros_mask[None, ...]
        return np.where(zeros_mask, 0.0, scaled)


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # Normalization type to use (NORMAL, BOUNDS, or BOUNDS_Q99)
    normalization_type: str = "normal"

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        normalization_type = self.normalization_type

        if normalization_type == "normal":
            unnormalize_fn = self._unnormalize
        elif normalization_type == "bounds":
            unnormalize_fn = self._unnormalize_bounds
        elif normalization_type == "bounds_q99":
            unnormalize_fn = self._unnormalize_quantile
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            unnormalize_fn,
            strict=False,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_bounds(self, x, stats: NormStats):
        assert stats.min is not None
        assert stats.max is not None
        min_val = pad_to_dim(stats.min, x.shape[-1], axis=-1, value=-1.0)
        max_val = pad_to_dim(stats.max, x.shape[-1], axis=-1, value=1.0)
        return (x + 1.0) / 2.0 * (max_val - min_val + 1e-8) + min_val

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class PadStates(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTCoTInputs(DataTransformFn):
    """Tokenize inputs for FAST CoT model.

    This combines CoT-style reasoning with FAST-style action token prediction:
    - Tokenizes prompt + language actions (if available)
    - Appends state and action representations as tokens
    - Creates appropriate masks for training

    Similar to upstream TokenizeFASTInputs but with support for language actions.
    For VQA and prediction samples, includes the language_actions (answer) tokens
    in the loss mask so they contribute to training.
    """

    tokenizer: FASTTokenizer
    discrete_state_input: bool = True
    state_dropout: float = 0.0
    dataset_name_pad_len: int = 100

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = data.get("state")
            if state is None:
                raise ValueError("State is required for FAST tokenization.")

        frame_description = data.pop("frame_description", "robot base frame")

        # Get language_actions (contains VQA answer or prediction answer)
        # Similar to TokenizePromptAndReasoning pattern
        language_actions = data.pop("language_actions", None)
        
        dataset_name = data.pop("dataset_name", None)  # if None, inference
        if dataset_name is not None:
            tokenized_dataset_name = self.tokenizer._tokenizer.encode(dataset_name)
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * (
                self.dataset_name_pad_len - len(tokenized_dataset_name)
            ) + tokenized_dataset_name
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)
        else:
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * self.dataset_name_pad_len
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)

        # Get state type if available
        state_type = data.pop("state_type", None)
        if state_type is not None and not isinstance(state_type, str):
            state_type = state_type.item() if hasattr(state_type, "item") else str(state_type)

        # Get actions (None during inference)
        actions = data.get("actions")

        # Tokenize using the FAST CoT tokenizer
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize_fast_cot(
            prompt=prompt,
            state=state,
            actions=actions,
            language_actions=language_actions,
            state_type=state_type,
            state_dropout=self.state_dropout,
            frame_description=frame_description,
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "tokenized_langact_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "tokenized_dataset_name": tokenized_dataset_name,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "tokens" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("tokens")
        original_actions = data.pop("actions", None)
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    x = x[..., :target_dim]  # Truncate if necessary
    return x
