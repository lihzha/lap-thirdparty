import dataclasses
import re
from typing import TypeAlias

import numpy as np
from openpi.shared import array_typing as at
from openpi.transforms import DataTransformFn
from openpi.transforms import flatten_dict
from openpi.transforms import unflatten_dict

from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer

DataDict: TypeAlias = at.PyTree


@dataclasses.dataclass(frozen=True)
class TokenizePromptAndReasoning(DataTransformFn):
    tokenizer: PaligemmaCoTTokenizer
    discrete_state_input: bool = False

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
                try:
                    val = float(m.group(2))
                except Exception:
                    continue
                if abs(val) > 1e-6:
                    any_nonzero = True
                    break
            return not any_nonzero

        is_idle = _is_idle_language_action(language_actions)
        example_mask = not is_idle

        tokens, pad_mask, reasoning_mask, numeric_mask = self.tokenizer.tokenize_cot(prompt, language_actions, state)

        return {
            **data,
            "tokenized_prompt": tokens,  # kept for compatibility with upstream
            "tokenized_prompt_mask": pad_mask,  # kept for compatibility with upstream
            "tokenized_reasoning_mask": reasoning_mask,
            "tokenized_numeric_mask": numeric_mask,
            # Expose example-level mask so loaders/models can skip or mask (True = keep, False = idle)
            "example_mask": np.asarray(example_mask, dtype=bool),
        }


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
