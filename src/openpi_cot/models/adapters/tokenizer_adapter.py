import logging
import re

import numpy as np
from openpi.models import tokenizer as _tokenizer


class PaligemmaCoTTokenizer(_tokenizer.PaligemmaTokenizer):
    def __init__(
        self,
        max_len: int = 48,
        left_pad: bool = True,
        include_decimal_point: bool = True,
    ):
        super().__init__(max_len)
        self._left_pad = left_pad
        self._include_decimal_point = include_decimal_point
        self._stop_token_id = self._tokenizer.eos_id()

    def tokenize_cot(
        self, prompt: str, reasoning: str | None = None, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            cleaned_prompt = f"Task: {cleaned_prompt}, State: {state_str};\nAction: "
        # eos_id = self._tokenizer.eos_id()
        pad_id = self._tokenizer.pad_id()

        tokens = self._tokenizer.encode(cleaned_prompt, add_bos=True, add_eos=False)
        if (
            state is None
        ):  # This is the Pi0 format, where the state is part of the continuous action expert input.
            tokens += self._tokenizer.encode("\n")

        reasoning_start = len(tokens)
        if reasoning is not None:
            clean_reason = reasoning.strip().replace("_", " ").replace("\n", " ")
            tokens += self._tokenizer.encode(clean_reason, add_bos=False, add_eos=True)
        reasoning_end = len(tokens)

        if len(tokens) > self._max_len:
            logging.warning(
                f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently."
            )
            tokens = tokens[: self._max_len]
            reasoning_end = min(reasoning_end, self._max_len)

        attn_mask = np.zeros(self._max_len, dtype=bool)
        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        numeric_mask = np.zeros(self._max_len, dtype=bool)

        if self._left_pad:
            # Left pad to max length for generation/training
            pad_count = self._max_len - len(tokens)
            if pad_count > 0:
                tokens = [pad_id] * pad_count + tokens

            attn_mask = np.zeros(self._max_len, dtype=bool)
            reasoning_mask = np.zeros(self._max_len, dtype=bool)
            numeric_mask = np.zeros(self._max_len, dtype=bool)
            attn_mask[pad_count:] = True
            # Shift reasoning indices by pad_count after left padding
            start_idx = max(0, min(self._max_len, reasoning_start + pad_count))
            end_idx = max(0, min(self._max_len, reasoning_end + pad_count))
            if end_idx > start_idx:
                reasoning_mask[start_idx:end_idx] = True
            # Build numeric mask: mark tokens that contain digits within reasoning span only
            try:
                pieces = [self._tokenizer.id_to_piece(t) for t in tokens]
            except Exception:
                pieces = [""] * len(tokens)

            def _has_digit(p: str) -> bool:
                return bool(re.search(r"[0-9]", p))

            def _is_decimal_point_index(i: int) -> bool:
                if not self._include_decimal_point:
                    return False
                p = pieces[i]
                if "." not in p:
                    return False
                prev_has = i - 1 >= 0 and _has_digit(pieces[i - 1])
                next_has = i + 1 < len(pieces) and _has_digit(pieces[i + 1])
                return prev_has or next_has

            for i in range(start_idx, end_idx):
                if i < 0 or i >= len(pieces):
                    continue
                if _has_digit(pieces[i]) or _is_decimal_point_index(i):
                    numeric_mask[i] = True
        else:
            attn_mask[: len(tokens)] = True
            reasoning_mask[reasoning_start:reasoning_end] = True
            tokens += [pad_id] * (self._max_len - len(tokens))
            # Build numeric mask without left padding
            try:
                pieces = [
                    self._tokenizer.id_to_piece(t) for t in tokens[:reasoning_end]
                ]
            except Exception:
                pieces = [""] * len(tokens[:reasoning_end])

            def _has_digit(p: str) -> bool:
                return bool(re.search(r"[0-9]", p))

            def _is_decimal_point_index(i: int) -> bool:
                if not self._include_decimal_point:
                    return False
                p = pieces[i]
                if "." not in p:
                    return False
                prev_has = i - 1 >= 0 and _has_digit(pieces[i - 1])
                next_has = i + 1 < len(pieces) and _has_digit(pieces[i + 1])
                return prev_has or next_has

            for i in range(reasoning_start, reasoning_end):
                idx = i
                if idx < len(pieces) and (
                    _has_digit(pieces[idx]) or _is_decimal_point_index(idx)
                ):
                    numeric_mask[i] = True

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            numeric_mask,
        )

    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens back to a string."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens)
