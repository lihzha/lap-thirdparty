import logging
import re

import numpy as np
from openpi.models import tokenizer as _tokenizer


class PaligemmaCoTTokenizer(_tokenizer.PaligemmaTokenizer):
    def __init__(
        self,
        max_len: int = 48,
        use_pi05_prompt_format: bool = True,
    ):
        super().__init__(max_len)
        self._stop_token_id = self._tokenizer.eos_id()
        self._use_pi05_prompt_format = use_pi05_prompt_format

    def tokenize_cot(
        self, prompt: str, reasoning: str | None = None, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            assert self._use_pi05_prompt_format, "State should only be provided when using Pi05 format."
            # This is the Pi05 format, where the state is part of the discrete language input.
            # State vectors are padded with trailing zeros to action_dim in the dataset pipeline.
            # Only include up to the last unpadded (non-zero) dimension when discretizing.
            state_arr = np.asarray(state)
            # Compute last non-zero column along the final axis (robust to 1D or ND inputs)
            eps = 1e-8
            if state_arr.ndim == 1:
                non_zero_mask = np.abs(state_arr) > eps
                last_idx = int(np.nonzero(non_zero_mask)[0][-1]) + 1 if np.any(non_zero_mask) else 0
                last_idx = max(last_idx, 7)  # 7 is the smallest number of dimensions that is not padded
                trimmed = state_arr[:last_idx]
            else:
                flat = state_arr.reshape(-1, state_arr.shape[-1])
                non_zero_cols = np.any(np.abs(flat) > eps, axis=0)
                last_idx = int(np.nonzero(non_zero_cols)[0][-1]) + 1 if np.any(non_zero_cols) else 0
                last_idx = max(last_idx, 7)
                trimmed = state_arr[..., :last_idx].reshape(-1)

            if trimmed.size > 0:
                discretized_state = np.digitize(trimmed, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
            else:
                state_str = ""
            cleaned_prompt = f"Task: {cleaned_prompt}, State: {state_str};\nAction: "
        elif self._use_pi05_prompt_format:
            cleaned_prompt = f"Task: {cleaned_prompt};\nAction: "

        # eos_id = self._tokenizer.eos_id()
        pad_id = self._tokenizer.pad_id()

        tokens = self._tokenizer.encode(cleaned_prompt, add_bos=True, add_eos=False)
        if state is None:  # This is the Pi0 format, where the state is part of the continuous action expert input.
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
        pieces = [self._tokenizer.id_to_piece(t) for t in tokens]

        def _has_digit(p: str) -> bool:
            return bool(re.search(r"[0-9]", p))

        for i in range(start_idx, end_idx):
            if i < 0 or i >= len(pieces):
                continue
            if _has_digit(pieces[i]):
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

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
