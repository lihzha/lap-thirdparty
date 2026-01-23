import logging
from typing import Literal

import numpy as np
from openpi.models import tokenizer as _tokenizer
import sentencepiece
from transformers import AutoProcessor

from openpi_cot.models.prompt_utils.checkers import is_number
from openpi_cot.models.prompt_utils.prompt import DEFAULT_VQA_PROMPT_FORMAT
from openpi_cot.models.prompt_utils.prompt import PREDICTION_PROMPT_FORMAT_REGISTRY
from openpi_cot.models.prompt_utils.prompt import PROMPT_FORMAT_REGISTRY
from openpi_cot.models.prompt_utils.prompt import PromptFormat
import openpi_cot.shared.download as download


class PaligemmaCoTTokenizer(_tokenizer.PaligemmaTokenizer):
    def __init__(
        self,
        max_len: int = 48,
        prompt_format: Literal[
            "pi05",
            "pi0",
            "pi05_notime",
        ]
        | PromptFormat = "pi05",
        prediction_format: Literal["default", "grouped"] | PromptFormat = "default",
        reasoning_mask_prob: float = 0.0,
    ):
        # super().__init__(max_len)
        self.reasoning_mask_prob = reasoning_mask_prob
        logging.info(f"Use reasoning_mask_prob: {self.reasoning_mask_prob}")
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._max_len = max_len

        # Support both string and PromptFormat instance
        if isinstance(prompt_format, str):
            if prompt_format not in PROMPT_FORMAT_REGISTRY:
                raise ValueError(
                    f"Unknown prompt format: {prompt_format}. Available formats: {list(PROMPT_FORMAT_REGISTRY.keys())}"
                )
            self._prompt_format = PROMPT_FORMAT_REGISTRY[prompt_format]
        else:
            self._prompt_format = prompt_format

        # Support both string and PromptFormat instance
        if isinstance(prediction_format, str):
            if prediction_format not in PREDICTION_PROMPT_FORMAT_REGISTRY:
                raise ValueError(
                    f"Unknown prediction format: {prediction_format}. Available formats: {list(PREDICTION_PROMPT_FORMAT_REGISTRY.keys())}"
                )
            self._prediction_format = PREDICTION_PROMPT_FORMAT_REGISTRY[prediction_format]
        else:
            self._prediction_format = prediction_format

        self._vqa_format = DEFAULT_VQA_PROMPT_FORMAT

    def tokenize_cot(
        self,
        prompt: str,
        reasoning: str | None = None,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        *,
        frame_description: str = "end-effector frame",
        state_dropout: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Tokenize prompt and reasoning for chain-of-thought model.

        Args:
            prompt: Task description
            reasoning: Optional language actions/reasoning
            state: Optional state vector
            state_type: Optional state type descriptor
            is_vqa_sample: Whether this is a VQA sample
            is_prediction_sample: Whether this is a prediction sample
            time_horizon_seconds: Optional time horizon for predictions

        Returns:
            Tuple of (tokens, attn_mask, reasoning_mask, number_mask, direction_mask,
                     units_number_mask, digit_values):
            - tokens: Tokenized sequence [max_len]
            - attn_mask: Attention mask (True for non-pad positions) [max_len]
            - reasoning_mask: Mask for reasoning/language action tokens [max_len]
            - number_mask: Mask for all number tokens [max_len]
            - direction_mask: Mask for direction tokens [max_len]
        """
        # Resolve prompt format

        fmt = self._prompt_format

        # Pass time_horizon_seconds to format_prompt (only for robot tasks, not VQA)
        formatted_prompt = fmt.format_prompt(
            prompt,
            state,
            state_type,
            frame_description=frame_description,
            state_dropout=state_dropout,
        )

        # print(formatted_prompt)

        # Tokenize
        pad_id = self._tokenizer.pad_id()

        tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        reasoning_start = len(tokens)
        if reasoning is not None:
            clean_reason = reasoning.strip().replace("_", " ").replace("\n", " ")
            tokens += self._tokenizer.encode(clean_reason, add_bos=False, add_eos=True)
        reasoning_end = len(tokens)

        if len(tokens) > self._max_len:
            # logging.warning(
            #     f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
            #     "Consider increasing the `max_token_len` in your model config if this happens frequently."
            # )
            tokens = tokens[: self._max_len]
            reasoning_end = min(reasoning_end, self._max_len)

        # Create masks
        attn_mask = np.zeros(self._max_len, dtype=bool)
        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        token_loss_mask = np.ones(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[: len(tokens)] = True
        # Shift reasoning indices by pad_count after left padding
        start_idx = max(0, min(self._max_len, reasoning_start))
        end_idx = max(0, min(self._max_len, reasoning_end))
        if end_idx > start_idx:
            reasoning_mask[start_idx:end_idx] = True

        if reasoning is None:
            reasoning_mask = None
            direction_mask = None
            number_mask = None
        else:
            if not 0.0 <= self.reasoning_mask_prob <= 1.0:
                raise ValueError(f"reasoning_mask_prob must be between 0.0 and 1.0, got {self.reasoning_mask_prob}")
            # Skip reasoning_mask_prob for VQA samples - they need stable loss
            if self.reasoning_mask_prob > 0.0 and end_idx > start_idx and not is_vqa_sample:
                reasoning_span_len = end_idx - start_idx
                drop_mask = np.random.rand(reasoning_span_len) < self.reasoning_mask_prob
                if np.any(drop_mask):
                    drop_indices = np.where(drop_mask)[0] + start_idx
                    token_loss_mask[drop_indices] = False
            number_mask = np.zeros(self._max_len, dtype=bool)
            direction_mask = np.zeros(self._max_len, dtype=bool)

        # Build number and direction masks using format-specific checkers
        # Only mark tokens within reasoning span (not in the prompt)
        for i in range(start_idx, end_idx):
            if not reasoning_mask[i]:
                continue
            piece = self._tokenizer.id_to_piece(tokens[i])
            if piece:
                if is_number(piece):
                    number_mask[i] = True
                if fmt.direction_token_checker(piece):
                    direction_mask[i] = True

        # Right pad
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = tokens + [pad_id] * pad_count

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            number_mask,
            direction_mask,
            token_loss_mask,
        )

    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens back to a string, skipping special tokens and placeholders."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        # Filter out tokens that are out of range for this tokenizer
        vocab_size = self._tokenizer.vocab_size()
        filtered_tokens = [t for t in tokens if 0 <= t < vocab_size]
        
        return self._tokenizer.decode(filtered_tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


class FASTTokenizer(PaligemmaCoTTokenizer):
    def __init__(self, fast_tokenizer_path: str, **kwargs):
        super().__init__(**kwargs)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    def tokenize_fast_cot(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None = None,
        language_actions: str | None = None,
        state_type: str | None = None,
        *,
        state_dropout: float = 0.0,
        clip_action: bool = False,
        frame_description: str = "end-effector frame",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize prompt and actions for FAST model.

        For VQA and prediction samples: uses the same logic as PaligemmaCoTTokenizer.tokenize_cot
        (no action tokens, just language tokens with proper loss masks).

        For regular robot samples: uses FAST-style action token prediction.

        Args:
            prompt: Task description
            state: Current state vector
            actions: Action sequence (optional, None during inference)
            language_actions: Language description of actions
            state_type: Type of state representation
            state_dropout: Probability of dropping state info
            clip_action: Whether to clip actions to [-3, 3]
            frame_description: Description of the coordinate frame

        Returns:
            (tokens, token_mask, ar_mask, loss_mask)
            - tokens: Token IDs
            - token_mask: Mask for valid (non-padding) tokens
            - ar_mask: Autoregressive mask (False=prefix attention, True=causal)
            - loss_mask: Mask for tokens that contribute to loss
        """

        # For regular robot samples, use FAST action token logic
        fmt = self._prompt_format

        formatted_prompt = fmt.format_prompt(
            prompt,
            state,
            state_type,
            state_dropout=state_dropout,
            frame_description=frame_description,
        )

        # Tokenize prompt
        pad_id = self._tokenizer.pad_id()
        prefix_tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        # Append action tokens for robot samples
        if actions is not None:
            if clip_action:
                actions = np.clip(actions, -3.0, 3.0)
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = action_tokens_in_pg.tolist() + self._tokenizer.encode("|", add_eos=True)
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        if len(tokens) > self._max_len:
            logging.warning(
                f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                "Consider increasing the `max_token_len` in your model config."
            )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        # Right pad to max length
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = tokens + [pad_id] * pad_count
            token_mask = token_mask + [False] * pad_count
            ar_mask = ar_mask + [False] * pad_count
            loss_mask = loss_mask + [False] * pad_count

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        if tokens.ndim > 1:
            tokens = tokens[0]
        decoded_tokens = self._tokenizer.decode(tokens.tolist())


        raw_action_tokens = np.array(self._tokenizer.encode(decoded_tokens.split("|")[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]
