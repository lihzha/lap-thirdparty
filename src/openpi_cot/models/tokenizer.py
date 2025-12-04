import logging
import os
from typing import Literal

from etils import epath  # optional, but handy
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

# Special token placeholder for image positions
TOKEN_PLACEHOLDER = -2


class PaligemmaCoTTokenizer(_tokenizer.PaligemmaTokenizer):
    # Gemma3 special tokens (only used when tokenizer_type == "gemma3")
    BEGIN_IMAGE_TOKEN = 255999
    END_IMAGE_TOKEN = 256000
    NEW_LINE_TOKEN = 108

    def __init__(
        self,
        max_len: int = 48,
        prompt_format: Literal[
            "pi05",
            "pi0",
            "vqa",
            "coordinate_system",
            "schema_compact",
            "schema_compact_with_rotation",
            "schema_compact_bimanual",
            "schema_compact_bimanual_with_rotation",
            "schema_compact_named_params",
            "verbose_state",
            "grouped_state",
            "grouped_state_verbose",
            "no_state",
            "pi05_notime",
        ]
        | PromptFormat = "pi05",
        prediction_format: Literal["default", "grouped"] | PromptFormat = "default",
        tokenizer_type: Literal["gemma3", "paligemma"] = "paligemma",
        num_images: int = 2,
        tokens_per_image: int = 256,
        enable_number_label_smoothing: bool = False,
    ):
        # super().__init__(max_len)
        self.enable_number_label_smoothing = enable_number_label_smoothing
        if tokenizer_type == "paligemma":
            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with path.open("rb") as f:
                self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        else:
            cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
            path = epath.Path(cache_dir + "/gemma3-tokenizer.model")
            with path.open("rb") as f:
                self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._max_len = max_len
        self._stop_token_id = self._tokenizer.eos_id()
        self._tokenizer_type = tokenizer_type
        self._num_images = num_images
        self._tokens_per_image = tokens_per_image

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

    def _create_image_placeholders(self) -> list[int]:
        """Create placeholder token sequence for images with special tokens (Gemma3 only).

        Format for each image: [NL, BEGIN_IMAGE, -2 x tokens_per_image, END_IMAGE, NL]
        Returns the full sequence for all images.
        """
        if self._tokenizer_type != "gemma3":
            return []

        return [TOKEN_PLACEHOLDER] * self._tokens_per_image

        # single_image_seq = (
        #     [self.NEW_LINE_TOKEN, self.BEGIN_IMAGE_TOKEN]
        #     + [TOKEN_PLACEHOLDER] * self._tokens_per_image
        #     + [self.NEW_LINE_TOKEN]
        #     # + [self.END_IMAGE_TOKEN, self.NEW_LINE_TOKEN]
        # )
        # return single_image_seq * self._num_images

    def tokenize_cot(
        self,
        prompt: str,
        reasoning: str | None = None,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
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
            - units_number_mask: Mask for units digit tokens only [max_len]
            - digit_values: Digit values 0-9 for number tokens, -1 for non-digits [max_len]
        """
        # Resolve prompt format

        if is_prediction_sample:
            fmt = self._prediction_format
        elif is_vqa_sample:
            fmt = self._vqa_format
        else:
            fmt = self._prompt_format

        # Pass time_horizon_seconds to format_prompt (only for robot tasks, not VQA)
        formatted_prompt = fmt.format_prompt(
            prompt, state, state_type, time_horizon_seconds=time_horizon_seconds if not is_vqa_sample else None
        )

        # Tokenize
        pad_id = self._tokenizer.pad_id()

        # For Gemma3: [image_placeholders] + [BOS] + [text] + [reasoning] + [EOS]
        # For others: [BOS] + [text] + [reasoning] + [EOS]
        if self._tokenizer_type == "gemma3":
            image_placeholders = self._create_image_placeholders()
            # formatted_prompt = "<start_of_turn>user\n" + formatted_prompt + "<end_of_turn>\n<start_of_turn>model"
            text_tokens = (
                self._tokenizer.encode("<start_of_turn>user\n<start_of_image>", add_bos=True, add_eos=False)
                + image_placeholders
                + self._tokenizer.encode("<end_of_image>\n<start_of_image>", add_bos=False, add_eos=False)
                + image_placeholders
                + self._tokenizer.encode(
                    "<end_of_image>\n" + formatted_prompt + "<end_of_turn>\n<start_of_turn>model",
                    add_bos=False,
                    add_eos=False,
                )
            )
            # text_tokens = (
            #     self._tokenizer.encode("\n<start_of_image>", add_bos=False, add_eos=False)
            #     + image_placeholders
            #     + self._tokenizer.encode("<end_of_image>\n<start_of_image>", add_bos=False, add_eos=False)
            #     + image_placeholders
            #     + self._tokenizer.encode(
            #         "<end_of_image>\n" + formatted_prompt + "",
            #         add_bos=False,
            #         add_eos=False,
            #     )
            # )
            tokens = text_tokens
            # text_tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
            # tokens = image_placeholders + text_tokens
        else:
            # print(formatted_prompt)
            tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

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

        # Left pad to max length for generation/training
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = [pad_id] * pad_count + tokens

        # Create masks
        attn_mask = np.zeros(self._max_len, dtype=bool)
        number_mask = np.zeros(self._max_len, dtype=bool)
        direction_mask = np.zeros(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[pad_count:] = True

        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        # Shift reasoning indices by pad_count after left padding
        start_idx = max(0, min(self._max_len, reasoning_start + pad_count))
        end_idx = max(0, min(self._max_len, reasoning_end + pad_count))
        if end_idx > start_idx:
            reasoning_mask[start_idx:end_idx] = True

        if reasoning is None:
            reasoning_mask = None

        # Build number and direction masks using format-specific checkers
        # Only mark tokens within reasoning span (not in the prompt)
        # Skip placeholder tokens and special tokens when building pieces
        pieces = []
        for t in tokens:
            if t == TOKEN_PLACEHOLDER:
                pieces.append("")  # Empty string for placeholders
            elif self._tokenizer_type == "gemma3" and t in (
                self.BEGIN_IMAGE_TOKEN,
                self.END_IMAGE_TOKEN,
                self.NEW_LINE_TOKEN,
            ):
                pieces.append("")  # Empty string for Gemma3 special tokens
            else:
                pieces.append(self._tokenizer.id_to_piece(t))

        if not is_vqa_sample:
            for i in range(start_idx, end_idx):
                if i < 0 or i >= len(pieces):
                    continue
                piece = pieces[i]
                if piece:
                    if is_number(piece):
                        number_mask[i] = True
                    if fmt.direction_token_checker(piece):
                        direction_mask[i] = True

        units_number_mask = None
        digit_values = None

        if not is_vqa_sample and self.enable_number_label_smoothing:
            logging.info("Building units_number_mask and digit_values for label smoothing.")
            # Create units_number_mask and digit_values for label smoothing
            units_number_mask = np.zeros(self._max_len, dtype=bool)
            digit_values = np.full(self._max_len, -1, dtype=np.int8)  # -1 for non-digits
            # Unit words that follow numbers in language actions
            unit_words = {"cm", "▁cm", "degrees", "▁degrees", "mm", "▁mm", "m", "▁m", "radians", "▁radians"}

            # Find digit sequences followed by unit words
            i = start_idx
            while i < end_idx:
                if i >= len(pieces):
                    break

                # Check if current position starts a digit sequence
                if number_mask[i]:
                    # Find the end of the digit sequence
                    seq_start = i
                    seq_end = i
                    while seq_end < end_idx and seq_end < len(pieces) and number_mask[seq_end]:
                        seq_end += 1

                    # Check if the next token is a unit word
                    next_idx = seq_end
                    is_followed_by_unit = False
                    if next_idx < len(pieces):
                        next_piece = pieces[next_idx]
                        if next_piece in unit_words:
                            is_followed_by_unit = True

                    # If followed by unit, mark the last digit in sequence as units digit
                    if is_followed_by_unit:
                        units_digit_idx = seq_end - 1
                        units_number_mask[units_digit_idx] = True

                        # Extract the digit value from the piece
                        units_piece = pieces[units_digit_idx]
                        # Find the rightmost digit in the piece
                        for char in reversed(units_piece):
                            if char.isdigit():
                                digit_values[units_digit_idx] = int(char)
                                break

                    # Also extract digit values for all number tokens (for potential future use)
                    for j in range(seq_start, seq_end):
                        if j < len(pieces):
                            piece = pieces[j]
                            for char in reversed(piece):
                                if char.isdigit():
                                    digit_values[j] = int(char)
                                    break

                    # Move to the end of the sequence
                    i = seq_end
                else:
                    i += 1

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            number_mask,
            direction_mask,
            units_number_mask,
            digit_values,
        )

    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens back to a string, skipping special tokens and placeholders."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        # Filter out placeholder tokens and special tokens
        filtered_tokens = []
        for t in tokens:
            if t == TOKEN_PLACEHOLDER:
                continue  # Skip placeholder tokens
            if self._tokenizer_type == "gemma3" and t in (
                self.BEGIN_IMAGE_TOKEN,
                self.END_IMAGE_TOKEN,
                self.NEW_LINE_TOKEN,
            ):
                continue  # Skip Gemma3 special tokens
            filtered_tokens.append(t)

        return self._tokenizer.decode(filtered_tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


class FASTTokenizer(PaligemmaCoTTokenizer):
    def __init__(self, fast_tokenizer_path: str, **kwargs):
        super().__init__(**kwargs)
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    def tokenize_fast_cot(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None = None,
        reasoning: str | None = None,
        state_type: str | None = None,
        *,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize prompt, language actions (if any), state, and actions for FAST model.

        This method combines CoT-style reasoning with FAST-style action token prediction:
        - First tokenizes prompt + language actions (optional)
        - Then appends state and action representations as tokens
        - Creates appropriate masks for training

        Args:
            prompt: Task description
            state: Current state vector
            actions: Action sequence (optional, None during inference)
            language_actions: Language description of actions (optional)
            state_type: Type of state representation
            is_vqa_sample: Whether this is a VQA sample
            is_prediction_sample: Whether this is a prediction sample

        Returns:
            (tokens, token_mask, ar_mask, loss_mask)
            - tokens: Token IDs including prompt, language actions, state, and actions
            - token_mask: Mask for valid (non-padding) tokens
            - ar_mask: Autoregressive mask (0=prefix attention, 1=causal)
            - loss_mask: Mask for tokens that contribute to loss
        """
        # Resolve prompt format
        if is_prediction_sample:
            fmt = self._prediction_format
        elif is_vqa_sample:
            fmt = self._vqa_format
        else:
            fmt = self._prompt_format

        # Pass time_horizon_seconds to format_prompt (only for robot tasks, not VQA)
        formatted_prompt = fmt.format_prompt(
            prompt, state, state_type, time_horizon_seconds=time_horizon_seconds if not is_vqa_sample else None
        )

        # Tokenize prompt
        pad_id = self._tokenizer.pad_id()

        if self._tokenizer_type == "gemma3":
            image_placeholders = self._create_image_placeholders()
            text_tokens = (
                self._tokenizer.encode("<start_of_turn>user\n<start_of_image>", add_bos=True, add_eos=False)
                + image_placeholders
                + self._tokenizer.encode("<end_of_image>\n<start_of_image>", add_bos=False, add_eos=False)
                + image_placeholders
                + self._tokenizer.encode(
                    "<end_of_image>\n" + formatted_prompt + "<end_of_turn>\n<start_of_turn>model",
                    add_bos=False,
                    add_eos=False,
                )
            )
            prefix_tokens = text_tokens
        else:
            prefix_tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        # # TODO: ignore language actions for now
        # reasoning_start = len(tokens)

        # # Add language actions if provided
        # if reasoning is not None:
        #     clean_reason = reasoning.strip().replace("_", " ").replace("\n", " ")
        #     reasoning_tokens = self._tokenizer.encode(clean_reason, add_bos=False, add_eos=False)
        #     tokens += reasoning_tokens

        # reasoning_end = len(tokens)

        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = action_tokens_in_pg.tolist() + self._tokenizer.encode("|", add_eos=True)
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
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

        # Left pad to max length
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = [pad_id] * pad_count + tokens
            token_mask = [False] * pad_count + token_mask
            ar_mask = [0] * pad_count + ar_mask
            loss_mask = [False] * pad_count + loss_mask

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._tokenizer.vocab_size() - 1 - 128 - tokens

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        # if "Action: " not in decoded_tokens:
        #     return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        # raw_action_tokens = np.array(
        #     self._tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        # )
        raw_action_tokens = np.array(self._tokenizer.encode(decoded_tokens.split("|")[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]
