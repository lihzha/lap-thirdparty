from collections.abc import Callable
import dataclasses
import logging
import os
import re
from typing import Literal

from etils import epath  # optional, but handy
import numpy as np
from openpi.models import tokenizer as _tokenizer
import sentencepiece

import openpi_cot.shared.download as download

# Special token placeholder for image positions
TOKEN_PLACEHOLDER = -2


@dataclasses.dataclass
class StateDiscretizationConfig:
    """Configuration for discretizing state vectors into text."""

    bins: int = 256
    min_dim: int = 7  # Minimum number of dimensions to include (avoid over-trimming)
    range_min: float = -1.0
    range_max: float = 1.0


def _is_critical_directional(piece: str) -> bool:
    """Check if token contains digits or directional words (for natural language formats)."""
    # Check for digits
    if re.search(r"[0-9]", piece):
        return True
    # Check for directional words (case-insensitive)
    piece_lower = piece.lower()
    directional_words = ["right", "left", "forward", "up", "down", "back"]
    return any(word in piece_lower for word in directional_words)


def _is_critical_schema(piece: str) -> bool:
    """Check if token contains digits or +/- symbols (for schema-based formats)."""
    # Check for digits
    if re.search(r"[0-9]", piece):
        return True
    # Check for +/- symbols
    if "+" in piece or "-" in piece:
        return True
    return False


def _is_critical_default(piece: str) -> bool:
    """Default critical token checker - only digits."""
    return bool(re.search(r"[0-9]", piece))


@dataclasses.dataclass
class PromptComponent:
    """A modular component of a prompt.

    Each component can be one of:
    - task_prefix: Format for task instruction (e.g., "Task: {prompt}")
    - state_prefix: Format for state (e.g., "State ({state_label}): {state}")
    - schema: Schema/instruction text (e.g., coordinate system description)
    - action_prefix: Prefix before action output (e.g., "Action: ")
    """

    type: Literal["task_prefix", "state_prefix", "schema", "action_prefix"]
    template: str
    # Whether to include state type label in state prefix
    include_state_type: bool = True


@dataclasses.dataclass
class PromptFormat:
    """Defines how to format prompts for tokenization using modular components.

    This allows easy extension to support different prompt formats by composing
    components in different ways.
    """

    name: str
    components: list[PromptComponent]
    state_config: StateDiscretizationConfig | None = None
    # Separator between components (e.g., ", " or "\n")
    separator: str = ""
    # Function to determine if a token piece is critical for this format
    critical_token_checker: Callable[[str], bool] = _is_critical_default

    @property
    def include_state(self) -> bool:
        """Check if this format includes state."""
        return any(c.type == "state_prefix" for c in self.components)

    def format_prompt(self, prompt: str, state: np.ndarray | None = None, state_type: str | None = None) -> str:
        """Format the prompt with optional state and state type.

        Args:
            prompt: The task prompt/instruction
            state: Optional state vector to discretize and include
            state_type: Optional state type ("joint_pos", "eef_pose", "none")

        Returns:
            Formatted prompt string ready for tokenization
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").rstrip(".")

        # Prepare state-related variables
        state_str = ""
        state_label = ""
        if state is not None and state_type != "none":
            # Map state_type to human-readable label
            state_type_labels = {
                "joint_pos": "joint position",
                "eef_pose": "end-effector pose",
            }
            state_label = state_type_labels.get(state_type, state_type) if state_type else ""

            if self.state_config is not None:
                state_str = self._discretize_state(state)

        # Build prompt by chaining components
        parts = []
        for component in self.components:
            if component.type == "task_prefix":
                parts.append(component.template.format(prompt=cleaned_prompt))
            elif component.type == "state_prefix":
                if state is None or state_type == "none":
                    # Skip state component if no state
                    if component.include_state_type:
                        parts.append(component.template.format(state="", state_label="None"))
                    else:
                        parts.append(component.template.format(state="", state_label=""))
                else:
                    if self.state_config is None:
                        raise ValueError(f"State config required for prompt format '{self.name}'")
                    if component.include_state_type:
                        parts.append(component.template.format(state=state_str, state_label=state_label))
                    else:
                        parts.append(component.template.format(state=state_str, state_label=""))
            elif component.type == "schema" or component.type == "action_prefix":
                parts.append(component.template)

        return self.separator.join(parts)

    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize state vector into string representation.

        Trims trailing zero-padded dimensions and discretizes to bins.
        """
        assert self.state_config is not None
        state_arr = np.asarray(state)
        eps = 1e-8

        # Trim zero-padded dimensions
        if state_arr.ndim == 1:
            non_zero_mask = np.abs(state_arr) > eps
            last_idx = int(np.nonzero(non_zero_mask)[0][-1]) + 1 if np.any(non_zero_mask) else 0
            last_idx = max(last_idx, self.state_config.min_dim)
            trimmed = state_arr[:last_idx]
        else:
            flat = state_arr.reshape(-1, state_arr.shape[-1])
            non_zero_cols = np.any(np.abs(flat) > eps, axis=0)
            last_idx = int(np.nonzero(non_zero_cols)[0][-1]) + 1 if np.any(non_zero_cols) else 0
            last_idx = max(last_idx, self.state_config.min_dim)
            trimmed = state_arr[..., :last_idx].reshape(-1)

        if trimmed.size > 0:
            bins = np.linspace(self.state_config.range_min, self.state_config.range_max, self.state_config.bins + 1)[
                :-1
            ]
            discretized_state = np.digitize(trimmed, bins=bins) - 1
            return " ".join(map(str, discretized_state))
        return ""


# Predefined prompt formats - easily extensible by adding new instances
PI05_PROMPT_FORMAT = PromptFormat(
    name="pi05",
    components=[
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Action: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=", ",
    critical_token_checker=_is_critical_directional,
)

PI0_PROMPT_FORMAT = PromptFormat(
    name="pi0",
    components=[
        PromptComponent("task_prefix", "{prompt}\n"),
    ],
    state_config=None,
    separator="",
)

VQA_PROMPT_FORMAT = PromptFormat(
    name="vqa",
    components=[
        PromptComponent("task_prefix", "{prompt}"),
    ],
    state_config=None,
    separator="",
)

COORDINATE_SYSTEM_PROMPT_FORMAT = PromptFormat(
    name="coordinate_system",
    components=[
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=True),
        PromptComponent("schema", "Actions are represented as [x,y,z], where +x is forward, +y is left, +z is up."),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=", ",
    critical_token_checker=_is_critical_schema,
)

SCHEMA_COMPACT_PROMPT_FORMAT = PromptFormat(
    name="schema_compact",
    components=[
        PromptComponent(
            "schema",
            "Schema: <dx dy dz g>; units cm; +x fwd, +y left, +z up; g∈{0=close,1=open}",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        # PromptComponent(
        #     "schema",
        #     "Actions schema: dx dy dz droll dpitch dyaw grip; units cm/deg; +x forward, +y left, +z up; grip∈{{0=open,1=close}}.",
        # ),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    # separator="\n",
    separator=". ",
    critical_token_checker=_is_critical_schema,
)

SCHEMA_COMPACT_WITH_ROTATION_PROMPT_FORMAT = PromptFormat(
    name="schema_compact_with_rotation",
    components=[
        PromptComponent(
            "schema",
            "Schema: <dx dy dz dr dp dy g>; units cm/deg; +x fwd, +y left, +z up; dr=roll, dp=pitch, dy=yaw; g∈{0=close,1=open}",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=". ",
    critical_token_checker=_is_critical_schema,
)

SCHEMA_COMPACT_BIMANUAL_PROMPT_FORMAT = PromptFormat(
    name="schema_compact_bimanual",
    components=[
        PromptComponent(
            "schema",
            "Schema: <L dx dy dz g R dx dy dz g>; units cm; +x fwd, +y left, +z up; L=left arm, R=right arm; g∈{0=close,1=open}",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=". ",
    critical_token_checker=_is_critical_schema,
)

SCHEMA_COMPACT_BIMANUAL_WITH_ROTATION_PROMPT_FORMAT = PromptFormat(
    name="schema_compact_bimanual_with_rotation",
    components=[
        PromptComponent(
            "schema",
            "Schema: <L dx dy dz dr dp dy g R dx dy dz dr dp dy g>; units cm/deg; +x fwd, +y left, +z up; dr=roll, dp=pitch, dy=yaw; L=left, R=right; g∈{0=close,1=open}",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7),
    separator=". ",
    critical_token_checker=_is_critical_schema,
)

# Registry for easy lookup
PROMPT_FORMAT_REGISTRY = {
    "pi05": PI05_PROMPT_FORMAT,
    "pi0": PI0_PROMPT_FORMAT,
    "vqa": VQA_PROMPT_FORMAT,
    "coordinate_system": COORDINATE_SYSTEM_PROMPT_FORMAT,
    "schema_compact": SCHEMA_COMPACT_PROMPT_FORMAT,
    "schema_compact_with_rotation": SCHEMA_COMPACT_WITH_ROTATION_PROMPT_FORMAT,
    "schema_compact_bimanual": SCHEMA_COMPACT_BIMANUAL_PROMPT_FORMAT,
    "schema_compact_bimanual_with_rotation": SCHEMA_COMPACT_BIMANUAL_WITH_ROTATION_PROMPT_FORMAT,
}


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
        ]
        | PromptFormat = "pi05",
        tokenizer_type: Literal["gemma3", "paligemma"] = "paligemma",
        num_images: int = 2,
        tokens_per_image: int = 256,
    ):
        # super().__init__(max_len)
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
        prompt_format: PromptFormat | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Resolve prompt format
        if prompt_format is None:
            fmt = self._prompt_format
        elif isinstance(prompt_format, str):
            if prompt_format not in PROMPT_FORMAT_REGISTRY:
                raise ValueError(
                    f"Unknown prompt format: {prompt_format}. Available formats: {list(PROMPT_FORMAT_REGISTRY.keys())}"
                )
            fmt = PROMPT_FORMAT_REGISTRY[prompt_format]
        else:
            fmt = prompt_format

        # Format the prompt using the PromptFormat
        formatted_prompt = fmt.format_prompt(prompt, state, state_type)

        # Tokenize
        pad_id = self._tokenizer.pad_id()

        # For Gemma3: [image_placeholders] + [BOS] + [text] + [reasoning] + [EOS]
        # For others: [BOS] + [text] + [reasoning] + [EOS]
        if self._tokenizer_type == "gemma3":
            image_placeholders = self._create_image_placeholders()
            # formatted_prompt = "<start_of_turn>user\n" + formatted_prompt + "<end_of_turn>\n<start_of_turn>model"
            text_tokens = (
                self._tokenizer.encode("<start_of_image>", add_bos=True, add_eos=False)
                + image_placeholders
                + self._tokenizer.encode("<end_of_image>\n<start_of_image>")
                + image_placeholders
                + self._tokenizer.encode(
                    "<end_of_image>\n" + formatted_prompt + "",
                    add_bos=False,
                    add_eos=False,
                )
            )
            tokens = text_tokens
            # text_tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
            # tokens = image_placeholders + text_tokens
        else:
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
        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        numeric_mask = np.zeros(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[pad_count:] = True

        # Shift reasoning indices by pad_count after left padding
        start_idx = max(0, min(self._max_len, reasoning_start + pad_count))
        end_idx = max(0, min(self._max_len, reasoning_end + pad_count))
        if end_idx > start_idx:
            reasoning_mask[start_idx:end_idx] = True

        # Build critical token mask using format-specific checker
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

        for i in range(start_idx, end_idx):
            if i < 0 or i >= len(pieces):
                continue
            piece = pieces[i]
            if piece and fmt.critical_token_checker(piece):
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
        return self._tokenizer.decode(tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def tokenize_prediction(
        self, prediction_prompt: str, prediction_language: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize for prediction task.

        Uses the prediction language action as the reasoning to be predicted.

        Args:
            prediction_prompt: The prompt for prediction task
            prediction_language: The prediction language action (reasoning target)
        """
        # The reasoning is the prediction language action
        # Use VQA format (no state) for prediction tasks
        return self.tokenize_cot(
            prediction_prompt, reasoning=prediction_language, state=None, prompt_format=VQA_PROMPT_FORMAT
        )
