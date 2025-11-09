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
class StateTemplate:
    """Template for formatting discretized state values.

    Allows flexible representation of state vectors with custom labels and formatting.
    """

    # Dimension labels in order (e.g., ["x", "y", "z", "rot1x", ...])
    # If None or shorter than values, uses generic labels
    dim_labels: list[str] | None = None

    # Format string for each dimension value
    # Can use {label} and {value} placeholders
    # e.g., "{label}={value:03d}" for "x=134" or just "{value}" for "134"
    dim_format: str = "{value}"

    # Separator between dimensions
    separator: str = " "

    def format_state(self, values: np.ndarray) -> str:
        """Format discretized state values according to template.

        Args:
            values: Array of discretized state values

        Returns:
            Formatted string representation
        """
        parts = []
        for i, val in enumerate(values):
            # Use provided label or generate generic one
            if self.dim_labels and i < len(self.dim_labels):
                label = self.dim_labels[i]
            else:
                label = f"dim{i}"

            parts.append(self.dim_format.format(label=label, value=int(val)))

        return self.separator.join(parts)


@dataclasses.dataclass
class GroupedStateTemplate:
    """Template for formatting state values grouped by semantic meaning.

    Groups dimensions into semantic categories (e.g., position, rotation, gripper).
    Example output: "position 134 088 076, rotation 201 054 233 128 254 033, gripper 200"
    """

    # Group labels (e.g., ["position", "rotation", "gripper"])
    group_labels: list[str]

    # Number of dimensions in each group (e.g., [3, 6, 1] for 10D state)
    group_sizes: list[int]

    # Format string for each value within a group
    # e.g., "{value:03d}" for zero-padded or "{value}" for no padding
    value_format: str = "{value:03d}"

    # Separator between groups
    group_separator: str = ", "

    # Separator between values within a group
    value_separator: str = " "

    def format_state(self, values: np.ndarray) -> str:
        """Format discretized state values with semantic grouping.

        Args:
            values: Array of discretized state values

        Returns:
            Formatted string representation with grouped dimensions
        """
        if sum(self.group_sizes) > len(values):
            # Handle case where state is shorter than expected
            # Adjust group sizes to fit available values
            adjusted_sizes = []
            remaining = len(values)
            for size in self.group_sizes:
                adjusted_sizes.append(min(size, remaining))
                remaining -= adjusted_sizes[-1]
                if remaining <= 0:
                    break
            group_sizes = adjusted_sizes
        else:
            group_sizes = self.group_sizes

        parts = []
        idx = 0
        for label, size in zip(self.group_labels, group_sizes):
            if idx >= len(values):
                break

            # Extract values for this group
            group_values = values[idx : idx + size]
            formatted_values = self.value_separator.join(self.value_format.format(value=int(v)) for v in group_values)

            # Format as "label value1 value2 ..."
            parts.append(f"{label} {formatted_values}")
            idx += size

        return self.group_separator.join(parts)


@dataclasses.dataclass
class StateDiscretizationConfig:
    """Configuration for discretizing state vectors into text."""

    bins: int = 256
    min_dim: int = 7  # Minimum number of dimensions to include (avoid over-trimming)
    range_min: float = -1.0
    range_max: float = 1.0
    template: StateTemplate | GroupedStateTemplate | None = None  # If None, uses default space-separated format

    def discretize_state(self, state: np.ndarray) -> str:
        """Discretize state vector into string representation.

        Trims trailing zero-padded dimensions and discretizes to bins.
        Uses the configured StateTemplate if provided, otherwise defaults to space-separated values.

        Args:
            state: State vector to discretize

        Returns:
            Formatted string representation of discretized state
        """
        state_arr = np.asarray(state)
        eps = 1e-8

        # Trim zero-padded dimensions
        if state_arr.ndim == 1:
            non_zero_mask = np.abs(state_arr) > eps
            last_idx = int(np.nonzero(non_zero_mask)[0][-1]) + 1 if np.any(non_zero_mask) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[:last_idx]
        else:
            flat = state_arr.reshape(-1, state_arr.shape[-1])
            non_zero_cols = np.any(np.abs(flat) > eps, axis=0)
            last_idx = int(np.nonzero(non_zero_cols)[0][-1]) + 1 if np.any(non_zero_cols) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[..., :last_idx].reshape(-1)

        if trimmed.size > 0:
            bins = np.linspace(self.range_min, self.range_max, self.bins + 1)[:-1]
            discretized_state = np.digitize(trimmed, bins=bins) - 1

            # Use template if provided, otherwise default to space-separated
            if self.template is not None:
                return self.template.format_state(discretized_state)
            return " ".join(map(str, discretized_state))
        return ""


def _is_number(piece: str) -> bool:
    """Check if token contains digits."""
    return bool(re.search(r"[0-9]", piece))


def _is_direction_natural(piece: str) -> bool:
    """Check if token contains directional words (for natural language formats)."""
    piece_lower = piece.lower()
    directional_words = ["right", "left", "forward", "up", "down", "back", "clockwise", "counterclockwise"]
    return any(word in piece_lower for word in directional_words)


def _is_direction_schema(piece: str) -> bool:
    """Check if token contains +/- symbols (for schema-based formats)."""
    return "+" in piece or "-" in piece


def _is_direction_none(piece: str) -> bool:
    """No direction tokens."""
    return False


def _is_critical_directional(piece: str) -> bool:
    """Check if token contains digits or directional words (for natural language formats)."""
    return _is_number(piece) or _is_direction_natural(piece)


def _is_critical_schema(piece: str) -> bool:
    """Check if token contains digits or +/- symbols (for schema-based formats)."""
    return _is_number(piece) or _is_direction_schema(piece)


def _is_critical_default(piece: str) -> bool:
    """Default critical token checker - only digits."""
    return _is_number(piece)


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
    # Function to determine if a token piece contains direction information
    direction_token_checker: Callable[[str], bool] = _is_direction_none

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
        Uses the configured StateTemplate if provided, otherwise defaults to space-separated values.
        """
        assert self.state_config is not None
        return self.state_config.discretize_state(state)


# Predefined state templates - easily extensible by adding new instances
DEFAULT_STATE_TEMPLATE = StateTemplate(
    dim_labels=None,
    dim_format="{value}",
    separator=" ",
)

NAMED_PARAMS_STATE_TEMPLATE = StateTemplate(
    dim_labels=["x", "y", "z", "rot1x", "rot1y", "rot1z", "rot2x", "rot2y", "rot2z", "grip"],
    dim_format="{label}={value:03d}",
    separator=" ",
)

VERBOSE_STATE_TEMPLATE = StateTemplate(
    dim_labels=[
        "position_x",
        "position_y",
        "position_z",
        "rotation_1_x",
        "rotation_1_y",
        "rotation_1_z",
        "rotation_2_x",
        "rotation_2_y",
        "rotation_2_z",
        "gripper",
    ],
    dim_format="{label}={value:03d}",
    separator=", ",
)

GROUPED_STATE_TEMPLATE = GroupedStateTemplate(
    group_labels=["position", "rotation", "gripper"],
    group_sizes=[3, 6, 1],
    value_format="{value:03d}",
    group_separator=", ",
    value_separator=" ",
)


# Predefined prediction state configurations
# These combine StateTemplate with discretization settings for prediction prompts
@dataclasses.dataclass
class PredictionStateConfig:
    """Configuration for including state in prediction prompts."""

    discretization: StateDiscretizationConfig
    prompt_prefix: str = (
        "Robot current state: {state}. Robot state is represented in the robot base frame, not in the camera's frame."
    )


# Prediction state templates - using same templates but with prediction-specific prompt
PREDICTION_STATE_DEFAULT = PredictionStateConfig(
    discretization=StateDiscretizationConfig(
        bins=256, min_dim=7, range_min=-1.0, range_max=1.0, template=DEFAULT_STATE_TEMPLATE
    ),
)

PREDICTION_STATE_NAMED_PARAMS = PredictionStateConfig(
    discretization=StateDiscretizationConfig(
        bins=256, min_dim=7, range_min=-1.0, range_max=1.0, template=NAMED_PARAMS_STATE_TEMPLATE
    ),
)

PREDICTION_STATE_VERBOSE = PredictionStateConfig(
    discretization=StateDiscretizationConfig(
        bins=256, min_dim=7, range_min=-1.0, range_max=1.0, template=VERBOSE_STATE_TEMPLATE
    ),
)

GROUPED_STATE_VERBOSE = PredictionStateConfig(
    discretization=StateDiscretizationConfig(
        bins=256, min_dim=7, range_min=-1.0, range_max=1.0, template=GROUPED_STATE_TEMPLATE
    ),
)

# Registry for prediction state configs
PREDICTION_STATE_CONFIG_REGISTRY = {
    "default": PREDICTION_STATE_DEFAULT,
    "named_params": PREDICTION_STATE_NAMED_PARAMS,
    "verbose": PREDICTION_STATE_VERBOSE,
    "grouped": GROUPED_STATE_VERBOSE,
}


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
    direction_token_checker=_is_direction_natural,
)

PI0_PROMPT_FORMAT = PromptFormat(
    name="pi0",
    components=[
        PromptComponent("task_prefix", "{prompt}\n"),
    ],
    state_config=None,
    separator="",
    direction_token_checker=_is_direction_none,
)

VQA_PROMPT_FORMAT = PromptFormat(
    name="vqa",
    components=[
        PromptComponent("task_prefix", "{prompt}"),
    ],
    state_config=None,
    separator="",
    direction_token_checker=_is_direction_none,
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
    direction_token_checker=_is_direction_schema,
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
    direction_token_checker=_is_direction_schema,
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
    direction_token_checker=_is_direction_schema,
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
    direction_token_checker=_is_direction_schema,
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
    direction_token_checker=_is_direction_schema,
)

SCHEMA_COMPACT_NAMED_PARAMS_PROMPT_FORMAT = PromptFormat(
    name="schema_compact_named_params",
    components=[
        PromptComponent(
            "schema",
            "Schema: <dx dy dz g>; units cm; +x fwd, +y left, +z up; g∈{0=close,1=open}",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7, template=NAMED_PARAMS_STATE_TEMPLATE),
    separator=". ",
    critical_token_checker=_is_critical_schema,
    direction_token_checker=_is_direction_schema,
)

VERBOSE_STATE_PROMPT_FORMAT = PromptFormat(
    name="verbose_state",
    components=[
        PromptComponent(
            "schema",
            "Your Robot control coordinate system: +x=forward, +y=left, +z=up.",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "Current state{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Predicted actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7, template=VERBOSE_STATE_TEMPLATE),
    separator="\n",
    critical_token_checker=_is_critical_schema,
    direction_token_checker=_is_direction_schema,
)

GROUPED_STATE_PROMPT_FORMAT = PromptFormat(
    name="grouped_state",
    components=[
        PromptComponent(
            "schema",
            "Your Robot control coordinate system: +x=forward, +y=left, +z=up.",
        ),
        PromptComponent("task_prefix", "Task: {prompt}"),
        PromptComponent("state_prefix", "State{state_label}: {state}", include_state_type=False),
        PromptComponent("action_prefix", "Actions: "),
    ],
    state_config=StateDiscretizationConfig(bins=256, min_dim=7, template=GROUPED_STATE_TEMPLATE),
    separator=". ",
    critical_token_checker=_is_critical_schema,
    direction_token_checker=_is_direction_schema,
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
    "schema_compact_named_params": SCHEMA_COMPACT_NAMED_PARAMS_PROMPT_FORMAT,
    "verbose_state": VERBOSE_STATE_PROMPT_FORMAT,
    "grouped_state": GROUPED_STATE_PROMPT_FORMAT,
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
            "schema_compact_named_params",
            "verbose_state",
            "grouped_state",
        ]
        | PromptFormat = "pi05",
        tokenizer_type: Literal["gemma3", "paligemma"] = "paligemma",
        num_images: int = 2,
        tokens_per_image: int = 256,
        prediction_state_config: Literal["default", "named_params", "verbose"] | PredictionStateConfig | None = None,
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

        # Support both string and PredictionStateConfig instance
        if isinstance(prediction_state_config, str):
            if prediction_state_config not in PREDICTION_STATE_CONFIG_REGISTRY:
                raise ValueError(
                    f"Unknown prediction state config: {prediction_state_config}. "
                    f"Available configs: {list(PREDICTION_STATE_CONFIG_REGISTRY.keys())}"
                )
            self._prediction_state_config = PREDICTION_STATE_CONFIG_REGISTRY[prediction_state_config]
        else:
            self._prediction_state_config = prediction_state_config

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
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        if not is_vqa_sample and not is_prediction_sample:
            # Format the prompt using the PromptFormat
            formatted_prompt = fmt.format_prompt(prompt, state, state_type)
        else:
            # For VQA samples and prediction samples, use simple formatting
            formatted_prompt = prompt.strip().replace("_", " ").replace("\n", " ")

            # For prediction samples, optionally prepend state information
            if is_prediction_sample and self._prediction_state_config is not None and state is not None:
                # Discretize state using the configured StateDiscretizationConfig
                state_str = self._prediction_state_config.discretization.discretize_state(state)

                # Prepend state information to the prompt using configured prefix
                if state_str:
                    formatted_prompt = (
                        f"{self._prediction_state_config.prompt_prefix.format(state=state_str)} {formatted_prompt}"
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
        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        number_mask = np.zeros(self._max_len, dtype=bool)
        direction_mask = np.zeros(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[pad_count:] = True

        # Shift reasoning indices by pad_count after left padding
        start_idx = max(0, min(self._max_len, reasoning_start + pad_count))
        end_idx = max(0, min(self._max_len, reasoning_end + pad_count))
        if end_idx > start_idx:
            reasoning_mask[start_idx:end_idx] = True

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
                    if _is_number(piece):
                        number_mask[i] = True
                    if fmt.direction_token_checker(piece):
                        direction_mask[i] = True

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            number_mask,
            direction_mask,
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
