from collections.abc import Callable
import dataclasses
import random

import numpy as np

import openpi_cot.models.prompt_utils.checkers as checkers
from openpi_cot.models.prompt_utils.state import StateDiscretizationConfig
from openpi_cot.models.prompt_utils.state import StateModule


@dataclasses.dataclass
class PrefixModule:
    """Module for system prompts and schemas that appear before the task.

    Examples:
    - Schema descriptions (e.g., "Schema: <dx dy dz g>; units cm; ...")
    - Coordinate system descriptions
    - Other system-level instructions
    """

    text: str

    def format_prefix(self) -> str:
        """Return the prefix text."""
        return self.text


@dataclasses.dataclass
class TaskModule:
    """Module for formatting task instructions.

    Handles cleaning and formatting of task prompts.
    """

    template: str = "Task: {prompt}, predict the robot's action in the {frame_description}"
    # Whether to include time horizon instruction when provided
    include_time_horizon: bool = False
    # Template for time horizon instruction
    time_horizon_template: str = (
        "predict the robot's action in the future {time_horizon_seconds} seconds in the end-effector frame"
    )

    def format_task(
        self, prompt: str, time_horizon_seconds: float | None = None, frame_description: str = "end-effector frame"
    ) -> str:
        """Format task prompt.

        Args:
            prompt: Raw task instruction

        Returns:
            Formatted task string
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").rstrip(".")
        if self.include_time_horizon:
            assert time_horizon_seconds is not None, "Time horizon must be provided if include_time_horizon is True"
            cleaned_prompt += ", "
            time_horizon_seconds = round(time_horizon_seconds * 2) / 2.0
            cleaned_prompt += self.time_horizon_template.format(time_horizon_seconds=time_horizon_seconds)
        return self.template.format(prompt=cleaned_prompt, frame_description=frame_description)


@dataclasses.dataclass
class ActionModule:
    """Module for action prefix that appears before action output.

    Examples:s
    - "Action: "

    Can optionally include time horizon information.
    """

    prefix: str = "Action: "

    def format_action_prefix(self) -> str:
        """Return the action prefix, optionally including time horizon.

        Args:
            time_horizon_seconds: Optional time horizon in seconds

        Returns:
            Formatted action prefix string
        """

        return self.prefix


@dataclasses.dataclass
class PromptFormat:
    """Defines how to format prompts for tokenization using modular components.

    This allows easy extension to support different prompt formats by composing
    modules in different ways. Supports both training and prediction prompts.
    """

    name: str
    # Optional modules - None means skip that component
    prefix_module: PrefixModule | None = None
    task_module: TaskModule | None = None
    state_module: StateModule | None = None
    action_module: ActionModule | None = None
    # Separator between components (e.g., ", " or "\n")
    separator: str = ""
    # Function to determine if a token piece is critical for this format
    critical_token_checker: Callable[[str], bool] = checkers.is_critical_default
    # Function to determine if a token piece contains direction information
    direction_token_checker: Callable[[str], bool] = checkers.is_direction_none

    @property
    def include_state(self) -> bool:
        """Check if this format includes state."""
        return self.state_module is not None

    def format_prompt(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        time_horizon_seconds: float | None = None,
        frame_description: str = "end-effector frame",
        state_dropout: float = 0.0,
    ) -> str:
        """Format the prompt with optional state, state type, and time horizon.

        Args:
            prompt: The task prompt/instruction
            state: Optional state vector to discretize and include
            state_type: Optional state type ("joint_pos", "eef_pose", "none")
            time_horizon_seconds: Optional time horizon in seconds for action prediction

        Returns:
            Formatted prompt string ready for tokenization
        """
        parts = []

        # Add prefix/schema if present
        if self.prefix_module is not None:
            parts.append(self.prefix_module.format_prefix())

        # Add task
        if self.task_module is not None:
            parts.append(
                self.task_module.format_task(
                    prompt=prompt, time_horizon_seconds=time_horizon_seconds, frame_description=frame_description
                )
            )

        # First determine if dropout applies
        add_state = True
        if self.state_module is None or state is None or (state_dropout > 0.0 and random.random() < state_dropout):
            add_state = False

        if add_state:
            state_str = self.state_module.format_state(state=state, state_type=state_type)
            if state_str:  # Only add if non-empty
                parts.append(state_str)

        # Add action prefix (with optional time horizon)
        if self.action_module is not None:
            parts.append(self.action_module.format_action_prefix())

        return self.separator.join(parts)


# Predefined prompt formats - easily extensible by adding new instances
PI05_PROMPT_FORMAT = PromptFormat(
    name="pi05",
    task_module=TaskModule(template="Task: {prompt}", include_time_horizon=True),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Action: "),
    separator="; ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)

PI05_NOTIME_PROMPT_FORMAT = PromptFormat(
    name="pi05_notime",
    task_module=TaskModule(include_time_horizon=False),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)

PI05_NOTIME_NOSTATE_PROMPT_FORMAT = PromptFormat(
    name="pi05_notime_nostate",
    task_module=TaskModule(include_time_horizon=False),
    state_module=None,
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)

PI05_NOTIME_ORI_PROMPT_FORMAT = PromptFormat(
    name="pi05_notime_ori",
    task_module=TaskModule(template="Task: {prompt}, ", include_time_horizon=False),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state};\n",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Action: "),
    separator="",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)

PI0_PROMPT_FORMAT = PromptFormat(
    name="pi0",
    task_module=TaskModule(template="{prompt}\n", include_time_horizon=False),
    state_module=None,
    action_module=None,
    separator="",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)


SCHEMA_COMPACT_PROMPT_FORMAT = PromptFormat(
    name="schema_compact",
    prefix_module=PrefixModule("Schema: <dx dy dz g>; units cm; +x fwd, +y left, +z up; g∈{0=close,1=open}"),
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Action: "),
    separator=". ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)

DEFAULT_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="default_prediction",
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    task_module=TaskModule(
        template="Task: predict the robot's action between two images in the {frame_description}",
        include_time_horizon=False,
    ),
    separator="; ",
    action_module=ActionModule(prefix="Answer: "),
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)


DEFAULT_VQA_PROMPT_FORMAT = PromptFormat(
    name="default_vqa",
    state_module=None,
    task_module=TaskModule(template="Task: {prompt}", include_time_horizon=False),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Registry for easy lookup
PROMPT_FORMAT_REGISTRY = {
    "pi05": PI05_PROMPT_FORMAT,
    "pi05_notime": PI05_NOTIME_PROMPT_FORMAT,
    "pi05_notime_ori": PI05_NOTIME_ORI_PROMPT_FORMAT,
    "pi0": PI0_PROMPT_FORMAT,
    "schema_compact": SCHEMA_COMPACT_PROMPT_FORMAT,
    "pi05_notime_nostate": PI05_NOTIME_NOSTATE_PROMPT_FORMAT,
}

PREDICTION_PROMPT_FORMAT_REGISTRY = {
    "default": DEFAULT_PREDICTION_PROMPT_FORMAT,
}
