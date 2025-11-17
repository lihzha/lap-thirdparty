from collections.abc import Callable
import dataclasses

import numpy as np

import openpi_cot.models.prompt_utils.checkers as checkers
from openpi_cot.models.prompt_utils.state import GROUPED_STATE_TEMPLATE
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

    template: str = "Task: {prompt}"

    def format_task(self, prompt: str) -> str:
        """Format task prompt.

        Args:
            prompt: Raw task instruction

        Returns:
            Formatted task string
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").rstrip(".")
        return self.template.format(prompt=cleaned_prompt)


@dataclasses.dataclass
class ActionModule:
    """Module for action prefix that appears before action output.

    Examples:
    - "Action: "
    - "Actions: "
    - "Predicted actions: "
    """

    prefix: str = "Action: "

    def format_action_prefix(self) -> str:
        """Return the action prefix."""
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
    ) -> str:
        """Format the prompt with optional state and state type.

        Args:
            prompt: The task prompt/instruction
            state: Optional state vector to discretize and include
            state_type: Optional state type ("joint_pos", "eef_pose", "none")
            is_prediction_sample: Whether this is a prediction sample (affects state formatting)
            is_vqa_sample: Whether this is a VQA sample

        Returns:
            Formatted prompt string ready for tokenization
        """
        parts = []

        # Add prefix/schema if present
        if self.prefix_module is not None:
            parts.append(self.prefix_module.format_prefix())

        # Add task
        if self.task_module is not None:
            parts.append(self.task_module.format_task(prompt))

        # Add state if module is present
        if self.state_module is not None:
            state_str = self.state_module.format_state(state=state, state_type=state_type)
            if state_str:  # Only add if non-empty
                parts.append(state_str)

        # Add action prefix
        if self.action_module is not None:
            parts.append(self.action_module.format_action_prefix())

        return self.separator.join(parts)


# Predefined prompt formats - easily extensible by adding new instances
PI05_PROMPT_FORMAT = PromptFormat(
    name="pi05",
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Action: "),
    separator=", ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)

PI0_PROMPT_FORMAT = PromptFormat(
    name="pi0",
    task_module=TaskModule(template="{prompt}\n"),
    separator="",
    direction_token_checker=checkers.is_direction_none,
)

VQA_PROMPT_FORMAT = PromptFormat(
    name="vqa",
    task_module=TaskModule(template="{prompt}"),
    separator="",
    direction_token_checker=checkers.is_direction_none,
)

COORDINATE_SYSTEM_PROMPT_FORMAT = PromptFormat(
    name="coordinate_system",
    prefix_module=PrefixModule("Actions are represented as [x,y,z], where +x is forward, +y is left, +z is up."),
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=True,
    ),
    action_module=ActionModule(prefix="Actions: "),
    separator=", ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)

SCHEMA_COMPACT_PROMPT_FORMAT = PromptFormat(
    name="schema_compact",
    prefix_module=PrefixModule("Schema: <dx dy dz g>; units cm; +x fwd, +y left, +z up; g∈{0=close,1=open}"),
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Actions: "),
    separator=". ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)

GROUPED_STATE_PROMPT_FORMAT = PromptFormat(
    name="grouped_state",
    prefix_module=PrefixModule("Your Robot control coordinate system: +x=forward, +y=left, +z=up."),
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7, template=GROUPED_STATE_TEMPLATE),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Actions: "),
    separator=". ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)


GROUPED_STATE_PREFIX_PROMPT_FORMAT = PromptFormat(
    name="grouped_state_verbose",
    prefix_module=PrefixModule("Predict what is the action that the robot should take"),
    task_module=TaskModule(template="Task: {prompt}"),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7, template=GROUPED_STATE_TEMPLATE),
        state_prefix_template="Robot control coordinate system is: +x=forward, +y=left, +z=up. Robot current state{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Actions: "),
    separator=". ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)

NO_STATE_FORMAT = PromptFormat(
    name="no_state",
    task_module=TaskModule(
        template="Task: {prompt}. What actions should the robot take at current step to fulfill the task? Represent the action in the end effector frame."
    ),
    state_module=None,
    action_module=ActionModule(prefix="Action: "),
    separator=" ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)


GROUPED_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="grouped_prediction",
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256, min_dim=7, template=GROUPED_STATE_TEMPLATE),
        state_prefix_template="Robot current state{state_label}: {state}. Robot state is represented in the robot base frame, not in the camera's frame.",
        include_state_type=False,
    ),
    task_module=TaskModule(template="{prompt}"),
    separator=" ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)

DEFAULT_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="default_prediction",
    state_module=None,
    task_module=TaskModule(template="{prompt}"),
    separator=" ",
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)


DEFAULT_VQA_PROMPT_FORMAT = PromptFormat(
    name="default_vqa",
    state_module=None,
    task_module=TaskModule(template="{prompt}"),
    separator=" ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Registry for easy lookup
PROMPT_FORMAT_REGISTRY = {
    "pi05": PI05_PROMPT_FORMAT,
    "pi0": PI0_PROMPT_FORMAT,
    "vqa": VQA_PROMPT_FORMAT,
    "coordinate_system": COORDINATE_SYSTEM_PROMPT_FORMAT,
    "schema_compact": SCHEMA_COMPACT_PROMPT_FORMAT,
    "grouped_state": GROUPED_STATE_PROMPT_FORMAT,
    "grouped_state_verbose": GROUPED_STATE_PREFIX_PROMPT_FORMAT,
    "no_state": NO_STATE_FORMAT,
}

PREDICTION_PROMPT_FORMAT_REGISTRY = {
    "default": DEFAULT_PREDICTION_PROMPT_FORMAT,
    "grouped": GROUPED_PREDICTION_PROMPT_FORMAT,
}
