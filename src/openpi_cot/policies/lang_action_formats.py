import dataclasses
import logging
import re
from typing import Literal

import numpy as np

from openpi_cot.policies.utils import transform_actions_from_eef_frame


@dataclasses.dataclass(frozen=True)
class LanguageActionFormat:
    """Defines how to format language actions.

    This modular structure allows easy extension to support different
    action description formats and styles.
    """

    name: str
    # Style of formatting
    style: Literal["verbose", "compact", "vla0"] = "verbose"
    # For verbose style: decimal places for numeric values
    decimal_places: int = 0
    # Whether to include rotation components in descriptions
    include_rotation: bool = False
    # Optional units for translation (cm, m, mm)
    translation_unit: str = "cm"
    # Whether to represent actions in end effector's frame (relative to first timestep)
    use_eef_frame: bool = False

    def get_sum_decimal(self) -> str:
        """Convert to legacy sum_decimal format for backward compatibility."""
        if self.style == "compact":
            return "compact"
        return f"{self.decimal_places}f"

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse language action(s) into translation/rotation deltas and gripper actions."""
        movement = np.zeros(6, dtype=float)  # [dx, dy, dz, droll, dpitch, dyaw]
        gripper_action = None

        if self.style == "compact":
            if self.include_rotation:
                pattern = re.compile(
                    r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                match = pattern.search(reasoning)
                if match:
                    groups = match.groups()
                    dx, dy, dz = groups[0], groups[1], groups[2]
                    movement[:3] = np.array([dx, dy, dz], dtype=float) / 100.0
                    if self.include_rotation:
                        droll, dpitch, dyaw = groups[3], groups[4], groups[5]
                        movement[3:6] = np.array([droll, dpitch, dyaw], dtype=float) * np.pi / 180.0

                    gripper_action = float(groups[-1])
        else:
            reasoning = reasoning.replace("slightly", "1.5 cm").replace("moderately", "5 cm").replace("a lot", "10 cm")
            move_pattern = re.compile(
                rf"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*{self.translation_unit})?",
                re.IGNORECASE,
            )

            dx_cm = dy_cm = dz_cm = 0.0
            for match in move_pattern.finditer(reasoning):
                direction = match.group(1).lower()
                value = float(match.group(2)) if match.group(2) is not None else 0.0
                if direction == "forward":
                    dx_cm += value
                elif direction in ("backward", "back"):
                    dx_cm -= value
                elif direction == "left":
                    dy_cm += value
                elif direction == "right":
                    dy_cm -= value
                elif direction == "up":
                    dz_cm += value
                elif direction == "down":
                    dz_cm -= value

            movement[:3] = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

            if self.include_rotation:
                rotation_pattern = re.compile(
                    r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
                    re.IGNORECASE,
                )
                droll_deg = dpitch_deg = dyaw_deg = 0.0
                for match in rotation_pattern.finditer(reasoning):
                    rotation_type = match.group(1).lower()
                    value = float(match.group(2))

                    if rotation_type == "tilt left":
                        droll_deg += value
                    elif rotation_type == "tilt right":
                        droll_deg -= value
                    elif rotation_type in {"tilt down", "tilt back"}:
                        dpitch_deg += value
                    elif rotation_type in {"tilt up", "tilt forward"}:
                        dpitch_deg -= value
                    elif rotation_type == "rotate counterclockwise":
                        dyaw_deg += value
                    elif rotation_type == "rotate clockwise":
                        dyaw_deg -= value

                movement[3:6] = [
                    droll_deg * np.pi / 180.0,
                    dpitch_deg * np.pi / 180.0,
                    dyaw_deg * np.pi / 180.0,
                ]

            grip_pattern = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)
            grip_match = grip_pattern.search(reasoning)
            if "open gripper" in reasoning.lower():
                gripper_action = 1.0
            elif "close gripper" in reasoning.lower():
                gripper_action = 0.0
            elif grip_match:
                gripper_action = float(grip_match.group(1))

        if self.use_eef_frame and initial_state is not None:
            movement = transform_actions_from_eef_frame(movement, initial_state)[0]

        return movement, gripper_action

# Predefined language action formats
VERBOSE_FORMAT = LanguageActionFormat(
    name="verbose",
    style="verbose",
    decimal_places=0,
    include_rotation=False,
)

VERBOSE_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="verbose_with_rotation",
    style="verbose",
    decimal_places=0,
    include_rotation=True,
)

PRECISION_FORMAT = LanguageActionFormat(
    name="precision",
    style="verbose",
    decimal_places=2,
    include_rotation=False,
)

COMPACT_FORMAT = LanguageActionFormat(
    name="compact",
    style="compact",
    decimal_places=0,
    include_rotation=False,
)

COMPACT_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="compact_with_rotation",
    style="compact",
    decimal_places=0,
    include_rotation=True,
)


EEF_FORMAT = LanguageActionFormat(
    name="verbose_eef", style="verbose", decimal_places=0, include_rotation=False, use_eef_frame=True
)

EEF_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="verbose_eef_with_rotation", style="verbose", decimal_places=0, include_rotation=True, use_eef_frame=True
)

COMPACT_BIMANUAL_FORMAT = LanguageActionFormat(
    name="compact_bimanual",
    style="compact",
    decimal_places=0,
    include_rotation=False,
)

COMPACT_BIMANUAL_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="compact_bimanual_with_rotation",
    style="compact",
    decimal_places=0,
    include_rotation=True,
)

# VLA-0 formats: actions as discretized integers
VLA0_FORMAT = VLA0ActionFormat(
    name="vla0",
    num_bins=1000,
    action_horizon=1,
    action_dim=7,
)

VLA0_CHUNKED_FORMAT = VLA0ActionFormat(
    name="vla0_chunked",
    num_bins=1000,
    action_horizon=10,
    action_dim=7,
)

LANGUAGE_ACTION_FORMAT_REGISTRY = {
    fmt.name: fmt
    for fmt in [
        VERBOSE_FORMAT,
        VERBOSE_WITH_ROTATION_FORMAT,
        PRECISION_FORMAT,
        COMPACT_FORMAT,
        COMPACT_WITH_ROTATION_FORMAT,
        EEF_FORMAT,
        EEF_WITH_ROTATION_FORMAT,
        COMPACT_BIMANUAL_FORMAT,
        COMPACT_BIMANUAL_WITH_ROTATION_FORMAT,
        VLA0_FORMAT,
        VLA0_CHUNKED_FORMAT,
    ]
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]
