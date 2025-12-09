import dataclasses
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
    style: Literal["verbose", "compact", "directional_only"] = "verbose"
    # For verbose style: decimal places for numeric values
    decimal_places: int = 0
    # Whether to include rotation components in descriptions
    include_rotation: bool = False
    # Optional units for translation (cm, m, mm)
    translation_unit: str = "cm"
    # Optional units for rotation (deg, rad)
    rotation_unit: str = "deg"
    # For compact style: use schema-based format like <+03 +05 -08 1>
    use_schema_format: bool = False
    # Whether to represent actions in end effector's frame (relative to first timestep)
    use_eef_frame: bool = False
    # Coordinate frame axis permutation and sign (for camera frame decoding)
    axis_perm: tuple[int, int, int] = (0, 2, 1)
    axis_sign: tuple[int, int, int] = (1, 1, 1)

    def get_sum_decimal(self) -> str:
        """Convert to legacy sum_decimal format for backward compatibility."""
        if self.style == "compact":
            return "compact"
        if self.style == "directional_only":
            return "no_number"
        # verbose
        return f"{self.decimal_places}f"

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        in_camera_frame: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse language action(s) into translation/rotation deltas and gripper actions."""
        sentences = [reasoning] if isinstance(reasoning, str) else list(reasoning)

        num_steps = len(sentences)
        translations = np.zeros((num_steps, 3), dtype=float)
        rotations = np.zeros((num_steps, 3), dtype=float)  # [roll, pitch, yaw] in radians
        gripper_actions = np.zeros((num_steps,), dtype=float)

        if self.use_schema_format and self.style == "compact":
            if self.include_rotation:
                pattern = re.compile(
                    r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, droll, dpitch, dyaw, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        rotations[i] = [
                            int(droll) * np.pi / 180.0,
                            int(dpitch) * np.pi / 180.0,
                            int(dyaw) * np.pi / 180.0,
                        ]
                        gripper_actions[i] = float(grip)
            else:
                pattern = re.compile(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>")
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        gripper_actions[i] = float(grip)
        else:
            move_pattern = (
                re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*{self.translation_unit})?",
                    re.IGNORECASE,
                )
                if self.style == "directional_only"
                else re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)\s+([\-\d\.]+)\s*{self.translation_unit}",
                    re.IGNORECASE,
                )
            )
            rotation_pattern = re.compile(
                r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
                re.IGNORECASE,
            )

            for i, sentence in enumerate(sentences):
                dx_cm = dy_cm = dz_cm = 0.0
                for match in move_pattern.finditer(sentence):
                    direction = match.group(1).lower()
                    value = float(match.group(2)) if match.group(2) is not None else 2.0
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

                v_m = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

                if in_camera_frame:
                    t_cam = np.zeros(3, dtype=float)
                    axis_perm = np.array(self.axis_perm)
                    axis_sign = np.array(self.axis_sign, dtype=float)
                    sign_safe = np.where(axis_sign == 0, 1.0, axis_sign)
                    t_mapped = v_m / sign_safe
                    t_cam[axis_perm] = t_mapped
                    translations[i] = t_cam
                else:
                    translations[i] = v_m

                if self.include_rotation:
                    droll_deg = dpitch_deg = dyaw_deg = 0.0
                    for match in rotation_pattern.finditer(sentence):
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

                    rotations[i] = [
                        droll_deg * np.pi / 180.0,
                        dpitch_deg * np.pi / 180.0,
                        dyaw_deg * np.pi / 180.0,
                    ]

                grip_pattern = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)
                grip_match = grip_pattern.search(sentence)
                if "open gripper" in sentence.lower():
                    gripper_actions[i] = 1.0
                elif "close gripper" in sentence.lower():
                    gripper_actions[i] = 0.0
                elif grip_match:
                    gripper_actions[i] = float(grip_match.group(1))
                else:
                    gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0

        if self.use_eef_frame and initial_state is not None:
            actions = np.concatenate([translations, rotations, gripper_actions[:, None]], axis=1)
            actions = transform_actions_from_eef_frame(actions, initial_state)
            translations = actions[:, :3]
            rotations = actions[:, 3:6]
            gripper_actions = actions[:, 6]

        return translations, rotations, gripper_actions


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
    use_schema_format=True,
)

COMPACT_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="compact_with_rotation",
    style="compact",
    decimal_places=0,
    include_rotation=True,
    use_schema_format=True,
)

DIRECTIONAL_ONLY_FORMAT = LanguageActionFormat(
    name="directional_only",
    style="directional_only",
    decimal_places=0,
    include_rotation=False,
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
    use_schema_format=True,
)

COMPACT_BIMANUAL_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="compact_bimanual_with_rotation",
    style="compact",
    decimal_places=0,
    include_rotation=True,
    use_schema_format=True,
)

LANGUAGE_ACTION_FORMAT_REGISTRY = {
    fmt.name: fmt
    for fmt in [
        VERBOSE_FORMAT,
        VERBOSE_WITH_ROTATION_FORMAT,
        PRECISION_FORMAT,
        COMPACT_FORMAT,
        COMPACT_WITH_ROTATION_FORMAT,
        DIRECTIONAL_ONLY_FORMAT,
        EEF_FORMAT,
        EEF_WITH_ROTATION_FORMAT,
        COMPACT_BIMANUAL_FORMAT,
        COMPACT_BIMANUAL_WITH_ROTATION_FORMAT,
    ]
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]


def get_decoding_schema(schema: LanguageActionFormat | str) -> LanguageActionFormat:
    """Resolve a decoding schema to a LanguageActionFormat."""
    if isinstance(schema, LanguageActionFormat):
        return schema
    if schema not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown decoding schema: {schema}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return get_language_action_format(schema)
