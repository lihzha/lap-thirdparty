import dataclasses
import re
from typing import Literal

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import is_idle_language_action
from openpi_cot.policies.utils import maybe_parse_serialized_tensor_to_ndarray
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import sum_language_actions
from openpi_cot.policies.utils import summarize_bimanual_numeric_actions
from openpi_cot.policies.utils import summarize_numeric_actions
from openpi_cot.policies.utils import to_str_list


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

    def get_sum_decimal(self) -> str:
        """Convert to legacy sum_decimal format for backward compatibility."""
        if self.style == "compact":
            return "compact"
        if self.style == "directional_only":
            return "no_number"
        # verbose
        return f"{self.decimal_places}f"


@dataclasses.dataclass(frozen=True)
class LanguageActionConfig:
    """Configuration for how to format and summarize language actions.

    This is a wrapper around LanguageActionFormat for backward compatibility.
    """

    name: str = "default"
    format: LanguageActionFormat | None = None
    # Legacy fields for backward compatibility
    sum_decimal: str | None = None
    include_rotation: bool | None = None

    def __post_init__(self):
        """Handle backward compatibility."""
        if self.format is None:
            # Create format from legacy fields
            if self.sum_decimal == "compact":
                style = "compact"
                decimal_places = 0
                use_schema_format = True
            elif self.sum_decimal == "no_number":
                style = "directional_only"
                decimal_places = 0
                use_schema_format = False
            else:
                style = "verbose"
                # Extract decimal places from format like "0f" or "2f"
                import re

                m = re.fullmatch(r"(\d+)f", self.sum_decimal or "0f")
                decimal_places = int(m.group(1)) if m else 0
                use_schema_format = False

            format_obj = LanguageActionFormat(
                name=self.name,
                style=style,
                decimal_places=decimal_places,
                include_rotation=self.include_rotation if self.include_rotation is not None else False,
                use_schema_format=use_schema_format,
            )
            object.__setattr__(self, "format", format_obj)

    def get_sum_decimal(self) -> str:
        """Get sum_decimal for backward compatibility with utils functions."""
        return self.format.get_sum_decimal()

    def get_include_rotation(self) -> bool:
        """Get include_rotation for backward compatibility."""
        return self.format.include_rotation


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

# Predefined language action configurations (backward compatible)
DEFAULT_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="default",
    format=VERBOSE_FORMAT,
)

WITH_ROTATION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="with_rotation",
    format=VERBOSE_WITH_ROTATION_FORMAT,
)

PRECISION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="precision",
    format=PRECISION_FORMAT,
)

COMPACT_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="compact",
    format=COMPACT_FORMAT,
)

COMPACT_WITH_ROTATION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="compact_with_rotation",
    format=COMPACT_WITH_ROTATION_FORMAT,
)

DIRECTIONAL_ONLY_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="directional_only",
    format=DIRECTIONAL_ONLY_FORMAT,
)

# Registry for easy lookup of language action formats
LANGUAGE_ACTION_FORMAT_REGISTRY = {
    "verbose": VERBOSE_FORMAT,
    "verbose_with_rotation": VERBOSE_WITH_ROTATION_FORMAT,
    "precision": PRECISION_FORMAT,
    "compact": COMPACT_FORMAT,
    "compact_with_rotation": COMPACT_WITH_ROTATION_FORMAT,
    "directional_only": DIRECTIONAL_ONLY_FORMAT,
}

# Registry for language action configs
LANGUAGE_ACTION_CONFIG_REGISTRY = {
    "default": DEFAULT_LANGUAGE_ACTION_CONFIG,
    "with_rotation": WITH_ROTATION_LANGUAGE_ACTION_CONFIG,
    "precision": PRECISION_LANGUAGE_ACTION_CONFIG,
    "compact": COMPACT_LANGUAGE_ACTION_CONFIG,
    "compact_with_rotation": COMPACT_WITH_ROTATION_LANGUAGE_ACTION_CONFIG,
    "directional_only": DIRECTIONAL_ONLY_LANGUAGE_ACTION_CONFIG,
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]


def get_language_action_config(name: str) -> LanguageActionConfig:
    """Get a language action config by name."""
    if name not in LANGUAGE_ACTION_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown language action config: {name}. Available configs: {list(LANGUAGE_ACTION_CONFIG_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_CONFIG_REGISTRY[name]


# TODO: during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Language action configuration (how to format/summarize actions)
    language_action_config: LanguageActionConfig = dataclasses.field(
        default_factory=lambda: COMPACT_LANGUAGE_ACTION_CONFIG
    )
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    # Prediction training parameters
    enable_prediction_training: bool = False
    prediction_prompt: str = "What is the robot's movement between two frames?"

    def _prepare_inputs(self, data: dict) -> tuple[dict, dict]:
        assert self.model_type == ExtendedModelType.PI_COT
        assert "observation" in data
        assert IMAGE_KEYS[0] in data["observation"]
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
        images = [base_image]
        image_masks = [base_image_mask]

        for k in IMAGE_KEYS[1:]:
            if k in data["observation"]:
                wrist_image = parse_image(data["observation"][k])
                wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_
            else:
                wrist_image = np.zeros_like(base_image)
                wrist_image_mask = np.False_

            # Optional dropout: randomly mask out wrist image
            if self.wrist_image_dropout_prob > 0.0 and np.random.rand() < float(self.wrist_image_dropout_prob):
                wrist_image_mask = np.False_

            images.append(wrist_image)
            image_masks.append(wrist_image_mask)

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
        }

        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"
        if isinstance(prompt, bytes):  # training time
            prompt_str = prompt.decode("utf-8")
        elif isinstance(prompt, str):  # inference time
            prompt_str = prompt
        else:
            raise ValueError(f"Prompt is not a string or bytes: {prompt}")
        inputs["prompt"] = prompt_str

        # Extract state_type if available
        # state_type = data.get("state_type")
        # if state_type is not None:
        #     if isinstance(state_type, bytes):
        #         state_type_str = state_type.decode("utf-8")
        #     elif isinstance(state_type, str):
        #         state_type_str = state_type
        #     else:
        #         state_type_str = str(state_type)
        #     inputs["state_type"] = state_type_str
        # else:
        #     # Default to "eef_pose" if not provided (for backward compatibility)
        #     inputs["state_type"] = "eef_pose"

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        return inputs

    def _prepare_text(self, data: dict, lang_action_key: str, trimmed_len_key: str) -> dict:
        la = data[lang_action_key]
        assert isinstance(la[0], bytes)
        if (
            maybe_parse_serialized_tensor_to_ndarray(la[0]) is not None
        ):  # not use json actions case. language_actions is raw action
            # Check if dataset is bimanual
            is_bimanual: bool = data.get("is_bimanual", False)

            # Only use the non-padded portion according to trimmed_len, if present
            trimmed_len: int = data.get(trimmed_len_key)
            la_used = la[:trimmed_len]
            raw_array = [maybe_parse_serialized_tensor_to_ndarray(x) for x in la_used]
            if is_bimanual:
                summed = summarize_bimanual_numeric_actions(
                    raw_array,
                    self.language_action_config.get_sum_decimal(),
                    self.language_action_config.get_include_rotation(),
                )
            else:
                summed = summarize_numeric_actions(
                    raw_array,
                    self.language_action_config.get_sum_decimal(),
                    self.language_action_config.get_include_rotation(),
                )
            return summed
        seq = to_str_list(la)
        assert seq is not None
        summed = sum_language_actions(
            seq, self.language_action_config.get_sum_decimal(), self.language_action_config.get_include_rotation()
        )
        assert summed is not None
        assert len(summed) > 0
        return summed

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)
        if self.language_action_config.get_include_rotation():
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Always prepare regular language actions for reasoning loss
        if "language_actions" in data:
            inputs["language_actions"] = self._prepare_text(data, "language_actions", "control_frequency")

            # Check if the language action represents idle movement
            is_idle = is_idle_language_action(
                inputs["language_actions"],
                self.language_action_config.get_sum_decimal(),
                self.language_action_config.get_include_rotation(),
            )
            inputs["sample_mask"] = not is_idle
        else:
            # If no language actions, default to active sample
            inputs["sample_mask"] = True

        # Additionally prepare prediction if available (independent of regular reasoning)
        if "prediction_language_action" in data:
            assert self.enable_prediction_training, (
                "Prediction language action found in data but prediction training not enabled in policy."
            )
            inputs["prediction_language_action"] = self._prepare_text(
                data, "prediction_language_action", "prediction_delta"
            )
            inputs["prediction_prompt"] = self.prediction_prompt
        else:
            assert not self.enable_prediction_training, (
                "Prediction training enabled in policy but no prediction language action found in data."
            )

        # Optional calibration/context passthroughs for visualization
        for k in ("camera_intrinsics", "camera_extrinsics"):
            if k in data["observation"]:
                inputs[k] = np.asarray(data[k])
        if "cartesian_position_window" in data["observation"]:
            inputs["cartesian_position_window"] = np.asarray(data["observation"]["cartesian_position_window"])

        return inputs


@dataclasses.dataclass(frozen=True)
class ActionDecodingSchema:
    """Defines how to decode language actions into numeric actions.

    This corresponds to the LanguageActionFormat used for encoding,
    allowing consistent round-trip conversion between actions and language.
    """

    name: str
    # Style of language action format
    style: Literal["verbose", "compact", "directional_only"] = "verbose"
    # Whether rotation is included
    include_rotation: bool = False
    # Translation unit (cm, m, mm)
    translation_unit: str = "cm"
    # Rotation unit (deg, rad)
    rotation_unit: str = "deg"
    # Schema format (for compact style)
    use_schema_format: bool = False
    # Coordinate frame axis permutation and sign (for camera frame)
    axis_perm: tuple[int, int, int] = (0, 2, 1)
    axis_sign: tuple[int, int, int] = (1, 1, 1)

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        in_camera_frame: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse language action(s) into translation deltas and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences
            in_camera_frame: Whether the output should be in camera frame coordinates

        Returns:
            (translation_deltas, gripper_actions)
            - translation_deltas: array of shape (num_steps, 3) in meters
            - gripper_actions: array of shape (num_steps,)
        """
        if isinstance(reasoning, str):
            sentences = [reasoning]
        else:
            sentences = reasoning

        num_steps = len(sentences)
        translations = np.zeros((num_steps, 3), dtype=float)
        gripper_actions = np.zeros((num_steps,), dtype=float)

        if self.use_schema_format and self.style == "compact":
            # Parse compact schema format
            if self.include_rotation:
                # Format with rotation: <+09 +09 -08 +10 -05 +15 1>
                pattern = re.compile(
                    r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, droll, dpitch, dyaw, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        gripper_actions[i] = float(grip)
                        # Note: Rotation values are parsed but not currently returned
                        # Could extend return type to include rotations if needed
            else:
                # Format without rotation: <+09 +09 -08 1>
                pattern = re.compile(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>")
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        dx, dy, dz, grip = match.groups()
                        translations[i] = [int(dx) / 100.0, int(dy) / 100.0, int(dz) / 100.0]
                        gripper_actions[i] = float(grip)
        else:
            # Parse verbose format: "move right X cm and move forward Y cm..."
            move_pattern = re.compile(
                rf"move\s+(right|left|forward|backward|back|up|down)\s+([\-\d\.]+)\s*{self.translation_unit}",
                re.IGNORECASE,
            )
            grip_pattern = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)

            for i, sentence in enumerate(sentences):
                # Parse movements in language frame (right=+x, forward=+y, up=-z)
                dx_cm = dy_cm = dz_cm = 0.0
                for match in move_pattern.finditer(sentence):
                    direction = match.group(1).lower()
                    value = float(match.group(2))
                    # if direction == "right":
                    #     dx_cm += value
                    # elif direction == "left":
                    #     dx_cm -= value
                    # elif direction == "forward":
                    #     dy_cm += value
                    # elif direction == "backward":
                    #     dy_cm -= value
                    # elif direction == "down":
                    #     dz_cm += value
                    # elif direction == "up":
                    #     dz_cm -= value
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

                # Convert to meters
                v_m = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

                # Transform to camera or robot frame if needed
                if in_camera_frame:
                    # Invert the axis permutation and sign used in encoding
                    t_cam = np.zeros(3, dtype=float)
                    axis_perm = np.array(self.axis_perm)
                    axis_sign = np.array(self.axis_sign, dtype=float)
                    sign_safe = np.where(axis_sign == 0, 1.0, axis_sign)
                    t_mapped = v_m / sign_safe
                    t_cam[axis_perm] = t_mapped
                    translations[i] = t_cam
                else:
                    translations[i] = v_m

                # Parse gripper action
                grip_match = grip_pattern.search(sentence)
                if grip_match:
                    gripper_actions[i] = float(grip_match.group(1))
                else:
                    # Maintain previous gripper state
                    gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0

        return translations, gripper_actions

    def parse_bimanual_language_to_deltas(
        self,
        reasoning: str | list[str],
        in_camera_frame: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse bimanual language action(s) into translation deltas and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences in format <L ... R ...>
            in_camera_frame: Whether the output should be in camera frame coordinates

        Returns:
            (left_translations, left_grippers, right_translations, right_grippers)
            - left_translations: array of shape (num_steps, 3) in meters
            - left_grippers: array of shape (num_steps,)
            - right_translations: array of shape (num_steps, 3) in meters
            - right_grippers: array of shape (num_steps,)
        """
        if isinstance(reasoning, str):
            sentences = [reasoning]
        else:
            sentences = reasoning

        num_steps = len(sentences)
        left_translations = np.zeros((num_steps, 3), dtype=float)
        left_grippers = np.zeros((num_steps,), dtype=float)
        right_translations = np.zeros((num_steps, 3), dtype=float)
        right_grippers = np.zeros((num_steps,), dtype=float)

        if self.use_schema_format and self.style == "compact":
            # Parse bimanual compact schema format: <L +09 +09 -08 1 R +03 -02 +01 0>
            if self.include_rotation:
                # Format with rotation: <L +09 +09 -08 +10 -05 +15 1 R +03 -02 +01 +20 +10 -05 0>
                pattern = re.compile(
                    r"<L\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)\s+"
                    r"R\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        (
                            l_dx,
                            l_dy,
                            l_dz,
                            l_droll,
                            l_dpitch,
                            l_dyaw,
                            l_grip,
                            r_dx,
                            r_dy,
                            r_dz,
                            r_droll,
                            r_dpitch,
                            r_dyaw,
                            r_grip,
                        ) = match.groups()
                        left_translations[i] = [int(l_dx) / 100.0, int(l_dy) / 100.0, int(l_dz) / 100.0]
                        left_grippers[i] = float(l_grip)
                        right_translations[i] = [int(r_dx) / 100.0, int(r_dy) / 100.0, int(r_dz) / 100.0]
                        right_grippers[i] = float(r_grip)
            else:
                # Format without rotation: <L +09 +09 -08 1 R +03 -02 +01 0>
                pattern = re.compile(
                    r"<L\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)\s+"
                    r"R\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                for i, sentence in enumerate(sentences):
                    match = pattern.search(sentence)
                    if match:
                        l_dx, l_dy, l_dz, l_grip, r_dx, r_dy, r_dz, r_grip = match.groups()
                        left_translations[i] = [int(l_dx) / 100.0, int(l_dy) / 100.0, int(l_dz) / 100.0]
                        left_grippers[i] = float(l_grip)
                        right_translations[i] = [int(r_dx) / 100.0, int(r_dy) / 100.0, int(r_dz) / 100.0]
                        right_grippers[i] = float(r_grip)
        else:
            # Parse verbose format: "Left arm: ... Right arm: ..."
            for i, sentence in enumerate(sentences):
                # Split by "Right arm:"
                parts = sentence.split("Right arm:")
                if len(parts) == 2:
                    left_part = parts[0].replace("Left arm:", "").strip()
                    right_part = parts[1].strip()

                    # Parse left arm
                    left_trans, left_grip = self.parse_language_to_deltas(left_part, in_camera_frame)
                    left_translations[i] = left_trans[0]
                    left_grippers[i] = left_grip[0]

                    # Parse right arm
                    right_trans, right_grip = self.parse_language_to_deltas(right_part, in_camera_frame)
                    right_translations[i] = right_trans[0]
                    right_grippers[i] = right_grip[0]

        return left_translations, left_grippers, right_translations, right_grippers


# Predefined decoding schemas matching language action formats
VERBOSE_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)

COMPACT_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact",
    style="compact",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_with_rotation",
    style="compact",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_BIMANUAL_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_bimanual",
    style="compact",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=True,
)

COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="compact_bimanual_with_rotation",
    style="compact",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=True,
)

DECODING_SCHEMA_REGISTRY = {
    "verbose": VERBOSE_DECODING_SCHEMA,
    "compact": COMPACT_DECODING_SCHEMA,
    "compact_with_rotation": COMPACT_WITH_ROTATION_DECODING_SCHEMA,
    "compact_bimanual": COMPACT_BIMANUAL_DECODING_SCHEMA,
    "compact_bimanual_with_rotation": COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA,
}


def get_decoding_schema(name: str) -> ActionDecodingSchema:
    """Get an action decoding schema by name."""
    if name not in DECODING_SCHEMA_REGISTRY:
        raise ValueError(f"Unknown decoding schema: {name}. Available schemas: {list(DECODING_SCHEMA_REGISTRY.keys())}")
    return DECODING_SCHEMA_REGISTRY[name]


@dataclasses.dataclass(frozen=True)
class CoTOutputs(upstream_transforms.DataTransformFn):
    # Optional decoding schema for parsing language actions to numeric actions
    decoding_schema: ActionDecodingSchema | str | None = None
    # Whether decoded actions should be in camera frame
    in_camera_frame: bool = False
    # Interpolatation steps

    def __post_init__(self):
        """Resolve string schema name to ActionDecodingSchema instance."""
        if isinstance(self.decoding_schema, str):
            schema = get_decoding_schema(self.decoding_schema)
            object.__setattr__(self, "decoding_schema", schema)

    def __call__(self, data: dict) -> dict:
        # Get actions and reasoning from data
        actions = data.get("actions")
        reasoning = data.get("reasoning")
        breakpoint()

        # If decoding schema is provided and we have reasoning, parse it to get actions
        if self.decoding_schema is not None and reasoning is not None:
            # Parse reasoning to translation deltas and gripper actions
            translations, gripper_actions = self.decoding_schema.parse_language_to_deltas(
                reasoning, in_camera_frame=self.in_camera_frame
            )

            # If we don't have actions from the model, use the parsed actions
            # Shape: (num_steps, 7) -> [dx, dy, dz, droll, dpitch, dyaw, gripper]
            # For now, assume zero rotation deltas
            num_steps = translations.shape[0]
            parsed_actions = np.concatenate(
                [
                    translations,  # (num_steps, 3)
                    np.zeros((num_steps, 3)),  # rotation deltas
                    gripper_actions[:, None],  # (num_steps, 1)
                ],
                axis=1,
            )

            if actions is None:
                actions = parsed_actions
            # Store parsed actions separately for inspection
            data["parsed_actions"] = parsed_actions

        # Only return the first 7 dims (xyz, rpy, gripper)
        if actions is not None:
            actions = np.asarray(actions[:, :7])

        return {"actions": parsed_actions, "reasoning": reasoning}
