import dataclasses
from typing import Literal

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
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
        default_factory=lambda: DEFAULT_LANGUAGE_ACTION_CONFIG
    )
    # Legacy fields for backward compatibility - will use language_action_config if not None
    sum_decimal: str | None = None
    include_rotation: bool | None = None
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    # Prediction training parameters
    enable_prediction_training: bool = False
    prediction_prompt: str = "What is the robot's movement between two frames?"

    def __post_init__(self):
        # Handle backward compatibility: if sum_decimal or include_rotation are set,
        # create a custom LanguageActionConfig
        if self.sum_decimal is not None or self.include_rotation is not None:
            config = LanguageActionConfig(
                name="custom",
                sum_decimal=self.sum_decimal
                if self.sum_decimal is not None
                else self.language_action_config.get_sum_decimal(),
                include_rotation=self.include_rotation
                if self.include_rotation is not None
                else self.language_action_config.get_include_rotation(),
            )
            # Use object.__setattr__ because this is a frozen dataclass
            object.__setattr__(self, "language_action_config", config)

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
        state_type = data.get("state_type")
        if state_type is not None:
            if isinstance(state_type, bytes):
                state_type_str = state_type.decode("utf-8")
            elif isinstance(state_type, str):
                state_type_str = state_type
            else:
                state_type_str = str(state_type)
            inputs["state_type"] = state_type_str
        else:
            # Default to "eef_pose" if not provided (for backward compatibility)
            inputs["state_type"] = "eef_pose"

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        return inputs

    def _prepare_text(self, data: dict, lang_action_key: str, trimmed_len_key: str) -> dict:
        assert lang_action_key in data
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
        inputs["language_actions"] = self._prepare_text(data, "language_actions", "control_frequency")

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
class CoTOutputs(upstream_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
