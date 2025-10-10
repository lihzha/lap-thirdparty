import dataclasses

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
class LanguageActionConfig:
    """Configuration for how to format and summarize language actions.

    This allows easy extension to support different action description formats
    by defining new LanguageActionConfig instances with custom settings.
    """

    name: str = "default"
    sum_decimal: str = "0f"  # Decimal format for numeric action summaries
    include_rotation: bool = False  # Whether to include rotation in action descriptions
    # Future extensions could include:
    # - coordinate_frame: str = "robot"  # "robot", "world", "camera"
    # - unit: str = "m"  # "m", "cm", "mm"
    # - description_style: str = "natural"  # "natural", "structured", "terse"


# Predefined language action configurations
DEFAULT_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="default",
    sum_decimal="0f",
    include_rotation=False,
)

WITH_ROTATION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="with_rotation",
    sum_decimal="0f",
    include_rotation=True,
)

PRECISION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="precision",
    sum_decimal="2f",  # 2 decimal places for more precision
    include_rotation=False,
)

COMPACT_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="compact",
    sum_decimal="compact",  # Compact format: <+03 +05 -08 1>
    include_rotation=False,
)

COMPACT_WITH_ROTATION_LANGUAGE_ACTION_CONFIG = LanguageActionConfig(
    name="compact_with_rotation",
    sum_decimal="compact",  # Compact format with rotation: <+03 +05 -08 +10 +00 +02 1>
    include_rotation=True,
)


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
                else self.language_action_config.sum_decimal,
                include_rotation=self.include_rotation
                if self.include_rotation is not None
                else self.language_action_config.include_rotation,
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

    def _prepare_language_actions(self, data: dict, lang_action_key: str, trimmed_len_key: str) -> dict:
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
                    raw_array, self.language_action_config.sum_decimal, self.language_action_config.include_rotation
                )
            else:
                summed = summarize_numeric_actions(
                    raw_array, self.language_action_config.sum_decimal, self.language_action_config.include_rotation
                )
            return summed
        seq = to_str_list(la)
        assert seq is not None
        summed = sum_language_actions(
            seq, self.language_action_config.sum_decimal, self.language_action_config.include_rotation
        )
        assert summed is not None
        assert len(summed) > 0
        return summed

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)
        if self.language_action_config.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Always prepare regular language actions for reasoning loss
        inputs["language_actions"] = self._prepare_language_actions(data, "language_actions", "control_frequency")

        # Additionally prepare prediction if available (independent of regular reasoning)
        if "prediction_language_action" in data:
            assert self.enable_prediction_training, (
                "Prediction language action found in data but prediction training not enabled in policy."
            )
            inputs["prediction_language_action"] = self._prepare_language_actions(
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
