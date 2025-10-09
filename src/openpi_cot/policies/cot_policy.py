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
        assert "exterior_image_1_left" in data["observation"]
        base_image = parse_image(data["observation"]["exterior_image_1_left"])
        if base_image is None:
            raise ValueError("Base image missing from observation")
        base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
        images = [base_image]
        image_masks = [base_image_mask]

        for k in ("wrist_image_left", "wrist_image_right"):
            if k in data["observation"]:
                wrist_image = parse_image(data["observation"][k])
                if np.all(wrist_image == 0.0):
                    wrist_image = np.zeros_like(base_image)
                    wrist_image_mask = np.False_
                else:
                    wrist_image_mask = np.True_
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

        prompt_val = data.get("prompt")
        breakpoint()
        prompt_str = None
        if prompt_val is not None:
            if isinstance(prompt_val, bytes):
                prompt_str = prompt_val.decode("utf-8")
            elif isinstance(prompt_val, str):
                prompt_str = prompt_val
            else:
                prompt_item = np.asarray(prompt_val).item()
                prompt_str = (
                    prompt_item.decode("utf-8") if isinstance(prompt_item, (bytes, np.bytes_)) else str(prompt_item)
                )
        if prompt_str is not None:
            inputs["prompt"] = prompt_str

        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        # images_for_check = {
        #     "base_0_rgb": [base_image, image_masks[0]],
        #     "left_wrist_0_rgb": [wrist_image, image_masks[1]],
        #     "right_wrist_0_rgb": [wrist_right_image, image_masks[2]],
        # }

        # if any(is_trivial_image(img, mask) for img, mask in images_for_check.values()) or (
        #     prompt_str is None or prompt_str.strip() == ""
        # ):
        #     log_payload = {
        #         "policy/anomaly_base": wandb.Image(
        #             base_image, caption=f"Dataset: {data['dataset_name'].decode('utf-8')}, prompt: {prompt_str}"
        #         )
        #         if base_image is not None
        #         else None,
        #         "policy/anomaly_wrist": wandb.Image(
        #             wrist_image,
        #             caption=f"Dataset: {data['dataset_name'].decode('utf-8')}, language actions: {inputs['language_actions']}",
        #         )
        #         if wrist_image is not None
        #         else None,
        #     }
        #     wandb.log({k: v for k, v in log_payload.items() if v is not None})
        #     logging.warning("Invalid policy inputs: trivial image or missing prompt")

        return inputs

    def _prepare_language_actions(self, data: dict) -> dict:
        if "language_actions" in data:
            la = data["language_actions"]
            breakpoint()
            assert isinstance(la[0], bytes)
            if maybe_parse_serialized_tensor_to_ndarray(la[0]) is not None:  # oxe case
                # Check if dataset is bimanual
                is_bimanual = data.get("is_bimanual", False)
                if isinstance(is_bimanual, np.ndarray):
                    is_bimanual = is_bimanual.item()

                # Only use the non-padded portion according to control_frequency, if present
                cf_val = data.get("control_frequency")
                try:
                    cf = int(np.asarray(cf_val).item()) if cf_val is not None else None
                except Exception:
                    cf = None
                if cf is not None:
                    la_used = la[: int(cf)]
                else:
                    la_used = la
                raw_array = [maybe_parse_serialized_tensor_to_ndarray(x) for x in la_used]

                # Use bimanual summarization for bimanual datasets
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
            if seq is not None:
                summed = sum_language_actions(
                    seq, self.language_action_config.sum_decimal, self.language_action_config.include_rotation
                )
                if summed is not None and len(summed) > 0:
                    return summed
            else:
                # Scalar/bytes case
                if isinstance(la, bytes):
                    la = la.decode("utf-8")
                else:
                    raise ValueError(f"Language actions is not a bytes string: {la}")
                return la
        return None

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)
        if self.language_action_config.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Always prepare regular language actions for reasoning loss
        language_actions = self._prepare_language_actions(data)
        if language_actions is not None:
            inputs["language_actions"] = language_actions

        # Additionally prepare prediction if available (independent of regular reasoning)
        if "prediction_language_action" in data:
            assert self.enable_prediction_training, (
                "Prediction language action found in data but prediction training not enabled in policy."
            )
            pred_lang = data["prediction_language_action"]

            # Handle padding: prediction_language_action has shape [summation_steps]
            # Only use the non-padded portion according to control_frequency or prediction_delta
            if isinstance(pred_lang, np.ndarray) and len(pred_lang.shape) > 0:
                # Array case: [summation_steps] with potential padding
                # Check if first element is serialized tensor (numeric case)
                breakpoint()
                if len(pred_lang) > 0 and isinstance(pred_lang[0], bytes):
                    if maybe_parse_serialized_tensor_to_ndarray(pred_lang[0]) is not None:
                        # Numeric case: use control_frequency to trim padding
                        cf_val = data.get("control_frequency")
                        try:
                            cf = int(np.asarray(cf_val).item()) if cf_val is not None else None
                        except Exception:
                            cf = None

                        # Also check prediction_delta if available (more accurate)
                        delta_val = data.get("prediction_delta")
                        try:
                            delta = int(np.asarray(delta_val).item()) if delta_val is not None else None
                        except Exception:
                            delta = None

                        # Use delta if available, otherwise control_frequency
                        trim_len = delta if delta is not None else (cf if cf is not None else len(pred_lang))
                        pred_lang_used = pred_lang[:trim_len]

                        # Parse and summarize numeric actions
                        raw_array = [maybe_parse_serialized_tensor_to_ndarray(x) for x in pred_lang_used]
                        # Filter out None values (from empty padding strings)
                        raw_array = [x for x in raw_array if x is not None]
                        if raw_array:
                            # Check if dataset is bimanual
                            is_bimanual = data.get("is_bimanual", False)
                            if isinstance(is_bimanual, np.ndarray):
                                is_bimanual = is_bimanual.item()

                            if is_bimanual:
                                pred_lang_str = summarize_bimanual_numeric_actions(
                                    raw_array,
                                    self.language_action_config.sum_decimal,
                                    self.language_action_config.include_rotation,
                                )
                            else:
                                pred_lang_str = summarize_numeric_actions(
                                    raw_array,
                                    self.language_action_config.sum_decimal,
                                    self.language_action_config.include_rotation,
                                )
                        else:
                            pred_lang_str = None
                    else:
                        # Text case: filter out empty strings (padding)
                        seq = to_str_list(pred_lang)
                        if seq is not None:
                            # Remove empty/whitespace-only strings (padding)
                            seq_trimmed = [s for s in seq if s and s.strip()]
                            pred_lang_str = sum_language_actions(
                                seq_trimmed,
                                self.language_action_config.sum_decimal,
                                self.language_action_config.include_rotation,
                            )
                        else:
                            pred_lang_str = None
                else:
                    pred_lang_str = None
            # Scalar/single value case (shouldn't happen with new format, but handle for compatibility)
            elif isinstance(pred_lang, bytes):
                pred_lang_str = pred_lang.decode("utf-8")
            elif isinstance(pred_lang, np.ndarray):
                pred_lang_item = pred_lang.item()
                pred_lang_str = (
                    pred_lang_item.decode("utf-8") if isinstance(pred_lang_item, bytes) else str(pred_lang_item)
                )
            else:
                pred_lang_str = str(pred_lang)

            # Pass through prediction language action and prompt if prediction language exists
            if pred_lang_str and pred_lang_str.strip():
                inputs["prediction_language_action"] = pred_lang_str
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
