import dataclasses
import random
import re

import numpy as np
from openpi import transforms as upstream_transforms
from openpi.models.model import ModelType

from openpi_cot.datasets.utils.helpers import ActionEncoding
from openpi_cot.models.model_adapter import IMAGE_KEYS
from openpi_cot.models.model_adapter import ExtendedModelType
from openpi_cot.policies.lang_action_formats import VERBOSE_FORMAT
from openpi_cot.policies.lang_action_formats import LanguageActionFormat
from openpi_cot.policies.lang_action_formats import get_language_action_format
from openpi_cot.policies.utils import describe_language_action_scale
from openpi_cot.policies.utils import is_all_1s_language_action
from openpi_cot.policies.utils import is_idle_language_action
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import summarize_bimanual_numeric_actions
from openpi_cot.policies.utils import summarize_numeric_actions
from openpi_cot.policies.utils import transform_actions_to_eef_frame

# Note: DATASETS_REQUIRING_WRIST_ROTATION has been moved to
# openpi_cot.datasets.utils.helpers and rotation is now applied at dataset level


# during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Language action format (how to format/summarize actions)
    language_action_format: LanguageActionFormat = dataclasses.field(default_factory=lambda: VERBOSE_FORMAT)
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    # Determines which model will be used.
    model_type: ExtendedModelType = ExtendedModelType.PI_COT
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    enable_langact_training: bool = True
    filter_all_1s_actions: bool = False
    use_rough_scale: bool = False
    filter_large_actions: bool = False
    stateless_gripper: bool = True
    random_base_prob: float = 0.0
    random_mask_prob: float = 0.0
    # DEPRECATED: not_rotate_wrist_prob is now handled at dataset level in prepare_batched_dataset
    # This parameter is kept for backward compatibility but has no effect
    not_rotate_wrist_prob: float = 0.0

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.language_action_format is not None and not isinstance(
            self.language_action_format, LanguageActionFormat
        ):
            schema = get_language_action_format(self.language_action_format)
            object.__setattr__(self, "language_action_format", schema)

    @staticmethod
    def _decode_text(value, default: str = "") -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return default

    def _dataset_name(self, data: dict) -> str:
        return self._decode_text(data.get("dataset_name"), default="")

    def _parse_prompt(self, data: dict) -> str:
        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"
        prompt_str = self._decode_text(prompt, default="")
        dataset_name = self._dataset_name(data)
        if "r1_lite" in dataset_name:
            prompt_str = prompt_str.split("@")[-1]
        return prompt_str

    @staticmethod
    def _image_mask(image: np.ndarray, random_mask_prob: float = 0.0) -> np.bool_:
        if np.all(image == 0.0):
            if random_mask_prob > 0.0 and np.random.rand() < random_mask_prob:
                return np.True_
            return np.False_
        return np.True_

    def _collect_images(
        self,
        data: dict,
        base_image: np.ndarray,
        is_prediction_sample: bool,
        pred_use_primary: bool,
        is_vqa_sample: bool = False,
    ) -> tuple[list[np.ndarray], list[np.bool_]]:
        """Collect images for model input.
        
        Note: Wrist image rotation is now handled at the dataset level in restructure functions.
        This method no longer applies rotation - images are already rotated if needed.
        """
        images: list[np.ndarray] = []
        image_masks: list[np.bool_] = []

        def add_image(image: np.ndarray, random_mask_prob: float = 0.0) -> None:
            image_mask = self._image_mask(image, random_mask_prob=random_mask_prob)
            images.append(image)
            image_masks.append(image_mask)

        if not is_prediction_sample:
            add_image(base_image)
            for key in IMAGE_KEYS[1:]:
                if key in data["observation"]:
                    wrist_image = parse_image(data["observation"][key])
                    # Skip random operations for VQA samples - they need deterministic processing
                    if not is_vqa_sample and self.wrist_image_dropout_prob > 0.0 and np.random.rand() < float(self.wrist_image_dropout_prob):
                        wrist_image = np.zeros_like(base_image)
                    # Skip random_mask_prob for VQA samples
                    effective_random_mask_prob = 0.0 if is_vqa_sample else self.random_mask_prob
                    add_image(wrist_image, random_mask_prob=effective_random_mask_prob)
                else:
                    effective_random_mask_prob = 0.0 if is_vqa_sample else self.random_mask_prob
                    add_image(np.zeros_like(base_image), random_mask_prob=effective_random_mask_prob)
        elif not pred_use_primary:
            for key in IMAGE_KEYS:
                if key in data["observation"]:
                    image = parse_image(data["observation"][key])
                    add_image(image)
                else:
                    add_image(np.zeros_like(base_image))
        else:
            add_image(base_image)
            for key in IMAGE_KEYS[1:]:
                if key in data["observation"]:
                    wrist_image = parse_image(data["observation"][key])
                    add_image(wrist_image)
                else:
                    add_image(np.zeros_like(base_image))

        return images, image_masks

    def _prepare_inputs(self, data: dict) -> dict:
        assert self.model_type in {ExtendedModelType.PI_COT, ExtendedModelType.PI_FAST, ModelType.PI0_FAST}
        assert "observation" in data
        # Base image may be empty for bbox samples that only use wrist image
        base_image_raw = data["observation"].get(IMAGE_KEYS[0])
        # Handle empty string (for bbox samples using only wrist image)
        if isinstance(base_image_raw, (str, bytes)) and len(base_image_raw) == 0:
            base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            base_image = parse_image(base_image_raw)
            if base_image is None:
                # Create a zero image as fallback (will be masked out)
                base_image = np.zeros((224, 224, 3), dtype=np.uint8)

        dataset_name = self._dataset_name(data)
        is_prediction_sample = data.get("is_prediction_sample", False)
        pred_use_primary = data.get("pred_use_primary", False)
        is_vqa_sample = data.get("is_vqa_sample", False)

        # Note: Wrist image rotation is now handled at the dataset level
        images, image_masks = self._collect_images(
            data, base_image, is_prediction_sample, pred_use_primary, is_vqa_sample
        )

        if self.model_type == ExtendedModelType.PI_FAST:
            image_masks = [np.True_ for _ in image_masks]

        # names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        # # We don't mask out padding images for FAST models.
        # images = (base_image, np.zeros_like(base_image), wrist_image)
        # image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
        }

        # inputs = {
        #     "state": data["observation"]["state"],
        #     "image": dict(zip(names, images, strict=True)),
        #     "image_mask": dict(zip(names, image_masks, strict=True)),
        # }

        inputs["prompt"] = self._parse_prompt(data)
        if dataset_name:
            inputs["dataset_name"] = dataset_name

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
            # if self.language_action_format.use_eef_frame and initial_state is not None:
            # actions = transform_actions_to_eef_frame(data["actions"], initial_state)
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        inputs["is_prediction_sample"] = is_prediction_sample
        
        # Get rotation_applied flag from data (set by decode function at dataset level)
        # This indicates whether wrist image rotation was actually applied
        rotation_applied = data["rotation_applied"]
        return inputs, rotation_applied

    def _summarize_language_actions(
        self, data: dict, lang_action_key: str, initial_state: np.ndarray = None, dataset_name=None, rotation_applied=False
    ) -> tuple[str | None, str]:
        language_actions = data[lang_action_key]
        is_bimanual: bool = data.get("is_bimanual", False)
        is_navigation: bool = data.get("is_navigation", False)
        has_wrist_image: bool = data["has_wrist_image"]
        frame_description = "robot base frame"

        use_eef_frame = self.language_action_format.use_eef_frame and initial_state is not None
        if self.random_base_prob > 0.0:
            use_eef_frame = use_eef_frame and has_wrist_image and random.random() < (1-self.random_base_prob)

        # Transform to EEF frame if requested
        # rotation_applied indicates whether wrist image was rotated at dataset level
        if use_eef_frame:
            language_actions = transform_actions_to_eef_frame(language_actions, initial_state, dataset_name, rotation_applied)
            frame_description = "end-effector frame"
        if is_bimanual:
            summed = summarize_bimanual_numeric_actions(
                language_actions,
                self.language_action_format.get_sum_decimal(),
                self.language_action_format.include_rotation,
            )
        elif is_navigation:
            summed = summarize_numeric_actions(
                language_actions,
                "nearest_10",
                include_rotation=True,
                rotation_precision=10,
                initial_state=initial_state,
                stateless_gripper=True,
            )
        else:
            summed = summarize_numeric_actions(
                language_actions,
                initial_state=initial_state,
                sum_decimal=self.language_action_format.get_sum_decimal(),
                include_rotation=self.language_action_format.include_rotation,
                stateless_gripper=self.stateless_gripper,
            )
        return summed, frame_description

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        # Extract initial state for EEF frame transformation

        inputs, rotation_applied = self._prepare_inputs(data)
        # Check if this is a VQA dataset (e.g., coco_captions, vqa)
        dataset_name = self._dataset_name(data)

        is_vqa_sample = data.get("is_vqa_sample")
        inputs["is_vqa_sample"] = is_vqa_sample

        inputs["time_horizon_seconds"] = data.get("time_horizon_seconds")
        
        # Pass through vqa_dataset_id for per-dataset metrics tracking
        vqa_dataset_id = data.get("vqa_dataset_id", 0)
        inputs["vqa_dataset_id"] = vqa_dataset_id

        # Special handling for VQA datasets
        if is_vqa_sample:
            # For VQA, use the caption field as language_actions
            # Caption is a single string, so wrap it in a list for consistency
            if "caption" in data:
                caption = data["caption"]
                caption = self._decode_text(caption, default="")
                # Store caption as single-element list for consistency
                inputs["language_actions"] = caption
            else:
                # Fallback if caption is missing
                inputs["language_actions"] = ""

            # VQA samples are always active (no idle filtering)
            inputs["sample_mask"] = True

            return inputs

        # Regular robot dataset processing
        if self.language_action_format.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Handle VLA-0 format: use normalized actions field directly (not language_actions)
        if self.language_action_format.style == "vla0" and self.enable_langact_training:
            # VLA-0 uses the normalized actions field directly
            if "actions" in inputs:
                # actions are already normalized to [-1, 1] and contain action_horizon timesteps
                normalized_actions = inputs["actions"]
                # Use VLA0ActionFormat.summarize_actions() to convert to text
                inputs["language_actions"] = self.language_action_format.summarize_actions(normalized_actions)
                inputs["frame_description"] = "normalized"
            else:
                inputs["language_actions"] = ""
                inputs["frame_description"] = "normalized"

            # VLA-0 doesn't filter idle actions - all samples are active
            inputs["sample_mask"] = True

        # Handle other formats: use language_actions field
        elif "language_actions" in data and self.enable_langact_training:
            initial_state = np.asarray(data["raw_state"])

            inputs["language_actions"], inputs["frame_description"] = self._summarize_language_actions(
                data, "language_actions", initial_state, dataset_name=dataset_name, rotation_applied=rotation_applied
            )
            if self.use_rough_scale:
                inputs["language_actions"] = describe_language_action_scale(inputs["language_actions"])

            # Only apply idle filtering for language actions and prediction
            if not is_vqa_sample and not self.use_rough_scale:
                is_idle = is_idle_language_action(
                    inputs["language_actions"],
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )
                if self.filter_large_actions:
                    has_large_action = False
                    if isinstance(inputs["language_actions"], str):
                        values = re.findall(r"[-+]?\d+(?:\.\d+)?", inputs["language_actions"])
                        has_large_action = any(abs(float(val)) >= 10.0 for val in values)
                else:
                    has_large_action = False

                # Check if all movements are exactly 1 cm (if filter is enabled)
                is_all_1s = False
                if self.filter_all_1s_actions:
                    is_all_1s = is_all_1s_language_action(
                        inputs["language_actions"],
                        self.language_action_format.get_sum_decimal(),
                        self.language_action_format.include_rotation,
                    )

                # Filter out samples that are idle OR all 1s (if enabled)
                inputs["sample_mask"] = (not has_large_action) or not (is_idle or is_all_1s)
            else:
                inputs["sample_mask"] = True
        else:
            inputs["sample_mask"] = True

        return inputs


@dataclasses.dataclass(frozen=True)
class CoTOutputs(upstream_transforms.DataTransformFn):
    # Optional decoding schema for parsing language actions to numeric actions
    language_action_format: LanguageActionFormat | str | None = None
    # Optional norm_stats for VLA0 format unnormalization
    # VLA0 outputs actions in normalized [-1, 1] space that need unnormalization
    norm_stats: dict | None = None
    normalization_type: str = "bounds_q99"

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.language_action_format is not None and not isinstance(
            self.language_action_format, LanguageActionFormat
        ):
            schema = get_language_action_format(self.language_action_format)
            object.__setattr__(self, "language_action_format", schema)

    def _unnormalize_vla0_actions(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize VLA0 actions from [-1, 1] to physical space."""
        if self.norm_stats is None:
            return actions
        
        actions_stats = self.norm_stats.get("actions")
        if actions_stats is None:
            return actions
        
        if self.normalization_type == "bounds_q99":
            q01 = getattr(actions_stats, "q01", None)
            q99 = getattr(actions_stats, "q99", None)
            if q01 is None or q99 is None:
                return actions
            # Unnormalize from [-1, 1] to original space
            # actions in [-1, 1] -> (actions + 1) / 2 * (q99 - q01) + q01
            q01 = np.asarray(q01)
            q99 = np.asarray(q99)
            # Handle dimension mismatch - only unnormalize dims that have stats
            dim = min(q01.shape[-1], actions.shape[-1])
            unnormed = (actions[..., :dim] + 1.0) / 2.0 * (q99[..., :dim] - q01[..., :dim] + 1e-6) + q01[..., :dim]
            if actions.shape[-1] > dim:
                # Keep extra dims as-is (e.g., already normalized gripper)
                unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)
            return unnormed
        elif self.normalization_type in ("bounds", "normal"):
            # Handle other normalization types if needed
            if self.normalization_type == "bounds":
                min_val = getattr(actions_stats, "min", None)
                max_val = getattr(actions_stats, "max", None)
                if min_val is None or max_val is None:
                    return actions
                min_val = np.asarray(min_val)
                max_val = np.asarray(max_val)
                dim = min(min_val.shape[-1], actions.shape[-1])
                unnormed = (actions[..., :dim] + 1.0) / 2.0 * (max_val[..., :dim] - min_val[..., :dim] + 1e-8) + min_val[..., :dim]
                if actions.shape[-1] > dim:
                    unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)
                return unnormed
            else:  # normal
                mean = getattr(actions_stats, "mean", None)
                std = getattr(actions_stats, "std", None)
                if mean is None or std is None:
                    return actions
                mean = np.asarray(mean)
                std = np.asarray(std)
                dim = min(mean.shape[-1], actions.shape[-1])
                unnormed = actions[..., :dim] * (std[..., :dim] + 1e-6) + mean[..., :dim]
                if actions.shape[-1] > dim:
                    unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)
                return unnormed
        return actions

    def __call__(self, data: dict) -> dict:
        # Get actions and reasoning from data

        if "reasoning" not in data:
            return {"actions": np.asarray(data["actions"][:, :7]), "reasoning": None}
        reasoning = data.get("reasoning")

        # If decoding schema is provided and we have reasoning, parse it to get actions
        assert self.language_action_format is not None
        assert reasoning is not None

        # Handle VLA-0 format: parse to full action array
        if self.language_action_format.style == "vla0":
            # VLA-0 format returns full action array [action_horizon, action_dim]
            from openpi_cot.policies.lang_action_formats import VLA0ActionFormat

            if isinstance(self.language_action_format, VLA0ActionFormat):
                actions = self.language_action_format.parse_to_full_actions(reasoning)
                # VLA0 actions are in normalized [-1, 1] space, unnormalize them
                actions = self._unnormalize_vla0_actions(actions)
                return {"actions": actions, "reasoning": reasoning}
            # Fallback for generic VLA0-style format
            movement, gripper_action = self.language_action_format.parse_language_to_deltas(reasoning)
            single_action = np.concatenate([movement, [gripper_action]]) if gripper_action is not None else movement
            return {"actions": single_action, "reasoning": reasoning}

        # Handle other formats: parse to movement deltas + gripper
        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.language_action_format.use_eef_frame and "raw_state" in data:
            initial_state = np.asarray(data["raw_state"])

        # Parse reasoning to translation deltas, rotation deltas, and gripper actions
        movement, gripper_action = self.language_action_format.parse_language_to_deltas(
            reasoning, initial_state=initial_state
        )

        # If we don't have actions from the model, use the parsed actions
        # Shape: (num_steps, 7) -> [dx, dy, dz, droll, dpitch, dyaw, gripper]
        single_action = np.concatenate([movement, [gripper_action]]) if gripper_action is not None else movement

        # Store parsed actions separately for inspection
        print(reasoning, gripper_action)

        return {"actions": single_action, "reasoning": reasoning}
