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

# Datasets that need wrist camera rotation by 180 degrees
DATASETS_REQUIRING_WRIST_ROTATION = {
    # "taco_play",
    "droid",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "berkeley_fanuc_manipulation",
    "berkeley_autolab_ur5",
    "fmb",
}


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
        needs_wrist_rotation: bool,
        is_prediction_sample: bool,
        pred_use_primary: bool,
    ) -> tuple[list[np.ndarray], list[np.bool_]]:
        images: list[np.ndarray] = []
        image_masks: list[np.bool_] = []

        def add_image(image: np.ndarray, apply_rotation: bool, random_mask_prob: float = 0.0) -> None:
            image_mask = self._image_mask(image, random_mask_prob=random_mask_prob)
            if apply_rotation and image_mask:
                image = np.rot90(image, k=2)
            images.append(image)
            image_masks.append(image_mask)
        
        need_flip_ee_frame = False

        if not is_prediction_sample:
            add_image(base_image, apply_rotation=False)
            for key in IMAGE_KEYS[1:]:
                if key in data["observation"]:
                    wrist_image = parse_image(data["observation"][key])
                    if self.wrist_image_dropout_prob > 0.0 and np.random.rand() < float(self.wrist_image_dropout_prob):
                        wrist_image = np.zeros_like(base_image)
                    actual_apply_rotation = needs_wrist_rotation and not (np.random.rand() < self.not_rotate_wrist_prob)
                    if actual_apply_rotation != need_flip_ee_frame:
                        need_flip_ee_frame = True
                    add_image(wrist_image, apply_rotation=actual_apply_rotation, random_mask_prob=self.random_mask_prob)
                else:
                    add_image(np.zeros_like(base_image), apply_rotation=False, random_mask_prob=self.random_mask_prob)
        elif not pred_use_primary:
            for key in IMAGE_KEYS:
                if key in data["observation"]:
                    image = parse_image(data["observation"][key])
                    actual_apply_rotation = needs_wrist_rotation and not (np.random.rand() < self.not_rotate_wrist_prob)
                    if actual_apply_rotation != need_flip_ee_frame:
                        need_flip_ee_frame = True
                    add_image(image, apply_rotation=actual_apply_rotation)
                else:
                    add_image(np.zeros_like(base_image), apply_rotation=False)
        else:
            add_image(base_image, apply_rotation=False)
            for key in IMAGE_KEYS[1:]:
                if key in data["observation"]:
                    wrist_image = parse_image(data["observation"][key])
                    add_image(wrist_image, apply_rotation=False)
                else:
                    add_image(np.zeros_like(base_image), apply_rotation=False)

        return images, image_masks, need_flip_ee_frame

    def _prepare_inputs(self, data: dict) -> dict:
        assert self.model_type in {ExtendedModelType.PI_COT, ExtendedModelType.PI_FAST, ModelType.PI0_FAST}
        assert "observation" in data
        assert IMAGE_KEYS[0] in data["observation"]
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")

        dataset_name = self._dataset_name(data)
        needs_wrist_rotation = any(ds_name in dataset_name for ds_name in DATASETS_REQUIRING_WRIST_ROTATION)
        is_prediction_sample = data.get("is_prediction_sample", False)
        pred_use_primary = data.get("pred_use_primary", False)

        images, image_masks, need_flip_ee_frame = self._collect_images(
            data, base_image, needs_wrist_rotation, is_prediction_sample, pred_use_primary
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
        return inputs, need_flip_ee_frame

    def _summarize_language_actions(
        self, data: dict, lang_action_key: str, initial_state: np.ndarray = None, dataset_name=None, need_flip_ee_frame=False
    ) -> tuple[str | None, str]:
        language_actions = data[lang_action_key]
        is_bimanual: bool = data.get("is_bimanual", False)
        is_navigation: bool = data.get("is_navigation", False)
        has_wrist_image: bool = data["has_wrist_image"]
        frame_description = "robot base frame"

        use_eef_frame = self.language_action_format.use_eef_frame and initial_state is not None
        if self.random_base_prob > 0.0:
            use_eef_frame = use_eef_frame and has_wrist_image and random.random() < self.random_base_prob

        # Transform to EEF frame if requested
        if use_eef_frame:
            language_actions = transform_actions_to_eef_frame(language_actions, initial_state, dataset_name, need_flip_ee_frame)
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

        inputs, need_flip_ee_frame = self._prepare_inputs(data)
        # Check if this is a VQA dataset (e.g., coco_captions, vqa)
        dataset_name = self._dataset_name(data)

        is_vqa_sample = data.get("is_vqa_sample")
        inputs["is_vqa_sample"] = is_vqa_sample

        inputs["time_horizon_seconds"] = data.get("time_horizon_seconds")

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
                data, "language_actions", initial_state, dataset_name=dataset_name, need_flip_ee_frame=need_flip_ee_frame
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

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.language_action_format is not None and not isinstance(
            self.language_action_format, LanguageActionFormat
        ):
            schema = get_language_action_format(self.language_action_format)
            object.__setattr__(self, "language_action_format", schema)

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
