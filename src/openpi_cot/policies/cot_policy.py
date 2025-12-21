import dataclasses
import re

import numpy as np
from openpi import transforms as upstream_transforms

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
    filter_lagre_actions: bool = False

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.language_action_format is not None and not isinstance(
            self.language_action_format, LanguageActionFormat
        ):
            schema = get_language_action_format(self.language_action_format)
            object.__setattr__(self, "language_action_format", schema)

    def _prepare_inputs(self, data: dict) -> tuple[dict, dict]:
        assert self.model_type in {ExtendedModelType.PI_COT, ExtendedModelType.PI_FAST}
        assert "observation" in data
        assert IMAGE_KEYS[0] in data["observation"]
        base_image = parse_image(data["observation"][IMAGE_KEYS[0]])
        if base_image is None:
            raise ValueError("Base image missing from observation")

        # Check if current dataset requires wrist rotation
        dataset_name = (
            data.get("dataset_name", b"").decode()
            if isinstance(data.get("dataset_name"), bytes)
            else data.get("dataset_name", "")
        )

        needs_wrist_rotation = any(ds_name in dataset_name for ds_name in DATASETS_REQUIRING_WRIST_ROTATION)
        is_prediction_sample = data.get("is_prediction_sample", False)
        pred_use_primary = data.get("pred_use_primary", False)

        images = []
        image_masks = []

        if not is_prediction_sample:
            # Training/validation: base image without rotation, wrist images with rotation if needed
            base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
            images.append(base_image)
            image_masks.append(base_image_mask)

            # Process wrist images (may need rotation)
            for k in IMAGE_KEYS[1:]:
                if k in data["observation"]:
                    wrist_image = parse_image(data["observation"][k])
                    wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_

                    # Rotate wrist image by 180 degrees for specific datasets
                    if needs_wrist_rotation and wrist_image_mask:
                        wrist_image = np.rot90(wrist_image, k=2)
                else:
                    wrist_image = np.zeros_like(base_image)
                    wrist_image_mask = np.False_

                # Optional dropout: randomly mask out wrist image
                if self.wrist_image_dropout_prob > 0.0 and np.random.rand() < float(self.wrist_image_dropout_prob):
                    wrist_image_mask = np.False_

                images.append(wrist_image)
                image_masks.append(wrist_image_mask)
        elif not pred_use_primary:
            # Prediction with secondary camera: both base and wrist images may need rotation
            for k in IMAGE_KEYS:
                if k in data["observation"]:
                    image = parse_image(data["observation"][k])
                    image_mask = np.False_ if np.all(image == 0.0) else np.True_

                    # Rotate both images by 180 degrees for specific datasets
                    if needs_wrist_rotation and image_mask:
                        image = np.rot90(image, k=2)
                else:
                    image = np.zeros_like(base_image)
                    image_mask = np.False_

                images.append(image)
                image_masks.append(image_mask)
        else:
            # Prediction with primary camera: both base and wrist images, no rotation needed
            base_image_mask = np.False_ if np.all(base_image == 0) else np.True_
            images.append(base_image)
            image_masks.append(base_image_mask)

            # Process wrist images (no rotation needed)
            for k in IMAGE_KEYS[1:]:
                if k in data["observation"]:
                    wrist_image = parse_image(data["observation"][k])
                    wrist_image_mask = np.False_ if np.all(wrist_image == 0.0) else np.True_
                else:
                    wrist_image = np.zeros_like(base_image)
                    wrist_image_mask = np.False_

                images.append(wrist_image)
                image_masks.append(wrist_image_mask)

        if self.model_type == ExtendedModelType.PI_FAST:
            image_masks = [np.True_ for _ in image_masks]

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

        if "dataset_name" in data and "r1_lite" in data["dataset_name"].decode():
            prompt_str = prompt_str.split("@")[-1]

        inputs["prompt"] = prompt_str
        if "dataset_name" in data:
            inputs["dataset_name"] = data["dataset_name"].decode()

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
        return inputs

    def _prepare_text(
        self, data: dict, lang_action_key: str, initial_state: np.ndarray = None
    ) -> tuple[dict, float | None]:
        la = data[lang_action_key]
        is_bimanual: bool = data.get("is_bimanual", False)
        is_navigation: bool = data.get("is_navigation", False)
        has_wrist_image: bool = data["has_wrist_image"]
        frame_desc = "robot base frame"

        # Transform to EEF frame if requested
        if self.language_action_format.use_eef_frame and initial_state is not None and has_wrist_image:
            la = transform_actions_to_eef_frame(la, initial_state)
            frame_desc = "end-effector frame"
        if is_bimanual:
            summed = summarize_bimanual_numeric_actions(
                la,
                self.language_action_format.get_sum_decimal(),
                self.language_action_format.include_rotation,
            )
        elif is_navigation:
            summed = summarize_numeric_actions(
                la,
                "nearest_10",
                include_rotation=True,
                rotation_precision=10,
                initial_state=initial_state,
            )
        else:
            summed = summarize_numeric_actions(
                la,
                initial_state=initial_state,
                sum_decimal=self.language_action_format.get_sum_decimal(),
                include_rotation=self.language_action_format.include_rotation,
            )
        return summed, frame_desc

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.language_action_format.use_eef_frame and "raw_state" in data:
            initial_state = np.asarray(data["raw_state"])

        inputs = self._prepare_inputs(data)
        # Check if this is a VQA dataset (e.g., coco_captions, vqa)
        dataset_name = data.get("dataset_name")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")

        is_vqa_sample = data.get("is_vqa_sample")
        inputs["is_vqa_sample"] = is_vqa_sample

        inputs["time_horizon_seconds"] = data.get("time_horizon_seconds")

        # Special handling for VQA datasets
        if is_vqa_sample:
            # For VQA, use the caption field as language_actions
            # Caption is a single string, so wrap it in a list for consistency
            if "caption" in data:
                caption = data["caption"]
                if isinstance(caption, bytes):
                    caption = caption.decode("utf-8")
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

        # Always prepare regular language actions for reasoning loss.
        if "language_actions" in data and self.enable_langact_training:
            inputs["language_actions"], inputs["frame_description"] = self._prepare_text(
                data, "language_actions", initial_state
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
                if self.filter_lagre_actions:
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
    # Whether decoded actions should be in camera frame
    in_camera_frame: bool = False
    # Interpolatation steps

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

        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.language_action_format.use_eef_frame and "state" in data:
            initial_state = np.asarray(data["state"])

        # Parse reasoning to translation deltas, rotation deltas, and gripper actions
        translations, rotations, gripper_actions = self.language_action_format.parse_language_to_deltas(
            reasoning, in_camera_frame=self.in_camera_frame, initial_state=initial_state
        )

        # If we don't have actions from the model, use the parsed actions
        # Shape: (num_steps, 7) -> [dx, dy, dz, droll, dpitch, dyaw, gripper]
        parsed_actions = np.concatenate(
            [
                translations,  # (num_steps, 3)
                rotations,  # (num_steps, 3) - rotation deltas in radians
                gripper_actions[:, None],  # (num_steps, 1)
            ],
            axis=1,
        )

        # Store parsed actions separately for inspection
        data["parsed_actions"] = parsed_actions
        print(reasoning, parsed_actions)

        return {"actions": parsed_actions, "reasoning": reasoning}
