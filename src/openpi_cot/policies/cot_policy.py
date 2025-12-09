import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.datasets.utils.helpers import ActionEncoding
from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.lang_action_formats import VERBOSE_FORMAT
from openpi_cot.policies.lang_action_formats import LanguageActionFormat
from openpi_cot.policies.lang_action_formats import get_decoding_schema
from openpi_cot.policies.utils import is_all_1s_language_action
from openpi_cot.policies.utils import is_idle_language_action
from openpi_cot.policies.utils import maybe_parse_serialized_tensor_to_ndarray
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import sum_language_actions
from openpi_cot.policies.utils import summarize_bimanual_numeric_actions
from openpi_cot.policies.utils import summarize_numeric_actions
from openpi_cot.policies.utils import to_str_list
from openpi_cot.policies.utils import transform_actions_to_eef_frame

# Datasets that need wrist camera rotation by 180 degrees
DATASETS_REQUIRING_WRIST_ROTATION = {
    # "taco_play",
    "droid",
    # "furniture_bench_dataset_converted_externally_to_rlds",
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
    # Whether to randomly sample time horizons for language actions (training only)
    random_time_horizon: bool = True
    # Available time horizons to sample from (in seconds)
    time_horizon_options: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    # Whether to filter out samples where all movements are exactly 1 cm (likely noisy data)
    filter_all_1s_actions: bool = False
    enable_langact_training: bool = True

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
                        wrist_image = np.stack([np.rot90(img, k=2) for img in wrist_image], axis=0)
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
                        image = np.stack([np.rot90(img, k=2) for img in image], axis=0)
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
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        inputs["is_prediction_sample"] = is_prediction_sample
        return inputs

    def _prepare_text(
        self, data: dict, lang_action_key: str, initial_state: np.ndarray = None
    ) -> tuple[dict, float | None]:
        la = data[lang_action_key]
        assert isinstance(la[0], bytes)

        # Get control frequency and valid language length from data
        control_frequency = int(data.get("control_frequency", 10))  # Default to 10Hz if not present
        valid_language_length = int(data.get("valid_language_length", len(la)))

        # Determine how many language actions to use
        sampled_time_horizon = None
        if self.random_time_horizon:
            # Randomly sample a time horizon that has enough valid data
            # Compute max time horizon based on valid length
            max_time_horizon = valid_language_length / control_frequency

            # Filter time horizon options to only those with enough data
            valid_horizons = [h for h in self.time_horizon_options if h <= max_time_horizon]

            if valid_horizons:
                # Randomly sample from valid horizons
                sampled_time_horizon = float(np.random.choice(valid_horizons))
                num_steps = int(sampled_time_horizon * control_frequency)
            else:
                # If no valid horizons (e.g., very short trajectory), use all valid data
                num_steps = valid_language_length
                sampled_time_horizon = valid_language_length / control_frequency
        else:
            # Use all valid language actions (validation/inference mode)
            num_steps = min(control_frequency, valid_language_length)
            sampled_time_horizon = None

        # Ensure we don't exceed available data
        num_steps = min(num_steps, len(la))

        if (
            maybe_parse_serialized_tensor_to_ndarray(la[0]) is not None
        ):  # not use json actions case. language_actions is raw action
            # Check if dataset is bimanual
            is_bimanual: bool = data.get("is_bimanual", False)

            # Use the computed number of steps
            la_used = la[:num_steps]
            raw_array = [maybe_parse_serialized_tensor_to_ndarray(x) for x in la_used]

            # Transform to EEF frame if requested
            if self.language_action_format.use_eef_frame and initial_state is not None:
                raw_array = [transform_actions_to_eef_frame(action, initial_state) for action in raw_array]

            if is_bimanual:
                summed = summarize_bimanual_numeric_actions(
                    raw_array,
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )
            else:
                summed = summarize_numeric_actions(
                    raw_array,
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )
            return summed, sampled_time_horizon
        seq = to_str_list(la)
        assert seq is not None
        summed = sum_language_actions(
            seq, self.language_action_format.get_sum_decimal(), self.language_action_format.include_rotation
        )
        assert summed is not None
        assert len(summed) > 0
        return summed, sampled_time_horizon

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)

        # Check if this is a VQA dataset (e.g., coco_captions, vqa)
        dataset_name = data.get("dataset_name")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")

        is_vqa_sample = data.get("is_vqa_sample")
        inputs["is_vqa_sample"] = is_vqa_sample

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

        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.language_action_format.use_eef_frame and "observation" in data and "state" in data["observation"]:
            initial_state = np.asarray(data["raw_state"])

        # Always prepare regular language actions for reasoning loss.
        if "language_actions" in data and self.enable_langact_training:
            inputs["language_actions"], time_horizon_seconds = self._prepare_text(
                data, "language_actions", initial_state
            )

            # Save the time horizon in seconds if it was sampled
            if time_horizon_seconds is not None:
                inputs["time_horizon_seconds"] = time_horizon_seconds

            # Only apply idle filtering for language actions and prediction
            if not is_vqa_sample:
                is_idle = is_idle_language_action(
                    inputs["language_actions"],
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )

                # Check if all movements are exactly 1 cm (if filter is enabled)
                is_all_1s = False
                if self.filter_all_1s_actions:
                    is_all_1s = is_all_1s_language_action(
                        inputs["language_actions"],
                        self.language_action_format.get_sum_decimal(),
                        self.language_action_format.include_rotation,
                    )

                # Filter out samples that are idle OR all 1s (if enabled)
                inputs["sample_mask"] = not (is_idle or is_all_1s)
            else:
                inputs["sample_mask"] = True
        else:
            inputs["sample_mask"] = True
            # inputs["time_horizon_seconds"] = 2.0 # test time

        # Optional calibration/context passthroughs for visualization
        for k in ("camera_intrinsics", "camera_extrinsics"):
            if k in data["observation"]:
                inputs[k] = np.asarray(data[k])
        if "cartesian_position_window" in data["observation"]:
            inputs["cartesian_position_window"] = np.asarray(data["observation"]["cartesian_position_window"])

        return inputs


@dataclasses.dataclass(frozen=True)
class CoTOutputs(upstream_transforms.DataTransformFn):
    # Optional decoding schema for parsing language actions to numeric actions
    decoding_schema: LanguageActionFormat | str | None = None
    # Whether decoded actions should be in camera frame
    in_camera_frame: bool = False
    # Interpolatation steps

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.decoding_schema is not None and not isinstance(self.decoding_schema, LanguageActionFormat):
            schema = get_decoding_schema(self.decoding_schema)
            object.__setattr__(self, "decoding_schema", schema)

    def __call__(self, data: dict) -> dict:
        # Get actions and reasoning from data

        if "reasoning" not in data:
            return {"actions": np.asarray(data["actions"][:, :7]), "reasoning": None}
        reasoning = data.get("reasoning")

        # If decoding schema is provided and we have reasoning, parse it to get actions
        assert self.decoding_schema is not None
        assert reasoning is not None

        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.decoding_schema.use_eef_frame and "state" in data:
            initial_state = np.asarray(data["state"])

        # Parse reasoning to translation deltas, rotation deltas, and gripper actions
        translations, rotations, gripper_actions = self.decoding_schema.parse_language_to_deltas(
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
