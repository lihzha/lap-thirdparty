import dataclasses
import re
from typing import Literal

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.models.adapters.model_adapter import IMAGE_KEYS
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
from openpi_cot.policies.utils import is_all_1s_language_action
from openpi_cot.policies.utils import is_idle_language_action
from openpi_cot.policies.utils import maybe_parse_serialized_tensor_to_ndarray
from openpi_cot.policies.utils import parse_image
from openpi_cot.policies.utils import sum_language_actions
from openpi_cot.policies.utils import summarize_bimanual_numeric_actions
from openpi_cot.policies.utils import summarize_numeric_actions
from openpi_cot.policies.utils import to_str_list
from openpi_cot.policies.utils import transform_actions_from_eef_frame
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

    def get_sum_decimal(self) -> str:
        """Convert to legacy sum_decimal format for backward compatibility."""
        if self.style == "compact":
            return "compact"
        if self.style == "directional_only":
            return "no_number"
        # verbose
        return f"{self.decimal_places}f"


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
    name="verbose", style="verbose", decimal_places=0, include_rotation=False, use_eef_frame=True
)

EEF_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="verbose", style="verbose", decimal_places=0, include_rotation=True, use_eef_frame=True
)

# Registry for easy lookup of language action formats
LANGUAGE_ACTION_FORMAT_REGISTRY = {
    "default": VERBOSE_FORMAT,  # "default" maps to "verbose" for backward compatibility
    "verbose": VERBOSE_FORMAT,
    "verbose_with_rotation": VERBOSE_WITH_ROTATION_FORMAT,
    "with_rotation": VERBOSE_WITH_ROTATION_FORMAT,  # Backward compatibility alias
    "precision": PRECISION_FORMAT,
    "compact": COMPACT_FORMAT,
    "compact_with_rotation": COMPACT_WITH_ROTATION_FORMAT,
    "directional_only": DIRECTIONAL_ONLY_FORMAT,
    "eef_frame": EEF_FORMAT,
    "eef_frame_with_rotation": EEF_WITH_ROTATION_FORMAT,
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]


# during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Language action format (how to format/summarize actions)
    language_action_format: LanguageActionFormat = dataclasses.field(default_factory=lambda: COMPACT_FORMAT)
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
    # Whether actions are in EEF frame (if True, will transform to base frame)
    use_eef_frame: bool = False

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        in_camera_frame: bool = False,
        initial_state: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse language action(s) into translation deltas, rotation deltas, and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences
            in_camera_frame: Whether the output should be in camera frame coordinates
            initial_state: Initial EEF state for EEF frame transformation (optional)

        Returns:
            (translation_deltas, rotation_deltas, gripper_actions)
            - translation_deltas: array of shape (num_steps, 3) in meters
            - rotation_deltas: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - gripper_actions: array of shape (num_steps,)
        """
        sentences = [reasoning] if isinstance(reasoning, str) else reasoning

        num_steps = len(sentences)
        translations = np.zeros((num_steps, 3), dtype=float)
        rotations = np.zeros((num_steps, 3), dtype=float)  # [roll, pitch, yaw] in radians
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
                        # Convert degrees to radians
                        rotations[i] = [
                            int(droll) * np.pi / 180.0,
                            int(dpitch) * np.pi / 180.0,
                            int(dyaw) * np.pi / 180.0,
                        ]
                        gripper_actions[i] = float(grip)
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

            if self.style == "directional_only":
                # For directional_only, accept format with optional numeric values
                # e.g., "move right" or "move right 5 cm"
                move_pattern = re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*{self.translation_unit})?",
                    re.IGNORECASE,
                )
            else:
                # For verbose formats, require explicit numeric values
                move_pattern = re.compile(
                    rf"move\s+(right|left|forward|backward|back|up|down)\s+([\-\d\.]+)\s*{self.translation_unit}",
                    re.IGNORECASE,
                )
            # Rotation pattern for verbose format
            rotation_pattern = re.compile(
                r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
                re.IGNORECASE,
            )

            for i, sentence in enumerate(sentences):
                # Parse movements in language frame (right=+x, forward=+y, up=-z)
                dx_cm = dy_cm = dz_cm = 0.0
                for match in move_pattern.finditer(sentence):
                    direction = match.group(1).lower()
                    # Default to 2cm if no numeric value provided (directional_only mode)
                    value = float(match.group(2)) if match.group(2) is not None else 2.0
                    value *= 1
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
                        # dz_cm -= 0

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

                # Parse rotation actions (if include_rotation is enabled)
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

                    # Convert degrees to radians
                    rotations[i] = [
                        droll_deg * np.pi / 180.0,
                        dpitch_deg * np.pi / 180.0,
                        dyaw_deg * np.pi / 180.0,
                    ]

                # Parse gripper action
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
                #     # Maintain previous gripper state
                #     gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0

        # Transform from EEF frame to base frame if needed
        if self.use_eef_frame and initial_state is not None:
            # Combine translations and rotations into action array
            actions = np.concatenate([translations, rotations, gripper_actions[:, None]], axis=1)
            # Transform from EEF frame to base frame
            actions = transform_actions_from_eef_frame(actions, initial_state)
            # Split back into components
            translations = actions[:, :3]
            rotations = actions[:, 3:6]
            gripper_actions = actions[:, 6]

        return translations, rotations, gripper_actions

    def parse_bimanual_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        in_camera_frame: bool = False,
        initial_state: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse bimanual language action(s) into translation deltas, rotation deltas, and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences in format <L ... R ...>
            in_camera_frame: Whether the output should be in camera frame coordinates
            initial_state: Initial EEF state for EEF frame transformation (optional)

        Returns:
            (left_translations, left_rotations, left_grippers, right_translations, right_rotations, right_grippers)
            - left_translations: array of shape (num_steps, 3) in meters
            - left_rotations: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - left_grippers: array of shape (num_steps,)
            - right_translations: array of shape (num_steps, 3) in meters
            - right_rotations: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - right_grippers: array of shape (num_steps,)
        """
        sentences = [reasoning] if isinstance(reasoning, str) else reasoning

        num_steps = len(sentences)
        left_translations = np.zeros((num_steps, 3), dtype=float)
        left_rotations = np.zeros((num_steps, 3), dtype=float)
        left_grippers = np.zeros((num_steps,), dtype=float)
        right_translations = np.zeros((num_steps, 3), dtype=float)
        right_rotations = np.zeros((num_steps, 3), dtype=float)
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
                        left_rotations[i] = [
                            int(l_droll) * np.pi / 180.0,
                            int(l_dpitch) * np.pi / 180.0,
                            int(l_dyaw) * np.pi / 180.0,
                        ]
                        left_grippers[i] = float(l_grip)
                        right_translations[i] = [int(r_dx) / 100.0, int(r_dy) / 100.0, int(r_dz) / 100.0]
                        right_rotations[i] = [
                            int(r_droll) * np.pi / 180.0,
                            int(r_dpitch) * np.pi / 180.0,
                            int(r_dyaw) * np.pi / 180.0,
                        ]
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
                    # For bimanual, initial_state should contain left arm state in first 7 or 6 elements
                    left_state = initial_state[:7] if initial_state is not None and len(initial_state) >= 7 else None
                    left_trans, left_rot, left_grip = self.parse_language_to_deltas(
                        left_part, in_camera_frame, left_state
                    )
                    left_translations[i] = left_trans[0]
                    left_rotations[i] = left_rot[0]
                    left_grippers[i] = left_grip[0]

                    # Parse right arm
                    # For bimanual, initial_state should contain right arm state starting at index 7
                    right_state = (
                        initial_state[7:14] if initial_state is not None and len(initial_state) >= 14 else None
                    )
                    right_trans, right_rot, right_grip = self.parse_language_to_deltas(
                        right_part, in_camera_frame, right_state
                    )
                    right_translations[i] = right_trans[0]
                    right_rotations[i] = right_rot[0]
                    right_grippers[i] = right_grip[0]

        # Transform from EEF frame to base frame if needed
        if self.use_eef_frame and initial_state is not None:
            # Combine left arm components
            left_actions = np.concatenate([left_translations, left_rotations, left_grippers[:, None]], axis=1)
            # For bimanual, initial_state should contain left arm state in first 7 or 6 elements
            left_state = initial_state[:7] if len(initial_state) >= 7 else initial_state[:6]
            left_actions = transform_actions_from_eef_frame(left_actions, left_state)
            left_translations = left_actions[:, :3]
            left_rotations = left_actions[:, 3:6]
            left_grippers = left_actions[:, 6]

            # Combine right arm components
            right_actions = np.concatenate([right_translations, right_rotations, right_grippers[:, None]], axis=1)
            # For bimanual, initial_state should contain right arm state starting at index 7
            right_state = initial_state[7:14] if len(initial_state) >= 14 else initial_state[7:13]
            right_actions = transform_actions_from_eef_frame(right_actions, right_state)
            right_translations = right_actions[:, :3]
            right_rotations = right_actions[:, 3:6]
            right_grippers = right_actions[:, 6]

        return left_translations, left_rotations, left_grippers, right_translations, right_rotations, right_grippers


# Predefined decoding schemas matching language action formats
VERBOSE_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)

DIRECTIONAL_SCHEMA = ActionDecodingSchema(
    name="directional",
    style="directional_only",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)


VERBOSE_EEF_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_eef",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
    use_eef_frame=True,
)

VERBOSE_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_rotation",
    style="verbose",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=False,
)

VERBOSE_EEF_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose_eef_rotation",
    style="verbose",
    include_rotation=True,
    translation_unit="cm",
    use_schema_format=False,
    use_eef_frame=True,
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
    "verbose_with_rotation": VERBOSE_WITH_ROTATION_DECODING_SCHEMA,
    "compact": COMPACT_DECODING_SCHEMA,
    "compact_with_rotation": COMPACT_WITH_ROTATION_DECODING_SCHEMA,
    "compact_bimanual": COMPACT_BIMANUAL_DECODING_SCHEMA,
    "compact_bimanual_with_rotation": COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA,
    "verbose_eef": VERBOSE_EEF_DECODING_SCHEMA,
    "verbose_eef_with_rotation": VERBOSE_EEF_WITH_ROTATION_DECODING_SCHEMA,
    "directional_only": DIRECTIONAL_SCHEMA,
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
