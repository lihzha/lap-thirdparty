import dataclasses
import re
from typing import Literal

import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.vqa_base import VQA_DATASET_NAMES
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
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]


# TODO: during inference, inputs need to be converted to the same encoding as the model first, normalize, and then convert to robot-acceptable encoding.
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

        # Datasets that need wrist camera rotation by 180 degrees
        DATASETS_REQUIRING_WRIST_ROTATION = {
            # "taco_play",
            "droid",
            # "furniture_bench_dataset_converted_externally_to_rlds",
            "berkeley_fanuc_manipulation",
            "berkeley_autolab_ur5",
            "fmb",
        }

        # Check if current dataset requires wrist rotation
        dataset_name = (
            data.get("dataset_name", b"").decode()
            if isinstance(data.get("dataset_name"), bytes)
            else data.get("dataset_name", "")
        )
        needs_wrist_rotation = any(ds_name in dataset_name for ds_name in DATASETS_REQUIRING_WRIST_ROTATION)

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

        if "dataset_name" in data:
            if "r1_lite" in data["dataset_name"].decode():
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
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )
            else:
                summed = summarize_numeric_actions(
                    raw_array,
                    self.language_action_format.get_sum_decimal(),
                    self.language_action_format.include_rotation,
                )
            return summed
        seq = to_str_list(la)
        assert seq is not None
        summed = sum_language_actions(
            seq, self.language_action_format.get_sum_decimal(), self.language_action_format.include_rotation
        )
        assert summed is not None
        assert len(summed) > 0
        return summed

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        inputs = self._prepare_inputs(data)

        # Check if this is a VQA dataset (e.g., coco_captions, vqa)
        dataset_name = data.get("dataset_name")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")
        is_vqa_dataset = dataset_name in VQA_DATASET_NAMES
        inputs["is_vqa_mask"] = False  # Default to False

        # Special handling for VQA datasets
        if is_vqa_dataset:
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

            # VQA datasets should not participate in prediction training
            inputs["is_vqa_mask"] = True

            # Skip prediction training for VQA
            # (VQA datasets don't have temporal structure for prediction)

            return inputs

        # Regular robot dataset processing
        if self.language_action_format.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Always prepare regular language actions for reasoning loss
        if "language_actions" in data:
            inputs["language_actions"] = self._prepare_text(data, "language_actions", "control_frequency")

            # Check if the language action represents idle movement
            is_idle = is_idle_language_action(
                inputs["language_actions"],
                self.language_action_format.get_sum_decimal(),
                self.language_action_format.include_rotation,
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

        # import wandb; wandb.log({"image": [wandb.Image(inputs["image"]["base_0_rgb"][0]), wandb.Image(inputs["image"]["base_0_rgb"][1])]})

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse language action(s) into translation deltas, rotation deltas, and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences
            in_camera_frame: Whether the output should be in camera frame coordinates

        Returns:
            (translation_deltas, rotation_deltas, gripper_actions)
            - translation_deltas: array of shape (num_steps, 3) in meters
            - rotation_deltas: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - gripper_actions: array of shape (num_steps,)
        """
        if isinstance(reasoning, str):
            sentences = [reasoning]
        else:
            sentences = reasoning

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
                        elif rotation_type in {"tilt up", "tilt forward"}:
                            dpitch_deg += value
                        elif rotation_type in {"tilt down", "tilt back"}:
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
                # grip_match = grip_pattern.search(sentence)
                if "open gripper" in sentence.lower():
                    gripper_actions[i] = 1.0
                elif "close gripper" in sentence.lower():
                    gripper_actions[i] = 0.0
                else:
                    gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0
                # if grip_match:
                #     gripper_actions[i] = float(grip_match.group(1))
                # else:
                #     # Maintain previous gripper state
                #     gripper_actions[i] = gripper_actions[i - 1] if i > 0 else 0.0

        return translations, rotations, gripper_actions

    def parse_bimanual_language_to_deltas(
        self,
        reasoning: str | list[str],
        in_camera_frame: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse bimanual language action(s) into translation deltas, rotation deltas, and gripper actions.

        Args:
            reasoning: Single sentence or list of reasoning sentences in format <L ... R ...>
            in_camera_frame: Whether the output should be in camera frame coordinates

        Returns:
            (left_translations, left_rotations, left_grippers, right_translations, right_rotations, right_grippers)
            - left_translations: array of shape (num_steps, 3) in meters
            - left_rotations: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - left_grippers: array of shape (num_steps,)
            - right_translations: array of shape (num_steps, 3) in meters
            - right_rotations: array of shape (num_steps, 3) in radians [roll, pitch, yaw]
            - right_grippers: array of shape (num_steps,)
        """
        if isinstance(reasoning, str):
            sentences = [reasoning]
        else:
            sentences = reasoning

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
                    left_trans, left_rot, left_grip = self.parse_language_to_deltas(left_part, in_camera_frame)
                    left_translations[i] = left_trans[0]
                    left_rotations[i] = left_rot[0]
                    left_grippers[i] = left_grip[0]

                    # Parse right arm
                    right_trans, right_rot, right_grip = self.parse_language_to_deltas(right_part, in_camera_frame)
                    right_translations[i] = right_trans[0]
                    right_rotations[i] = right_rot[0]
                    right_grippers[i] = right_grip[0]

        return left_translations, left_rotations, left_grippers, right_translations, right_rotations, right_grippers


# Predefined decoding schemas matching language action formats
VERBOSE_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose",
    style="verbose",
    include_rotation=False,
    translation_unit="cm",
    use_schema_format=False,
)

VERBOSE_WITH_ROTATION_DECODING_SCHEMA = ActionDecodingSchema(
    name="verbose",
    style="verbose",
    include_rotation=True,
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
    "verbose_with_rotation": VERBOSE_WITH_ROTATION_DECODING_SCHEMA,
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

        if "reasoning" not in data:
            return {"actions": np.asarray(data["actions"][:, :7]), "reasoning": None}
        reasoning = data.get("reasoning")

        # If decoding schema is provided and we have reasoning, parse it to get actions
        assert self.decoding_schema is not None and reasoning is not None
        # Parse reasoning to translation deltas, rotation deltas, and gripper actions
        translations, rotations, gripper_actions = self.decoding_schema.parse_language_to_deltas(
            reasoning, in_camera_frame=self.in_camera_frame
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
