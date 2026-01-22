import dataclasses
import numpy as np
from openpi import transforms as upstream_transforms

from openpi_cot.models.model_adapter import IMAGE_KEYS
from openpi_cot.models.model_adapter import ExtendedModelType
from openpi_cot.policies.lang_action_formats import VERBOSE_FORMAT
from openpi_cot.policies.lang_action_formats import LanguageActionFormat
from openpi_cot.policies.lang_action_formats import get_language_action_format
from openpi_cot.policies.utils import parse_image



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

        add_image(base_image)
        for key in IMAGE_KEYS[1:]:
            if key in data["observation"]:
                wrist_image = parse_image(data["observation"][key])
                add_image(wrist_image, random_mask_prob=self.random_mask_prob)
            else:
                add_image(np.zeros_like(base_image), random_mask_prob=self.random_mask_prob)

        return images, image_masks

    def _prepare_inputs(self, data: dict) -> dict:
        assert "observation" in data
        # Base image may be empty for bbox samples that only use wrist image
        base_image_raw = data["observation"].get(IMAGE_KEYS[0])
       
        base_image = parse_image(base_image_raw)
        assert base_image is not None

        # Note: Wrist image rotation is now handled at the dataset level
        images, image_masks = self._collect_images(
            data, base_image
        )

        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
            "prompt": self._parse_prompt(data),
        }


        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        return inputs

    def __call__(self, data: dict) -> dict:

        inputs = self._prepare_inputs(data)
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

    def __call__(self, data: dict) -> dict:
        # Get actions and reasoning from data

        if "reasoning" not in data:
            return {"actions": np.asarray(data["actions"][:, :7]), "reasoning": None}
        reasoning = data.get("reasoning")

        # If decoding schema is provided and we have reasoning, parse it to get actions
        assert self.language_action_format is not None
        assert reasoning is not None

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
