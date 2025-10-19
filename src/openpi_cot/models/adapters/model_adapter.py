from collections.abc import Sequence
import copy
import dataclasses
from enum import Enum
import logging
from typing import Generic, TypeVar

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.shared import image_tools
import openpi.shared.array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray)

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)

logger = logging.getLogger("openpi")


class ExtendedModelType(str, Enum):
    """All model types, including CoT extensions."""

    PI0 = _model.ModelType.PI0.value
    PI0_FAST = _model.ModelType.PI0_FAST.value
    PI05 = _model.ModelType.PI05.value
    PI_COT = "pi_cot"


@at.typecheck
@struct.dataclass
class CoTObservation(_model.Observation[ArrayT], Generic[ArrayT]):
    # --- CoT / vis extras (all optional) ---
    images: dict[str, at.Float[ArrayT, "*b t h w c"]]
    tokenized_langact_mask: at.Bool[ArrayT, "*b l"] | None = None
    crictical_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    sample_mask: at.Bool[ArrayT, "*b"] | None = None
    camera_intrinsics: at.Float[ArrayT, "*b t 4"] | None = None
    camera_extrinsics: at.Float[ArrayT, "*b t 4 4"] | None = None
    cartesian_position_window: at.Float[ArrayT, "*b t 6"] | None = None
    # Tokenized prediction fields (prediction_prompt + prediction_language_action combined)
    tokenized_prediction: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prediction_mask: at.Bool[ArrayT, "*b l"] | None = None
    tokenized_prediction_langact_mask: at.Bool[ArrayT, "*b l"] | None = None
    prediction_crictical_token_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "CoTObservation[ArrayT]":
        # Build the base Observation first (handles images, masks, dtype fixes, etc.)
        data_dict = dict(data)

        # For base Observation, use first frame only
        data_dict_downsampled = copy.deepcopy(data_dict)
        data_dict_downsampled["image"] = {k: v[:, 0] for k, v in data_dict["image"].items() if v is not None}

        base: _model.Observation[ArrayT] = _model.Observation.from_dict(data_dict_downsampled)

        # Pull CoT extras from either flat keys or a namespaced location.
        cot_src = data.get("extras", {}).get("cot", {})

        # Allow flat keys as well:
        def getk(k):
            return data.get(k, cot_src.get(k, None))

        # Process images: normalize to [-1, 1]
        images_processed = {}
        for k, v in data_dict["image"].items():
            if v is not None:
                # Handle both [B, H, W, C] and [B, T, H, W, C]
                images_processed[k] = v.astype(np.float32) / 255.0 * 2.0 - 1.0

        # Construct subclass using base fields
        base_dict = dataclasses.asdict(base)
        base_dict["images"] = images_processed

        return cls(
            **base_dict,
            tokenized_langact_mask=getk("tokenized_langact_mask"),
            crictical_token_mask=getk("crictical_token_mask"),
            sample_mask=getk("sample_mask"),
            camera_intrinsics=getk("camera_intrinsics"),
            camera_extrinsics=getk("camera_extrinsics"),
            cartesian_position_window=getk("cartesian_position_window"),
            tokenized_prediction=getk("tokenized_prediction"),
            tokenized_prediction_mask=getk("tokenized_prediction_mask"),
            tokenized_prediction_langact_mask=getk("tokenized_prediction_langact_mask"),
            prediction_crictical_token_mask=getk("prediction_crictical_token_mask"),
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: CoTObservation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
    aug_wrist_image: bool = True,
) -> CoTObservation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    # if not set(image_keys).issubset(observation.images):
    #     raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        if len(image.shape) == 4:
            image = image[:, None]

        b, t, h, w, c = image.shape

        # Resize if needed (before augmentation)
        if (h, w) != image_resolution:
            logger.info(f"Resizing image {key} from {(h, w)} to {image_resolution}")
            # Process each frame
            frames_resized = []
            for i in range(t):
                frame_resized = image_tools.resize_with_pad(image[:, i], *image_resolution)
                frames_resized.append(frame_resized)
            image = jnp.stack(frames_resized, axis=1)  # [b, t, h', w', c]

        if train:
            # Augmentation: apply to each frame independently
            # Flatten: [b*t, h, w, c]
            h_new, w_new = image_resolution
            image_flat = image.reshape(b * t, h_new, w_new, c)

            # Convert to [0, 1]
            image_flat = image_flat / 2.0 + 0.5

            # Build transforms
            transforms = []
            if "wrist" in key and aug_wrist_image:
                transforms += [
                    augmax.RandomCrop(int(w_new * 0.95), int(h_new * 0.95)),
                    augmax.Resize(w_new, h_new),
                    augmax.Rotate((-5, 5)),
                ]
            else:
                transforms += [
                    augmax.RandomCrop(int(w_new * 0.95), int(h_new * 0.95)),
                    augmax.Resize(w_new, h_new),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]

            # Apply augmentation
            sub_rngs = jax.random.split(rng, b * t)
            image_flat = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image_flat)

            # Back to [-1, 1]
            image_flat = image_flat * 2.0 - 1.0

            # Reshape: [b, t, h', w', c]
            image = image_flat.reshape(b, t, h_new, w_new, c)

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return CoTObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        tokenized_langact_mask=getattr(observation, "tokenized_langact_mask", None),
        crictical_token_mask=getattr(observation, "crictical_token_mask", None),
        sample_mask=getattr(observation, "sample_mask", None),
        camera_intrinsics=getattr(observation, "camera_intrinsics", None),
        camera_extrinsics=getattr(observation, "camera_extrinsics", None),
        cartesian_position_window=getattr(observation, "cartesian_position_window", None),
        tokenized_prediction=getattr(observation, "tokenized_prediction", None),
        tokenized_prediction_mask=getattr(observation, "tokenized_prediction_mask", None),
        tokenized_prediction_langact_mask=getattr(observation, "tokenized_prediction_langact_mask", None),
        prediction_crictical_token_mask=getattr(observation, "prediction_crictical_token_mask", None),
    )
