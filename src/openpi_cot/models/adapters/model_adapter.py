from collections.abc import Sequence
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
    image: dict[str, at.Float[ArrayT, "*b t h w c"]] | None = None
    tokenized_reasoning_mask: at.Bool[ArrayT, "*b l"] | None = None
    tokenized_numeric_mask: at.Bool[ArrayT, "*b l"] | None = None
    example_mask: at.Bool[ArrayT, "*b"] | None = None
    camera_intrinsics: at.Float[ArrayT, "*b t 4"] | None = None
    camera_extrinsics: at.Float[ArrayT, "*b t 4 4"] | None = None
    cartesian_position_window: at.Float[ArrayT, "*b t 6"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "CoTObservation[ArrayT]":
        # Build the base Observation first (handles images, masks, dtype fixes, etc.)
        data_dict = dict(data)
        data_dict["image"] = {k: v[:, 0] for k, v in data_dict["image"].items() if v is not None}
        base: _model.Observation[ArrayT] = _model.Observation.from_dict(data_dict)
        # Pull CoT extras from either flat keys or a namespaced location.
        cot_src = data.get("extras", {}).get("cot", {})

        # Allow flat keys as well:
        def getk(k):
            return data.get(k, cot_src.get(k, None))

        # Construct subclass using base fields
        base_dict = dataclasses.asdict(base)
        base_dict["image"] = getk("image")
        return cls(
            **base_dict,
            tokenized_reasoning_mask=getk("tokenized_reasoning_mask"),
            tokenized_numeric_mask=getk("tokenized_numeric_mask"),
            example_mask=getk("example_mask"),
            camera_intrinsics=getk("camera_intrinsics"),
            camera_extrinsics=getk("camera_extrinsics"),
            cartesian_position_window=getk("cartesian_position_window"),
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: CoTObservation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
    aug_wrist: bool = False,
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
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if not aug_wrist and "wrist" in key:
                pass
            else:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

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
        tokenized_reasoning_mask=observation.tokenized_reasoning_mask,
        tokenized_numeric_mask=getattr(observation, "tokenized_numeric_mask", None),
        example_mask=getattr(observation, "example_mask", None),
        camera_intrinsics=getattr(observation, "camera_intrinsics", None),
        camera_extrinsics=getattr(observation, "camera_extrinsics", None),
        cartesian_position_window=getattr(observation, "cartesian_position_window", None),
    )
