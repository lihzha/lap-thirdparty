#!/usr/bin/env python3
import argparse
from pathlib import Path

import augmax
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import image_tools
from PIL import Image

from openpi_cot.models.model_adapter import CoTObservation

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)


def preprocess_observation(
    rng,
    observation: CoTObservation,
    *,
    train: bool = False,
    image_keys=IMAGE_KEYS,
    image_resolution: tuple[int, int] = (224, 224),
    aug_wrist_image: bool = True,
    vqa_mask=None,
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

        b, h, w, c = image.shape
        if train:
            if "wrist" in key and aug_wrist_image:
                image = image / 2.0 + 0.5
                rng, crop_rng = jax.random.split(rng)
                crop_h_frac = jax.random.uniform(crop_rng, (), minval=0.65, maxval=0.9)
                transforms = [
                    augmax.RandomCrop(int(w * 0.9), int(h * crop_h_frac)),
                    augmax.Resize(w, h),
                    augmax.Rotate((-10, 10)),
                    augmax.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
                sub_rngs = jax.random.split(rng, b)
                image_aug = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
                # Skip augmentation for VQA samples if vqa_mask is provided
                image = (
                    jnp.where(vqa_mask[:, None, None, None], image, image_aug) if vqa_mask is not None else image_aug
                )

                # Back to [-1, 1]
                image = image * 2.0 - 1.0
                if (h, w) != image_resolution:
                    # Process each frame
                    image = image_tools.resize_with_pad(image, *image_resolution)

            else:
                # Resize if needed (before augmentation)
                if (h, w) != image_resolution:
                    # Process each frame
                    image = image_tools.resize_with_pad(image, *image_resolution)

                # Augmentation: apply to each frame independently
                # Flatten: [b*t, h, w, c]
                h_new, w_new = image_resolution

                # Convert to [0, 1]
                image = image / 2.0 + 0.5

                # Build transforms
                transforms = [
                    augmax.RandomCrop(int(w_new * 0.9), int(h_new * 0.99)),
                    augmax.Resize(w_new, h_new),
                    augmax.Rotate((-5, 5)),
                    augmax.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]

                # Apply augmentation
                sub_rngs = jax.random.split(rng, b)
                image_aug = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

                # Skip augmentation for VQA samples if vqa_mask is provided
                image = (
                    jnp.where(vqa_mask[:, None, None, None], image, image_aug) if vqa_mask is not None else image_aug
                )

                # Back to [-1, 1]
                image = image * 2.0 - 1.0
        # Resize if needed (before augmentation)
        elif (h, w) != image_resolution:
            # Process each frame
            image = image_tools.resize_with_pad(image, *image_resolution)

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
        critical_token_mask=getattr(observation, "critical_token_mask", None),
        number_token_mask=getattr(observation, "number_token_mask", None),
        direction_token_mask=getattr(observation, "direction_token_mask", None),
        sample_mask=getattr(observation, "sample_mask", None),
        tokenized_dataset_name=getattr(observation, "tokenized_dataset_name", None),
        is_vqa_sample=getattr(observation, "is_vqa_sample", None),
        is_prediction_sample=getattr(observation, "is_prediction_sample", None),
    )


def _load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _to_model_range(image_u8: np.ndarray) -> np.ndarray:
    return image_u8.astype(np.float32) / 255.0 * 2.0 - 1.0


def _to_u8(image: np.ndarray) -> np.ndarray:
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0)
    return image.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Apply preprocess_observation augmentation to a single image and save the result.")
    )
    parser.add_argument("input", type=Path, help="Path to the input image.")
    parser.add_argument("output", type=Path, help="Path to save the augmented image.")
    parser.add_argument(
        "--key",
        default="left_wrist_0_rgb",
        help="Image key used by preprocess_observation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for augmentation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Target width for preprocessing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Target height for preprocessing.",
    )
    parser.add_argument(
        "--no-wrist-aug",
        action="store_true",
        help="Disable wrist-specific augmentation branch.",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Also save the input image (after resize) next to the output.",
    )
    args = parser.parse_args()

    for i in range(100):
        image_u8 = _load_rgb(args.input)
        image = _to_model_range(image_u8)[None, ...]

        observation = CoTObservation(
            images={args.key: jnp.asarray(image)},
            image_masks={},
            state=jnp.zeros((1, 1), dtype=jnp.float32),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
            token_ar_mask=None,
            token_loss_mask=None,
        )
        rng = jax.random.PRNGKey(args.seed + i)

        processed = preprocess_observation(
            rng,
            observation,
            train=True,
            image_keys=(args.key,),
            image_resolution=(args.height, args.width),
            aug_wrist_image=not args.no_wrist_aug,
            vqa_mask=None,
        )

        augmented = np.array(processed.images[args.key][0])
        Image.fromarray(_to_u8(augmented)).save(args.output.with_name(f"{args.output.stem}_{i}{args.output.suffix}"))

        if args.save_input:
            input_processed = preprocess_observation(
                rng,
                observation,
                train=False,
                image_keys=(args.key,),
                image_resolution=(args.height, args.width),
                aug_wrist_image=not args.no_wrist_aug,
                vqa_mask=None,
            )
            input_image = np.array(input_processed.images[args.key][0])
            input_path = args.output.with_name(f"{args.output.stem}_input{args.output.suffix}_{i}")
            Image.fromarray(_to_u8(input_image)).save(input_path)


if __name__ == "__main__":
    main()
