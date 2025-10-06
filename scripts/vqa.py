from collections.abc import Iterator
import dataclasses
import enum
import logging

import numpy as np

try:
    import tensorflow_datasets as tfds  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    tfds = None  # type: ignore

from openpi.policies import policy as _policy
from tqdm import tqdm
import tyro

import openpi_cot.policies.adapters.policy_config_adapter as _policy_config
from openpi_cot.training import config as _config

try:
    import tensorflow_datasets as tfds  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    tfds = None  # type: ignore
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

import openpi_cot.training.weight_loaders as _weight_loaders


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = "what is in the image?"

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # DROID dataloader options (used when env == EnvMode.DROID)
    droid_max_examples: int = 100


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid_pytorch",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def init_model(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
) -> Any:
    rng, model_rng = jax.random.split(init_rng)
    # initialize the model (and its parameters).
    model = config.model.create(model_rng)

    # Get the state and load partial params
    graphdef, state = nnx.split(model)
    partial_params = _load_weights_and_validate(config.weight_loader, state.to_pure_dict())

    # Replace the state with partial params (this modifies state in place)
    state.replace_by_pure_dict(partial_params)

    # Merge the updated state back into the model
    model = nnx.merge(graphdef, state)

    # Apply frozen param conversion to bfloat16 (replicating train.py behavior)
    params = nnx.state(model)
    params = nnx_utils.state_map(
        params,
        config.freeze_filter,
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )

    # Update the model with the converted params
    nnx.update(model, params)

    return model


def create_model(config):
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    model = init_model(config, init_rng)
    return model


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy_cot(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args, model=None) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if model is not None:
        return _policy_config.create_trained_policy_from_model(
            model, _config.get_config(args.policy.config), default_prompt=args.default_prompt
        )
    return _policy_config.create_trained_policy_cot(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )


def _iter_droid_request_data(
    data_dir: str, split: str, droid_dataset_name: str, *, prompt: str | None = None
) -> Iterator[dict]:
    """Yield request_data dicts from the DROID TFDS dataset.

    Produces keys compatible with `DroidInputs` and `Policy.vqa_infer`.
    """
    if tfds is None:
        raise ImportError("tensorflow_datasets is required for DROID loading but is not installed.")

    ds = tfds.load(droid_dataset_name, data_dir=data_dir, split=split, shuffle_files=False)

    for example in ds:  # Eager iteration; fields are tf.Tensors
        step = next(iter(example["steps"]))

        base_img_t = step["observation"]["exterior_image_1_left"]
        wrist_img_t = step["observation"]["wrist_image_left"]

        def to_np(elem, to_uint8=True):
            arr = elem.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
            if arr.dtype != np.uint8 and to_uint8:
                arr = (
                    (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr
                )
            return arr

        base_img = to_np(base_img_t)
        wrist_img = to_np(wrist_img_t)

        # State
        cartesian_pos = to_np(step["observation"]["cartesian_position"], to_uint8=False)
        joint_pos = to_np(step["observation"]["joint_position"], to_uint8=False)
        grip_obs = to_np(step["observation"]["gripper_position"], to_uint8=False)

        req = {
            "observation": {
                "exterior_image_1_left": base_img,
                "wrist_image_left": wrist_img,
                "state": np.concatenate([cartesian_pos, grip_obs]),
            },
        }
        # req["prompt"] = prompt
        req["prompt"] = step["language_instruction"]
        yield req


def main(args: Args) -> None:
    config: _config.TrainConfig = _config.get_config(args.policy.config)
    model = create_model(config)
    policy = create_policy(args, model=model)

    if tfds is None:
        raise ImportError("Please install tensorflow_datasets to use the DROID dataloader.")

    prompt = args.default_prompt or "what is in the image?"
    for idx, req in enumerate(
        tqdm(
            _iter_droid_request_data(config.data.rlds_data_dir, "all", config.data.droid_dataset_name, prompt=prompt),
            total=args.droid_max_examples,
            desc="DROID samples",
        )
    ):
        outputs = policy.vqa_infer(req)
        print({"request_keys": list(req.keys()), "prompt": req["prompt"], "text": outputs.get("reasoning")})
        if idx + 1 >= args.droid_max_examples:
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
