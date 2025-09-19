from __future__ import annotations

import os
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi.policies import policy_config as _policy_config
import openpi.policies.policy as _policy
import openpi.transforms as transforms
from openpi.transforms import Group as _TransformGroup

from openpi_cot.policies.adapters.policy_adaptor import CoTPolicy
from openpi_cot.training.weight_loaders import _merge_params


def create_trained_policy_cot(*args, sample_kwargs: dict | None = None, **kwargs) -> CoTPolicy:
    """Build the standard policy via upstream, then wrap with CoTPolicy."""
    base = _policy_config.create_trained_policy(*args, **kwargs)
    return CoTPolicy(base, sample_kwargs=sample_kwargs)


def create_trained_policy_cot_paligemma(*args, sample_kwargs: dict | None = None, **kwargs) -> CoTPolicy:
    """Build the standard policy via upstream, then wrap with CoTPolicy.

    Additionally, if the provided checkpoint path points to a PaliGemma npz
    (e.g., contains "paligemma" or ends with ".npz"), initialize the model
    from config and merge the PaliGemma weights into it before creating the
    policy. This mirrors the logic in the training weight loader.
    """

    # Extract positional/keyword arguments matching upstream signature:
    # create_trained_policy(train_config, checkpoint_dir, *, repack_transforms=None,
    #                      sample_kwargs=None, default_prompt=None, norm_stats=None, pytorch_device=None)
    train_config = args[0] if len(args) > 0 else kwargs.get("train_config")
    checkpoint_dir = args[1] if len(args) > 1 else kwargs.get("checkpoint_dir")

    def _is_paligemma_path(p: Any) -> bool:
        if not isinstance(p, (str, os.PathLike)):
            return False
        s = str(p).lower()
        return s.endswith(".npz") or ("paligemma" in s)

    if not _is_paligemma_path(checkpoint_dir):
        # Default behavior: delegate fully to upstream.
        base = _policy_config.create_trained_policy(*args, **kwargs)
        return CoTPolicy(base, sample_kwargs=sample_kwargs)

    # Special handling for PaliGemma npz checkpoints.
    repack_transforms: _TransformGroup | None = kwargs.get("repack_transforms")
    upstream_sample_kwargs: dict[str, Any] | None = kwargs.get("sample_kwargs")
    default_prompt: str | None = kwargs.get("default_prompt")
    norm_stats = kwargs.get("norm_stats")

    if repack_transforms is None:
        repack_transforms = transforms.Group()

    # 1) Initialize model from config (JAX) and obtain full parameter tree.
    rng = jax.random.key(0)
    model = train_config.model.create(rng)
    graphdef, state = nnx.split(model)
    ref_params = state.to_pure_dict()

    # 2) Load PaliGemma npz and merge into reference params.
    #    Expect npz with flat keys like "PaliGemma/params/..." or "params/...".
    npz_path = str(checkpoint_dir)
    with open(npz_path, "rb") as f:
        flat_params = dict(np.load(f, allow_pickle=False))

    # Unflatten and map to expected subtree.
    unflat = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    if "params" in unflat:
        paligemma_params = {"PaliGemma": unflat["params"]}
    else:
        paligemma_params = {"PaliGemma": unflat}

    merged_params = _merge_params(paligemma_params, ref_params, missing_regex=".*")
    # Cast floating weights to bfloat16 to mirror upstream JAX loading behavior.
    merged_params = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x,
        merged_params,
    )

    # 3) Replace model state with merged params.
    state.replace_by_pure_dict(merged_params)
    model = nnx.merge(graphdef, state)

    # 4) Build data config and normalization stats, mirroring upstream.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # if norm_stats is None:
    #     if data_config.asset_id is None:
    #         raise ValueError("Asset id is required to load norm stats.")
    #     # There is no checkpoint assets dir for a raw npz; fall back to config assets.
    #     norm_stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)

    # 5) Assemble the Policy with the same transforms as upstream.
    base = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            # transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            # transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=upstream_sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
    return CoTPolicy(base, sample_kwargs=sample_kwargs)
