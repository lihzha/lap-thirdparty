from __future__ import annotations

from typing import Any

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
import openpi.policies.policy as _policy
from openpi.training import config as _config
import openpi.transforms as transforms

from openpi_cot.policies.adapters.policy_adaptor import CoTPolicy


def create_trained_policy_cot(*args, sample_kwargs: dict | None = None, **kwargs) -> CoTPolicy:
    """Build the standard policy via upstream, then wrap with CoTPolicy."""
    base = _policy_config.create_trained_policy(*args, **kwargs)
    return CoTPolicy(base, sample_kwargs=sample_kwargs)


def create_trained_policy_from_model(
    model: _model.BaseModel,
    train_config: _config.TrainConfig,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    base = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
    return CoTPolicy(base, sample_kwargs=sample_kwargs)
