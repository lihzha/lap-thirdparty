from __future__ import annotations

from openpi.policies import policy_config as _policy_config

from openpi_cot.policies.adapters.policy_adaptor import CoTPolicy


def create_trained_policy_cot(*args, sample_kwargs: dict | None = None, **kwargs) -> CoTPolicy:
    """Build the standard policy via upstream, then wrap with CoTPolicy."""
    base = _policy_config.create_trained_policy(*args, **kwargs)
    return CoTPolicy(base, sample_kwargs=sample_kwargs)
