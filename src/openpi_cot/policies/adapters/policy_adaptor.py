import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import openpi.policies.policy as _policy
from openpi.shared import nnx_utils

from openpi_cot.models.adapters.model_adapter import CoTObservation


class CoTPolicy:
    """A policy that uses Chain of Thought (CoT) reasoning."""

    def __init__(self, base: _policy.Policy, *, sample_kwargs: dict[str, Any] | None = None):
        self._base = base
        assert hasattr(base._model, "sample_reasoning"), "Model must have a sample_reasoning method"  # noqa: SLF001
        self._sample_reasoning = nnx_utils.module_jit(base._model.sample_reasoning)  # noqa: SLF001

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def infer_reasoning(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._base._input_transform(inputs)  # noqa: SLF001
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        breakpoint()

        start_time = time.monotonic()
        self._rng, _ = jax.random.split(self._base._rng)  # noqa: SLF001
        logits, t = self._sample_reasoning(CoTObservation.from_dict(inputs))
        outputs = {
            "state": inputs["state"],
            "actions": jnp.zeros((1, 1, 32)),  # TODO
            "reasoning_logits": logits,
            "final_length": t,
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        # outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._base._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        return self.infer_reasoning(obs)

    def vqa_infer(self, obs: dict) -> dict:
        """Run VQA inference using the VLM backbone.

        Expects `obs` to contain images and a tokenized prompt compatible with the
        PaliGemma tokenizer. Returns generated text.
        """

        return self.infer_reasoning(obs)
