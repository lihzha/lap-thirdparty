from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import openpi.models.gemma as _gemma
import openpi.shared.array_typing as at

from openpi_cot.models import gemma2 as _gemma2
from openpi_cot.models import gemma3 as _gemma3


class ModuleWithDecode(_gemma.Module):
    """Gemma Module + a decode head that ties to the embedder weights."""

    @at.typecheck
    def decode(
        self,
        embedded: at.Float[at.Array, "b _t _d"] | Sequence[at.Float[at.Array, "b _t _d"] | None],
    ) -> at.Float[at.Array, "b _t _v"] | list[at.Float[at.Array, "b _t _v"] | None]:
        """
        Decode hidden states to vocabulary logits using the embedder matrix.
        Supports single tensor or a list[Array|None] for multi-expert outputs.
        """

        def _decode_one(x):
            # self.embedder is created in setup(); params are shared/tied.
            logits = self.embedder.decode(x)
            # most losses expect float32 logits; cast if your pipeline needs it
            return jnp.asarray(logits, jnp.float32)

        if isinstance(embedded, (list, tuple)):
            return [_decode_one(x) if x is not None else None for x in embedded]
        return _decode_one(embedded)


class Gemma2ModuleWithDecode(_gemma2.Module):
    """Gemma2 Module + a decode head that ties to the embedder weights and applies final_logits_softcap."""

    @at.typecheck
    def decode(
        self,
        embedded: at.Float[at.Array, "b _t _d"] | Sequence[at.Float[at.Array, "b _t _d"] | None],
    ) -> at.Float[at.Array, "b _t _v"] | list[at.Float[at.Array, "b _t _v"] | None]:
        """
        Decode hidden states to vocabulary logits using the embedder matrix.
        Applies final_logits_softcap only to the first expert (index 0) if configured.
        Supports single tensor or a list[Array|None] for multi-expert outputs.
        """

        def _decode_one(x, apply_softcap=False):
            # self.embedder is created in setup(); params are shared/tied.
            logits = self.embedder.decode(x)
            # Apply final_logits_softcap only to the first expert if configured
            if apply_softcap and self.configs[0].final_logits_softcap is not None:
                softcap = self.configs[0].final_logits_softcap
                logits = jnp.tanh(logits / softcap) * softcap
            # most losses expect float32 logits; cast if your pipeline needs it
            return jnp.asarray(logits, jnp.float32)

        if isinstance(embedded, (list, tuple)):
            return [_decode_one(x, apply_softcap=(i == 0)) if x is not None else None for i, x in enumerate(embedded)]
        return _decode_one(embedded, apply_softcap=True)


class Gemma3ModuleWithDecode(_gemma3.Module):
    """Gemma2 Module + a decode head that ties to the embedder weights and applies final_logits_softcap."""

    @at.typecheck
    def decode(
        self,
        embedded: at.Float[at.Array, "b _t _d"] | Sequence[at.Float[at.Array, "b _t _d"] | None],
    ) -> at.Float[at.Array, "b _t _v"] | list[at.Float[at.Array, "b _t _v"] | None]:
        """
        Decode hidden states to vocabulary logits using the embedder matrix.
        Applies final_logits_softcap only to the first expert (index 0) if configured.
        Supports single tensor or a list[Array|None] for multi-expert outputs.
        """

        def _decode_one(x, apply_softcap=False):
            # self.embedder is created in setup(); params are shared/tied.
            logits = self.embedder.decode(x)
            # # Apply final_logits_softcap only to the first expert if configured
            # if apply_softcap and self.configs[0].final_logits_softcap is not None:
            #     softcap = self.configs[0].final_logits_softcap
            #     logits = jnp.tanh(logits / softcap) * softcap
            # most losses expect float32 logits; cast if your pipeline needs it
            return jnp.asarray(logits, jnp.float32)

        if isinstance(embedded, (list, tuple)):
            return [_decode_one(x, apply_softcap=(i == 0)) if x is not None else None for i, x in enumerate(embedded)]
        return _decode_one(embedded, apply_softcap=True)
