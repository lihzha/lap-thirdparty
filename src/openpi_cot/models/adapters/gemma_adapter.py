from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import openpi.models.gemma as _gemma
import openpi.shared.array_typing as at


class ModuleWithDecode(_gemma.Module):
    """Gemma Module + a decode head that ties to the embedder weights."""

    @at.typecheck
    def decode(
        self,
        embedded: at.Float[at.Array, "b _t _d"]
        | Sequence[at.Float[at.Array, "b _t _d"] | None],
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
