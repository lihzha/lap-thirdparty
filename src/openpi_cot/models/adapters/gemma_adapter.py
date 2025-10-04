from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import openpi.models.gemma as _gemma
import openpi.shared.array_typing as at

Variant = Literal[
    "dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "paligemma2_3b", "paligemma2_300m"
]


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


def get_extended_config(variant: _gemma.Variant | str) -> _gemma.Config:
    """Return Gemma config, extended to include larger PaliGemma 2 variants.

    Falls back to upstream get_config for known variants.
    Supported extras:
      - "paligemma2_10b": Approx Gemma2-9B text backbone used by PaliGemma 2 10B

    Notes:
      The values are derived from public Gemma 2 9B settings commonly used:
        width (d_model)= 3072
        depth (n_layers)= 42
        num_heads = 24
        num_kv_heads = 8
        head_dim = 128   (so 24*128 = 3072)
        mlp_dim = 8192   (approximate intermediate size)
    """
    # Known upstream variants
    try:
        return _gemma.get_config(variant)  # type: ignore[arg-type]
    except Exception:
        pass

    v = str(variant)
    if v == "paligemma2_3b":
        return _gemma.Config(
            width=2304,
            depth=26,
            mlp_dim=9216,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
        )
    if v == "paligemma2_300m":
        return _gemma.Config(
            width=1024,
            depth=26,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
        )

    raise ValueError(f"Unknown variant: {variant}")
