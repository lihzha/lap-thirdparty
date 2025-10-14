"""
Gemma 3 implementation with a "mixture of experts" style input structure.

This module is designed to function like the original `gemma2.py` but uses the
canonical, official components from the Gemma 3 source files.
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from gemma.gm.nn._modules import AttentionType, Block as OfficialGemma3Block, Embedder as OfficialGemma3Embedder
from gemma.gm.nn._layers import RMSNorm as OfficialGemma3RMSNorm
from gemma.gm.nn._config import TransformerConfig as OfficialGemma3Config

# --- Official Gemma 3 model classes (used for their configs) copied from `_gemma.py` ---
from gemma.gm.nn._gemma import (
    Gemma3_270M, Gemma3_1B, Gemma3_4B, Gemma3_12B, Gemma3_27B
)

# Define KVCache for type hinting, consistent with your original file.
KVCache = list[dict[str, jax.Array]]

Variant = Literal["gemma3_270m", "gemma3_1b", "gemma3_4b", "gemma3_12b", "gemma3_27b"]

def get_config(variant: Variant) -> OfficialGemma3Config:
  """Returns the configuration for the specified Gemma 3 variant."""
  if variant == "gemma3_270m":
    return Gemma3_270M.config
  elif variant == "gemma3_1b":
    return Gemma3_1B.config
  elif variant == "gemma3_4b":
    return Gemma3_4B.config
  elif variant == "gemma3_12b":
    return Gemma3_12B.config
  elif variant == "gemma3_27b":
    return Gemma3_27B.config
  else:
    raise ValueError(f"Unknown variant: {variant}")
  
def _gated_residual(x, y, gate):
    """
    Helper for the gated residual connection, same as in your gemma2.py.
    This is applied *after* the main transformer block output.
    """
    if x is None:
        return None
    # The official block already adds the main residual (y += x).
    # To implement the gate, we subtract the original input and add the gated version.
    # new_x = (x + y)  becomes  new_x = x + gate * y
    # So, y_gated = (y + x) - x + gate * (y) -> This is not quite right.
    # The official block returns `outputs + attn_output`. The original `x` is `attn_output`.
    # Let's re-read the official block. `outputs += attn_output` is the residual.
    # Okay, let's assume the official block returns `y` and we need to compute `x + gate*y`
    if gate is None:
        return x + y
    return x + gate * y


class AdaptiveRMSNorm(nn.Module):
    """
    Wrapper around the official `RMSNorm` to support the adaptive conditioning
    (`adarms_cond`) from your original `RMSNormBF16`.
    """
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array | None):
        # Use the official RMSNorm for the core normalization.
        normed_x = OfficialGemma3RMSNorm()(x)

        gate = None
        # If a condition is provided, apply the adaptive modulation.
        if cond is not None:
            modulation = nn.Dense(
                features=x.shape[-1] * 3,
                kernel_init=nn.initializers.zeros,
                dtype=x.dtype,
                param_dtype=self.param_dtype,
            )(cond)
            scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
            normed_x = normed_x * (1 + scale) + shift

        return normed_x, gate
    

class Module(nn.Module):
    """
    Gemma 3 Transformer model supporting a mixture of experts. This version
    is built using the canonical components from the official Gemma 3 library.
    """
    configs: Sequence[OfficialGemma3Config]
    embed_dtype: str
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        assert all(c.num_layers == self.configs[0].num_layers for c in self.configs)
        param_dtype = self.dtype

        self.embedder = OfficialGemma3Embedder(
            vocab_size=self.configs[0].num_embed,
            embed_dim=self.configs[0].embed_dim,
            name="embedder",
        )

        self.expert_blocks = [
            [OfficialGemma3Block(
                name=f'expert_{i}_layer_{j}',
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                embed_dim=config.embed_dim,
                head_dim=config.head_dim,
                hidden_dim=config.hidden_dim,
                use_post_attn_norm=config.use_post_attn_norm,
                use_post_ffw_norm=config.use_post_ffw_norm,
                attn_type=attn_type,
                query_pre_attn_scalar=config.query_pre_attn_scalar(),
                transpose_gating_einsum=config.transpose_gating_einsum,
                attn_logits_soft_cap=config.attn_logits_soft_cap,
                sliding_window_size=config.sliding_window_size,
                use_qk_norm=config.use_qk_norm,
            ) for j, attn_type in enumerate(config.attention_types)]
            for i, config in enumerate(self.configs)
        ]

        self.final_norms = [
            AdaptiveRMSNorm(param_dtype=param_dtype, name=f'final_norm_{i}')
            for i in range(len(self.configs))
        ]

    def embed(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def __call__(
        self,
        embedded: Sequence[jnp.ndarray | None],
        positions: jnp.ndarray,
        mask: jnp.ndarray,
        adarms_cond: Sequence[jnp.ndarray | None] | None = None,
        *,
        kv_cache: list[dict] | None = None, # The cache is a list of expert caches
        deterministic: bool = True,
    ) -> tuple[Sequence[jnp.ndarray | None], list[dict]]:

        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        
        # If no cache is provided, create a list of Nones for each expert.
        if kv_cache is None:
            kv_cache = [None] * len(self.configs)

        drop = nn.Dropout(self.dropout) if self.dropout > 0 else lambda x, **kwargs: x
        
        # This will hold the final hidden states for each expert
        final_expert_outputs = [None] * len(self.configs)
        # This will hold the updated K/V caches for each expert
        next_kv_cache_all_experts = []

        # --- RESTRUCTURED LOGIC: Loop over experts first ---
        for i in range(len(self.configs)):
            expert_input = embedded[i]
            
            # If this expert is not active for this batch, skip it.
            if expert_input is None:
                next_kv_cache_all_experts.append(kv_cache[i]) # Pass through the old cache
                continue

            # This is the cache for this specific expert (a dictionary of layer caches)
            expert_cache_dict = kv_cache[i] or {}
            new_expert_cache_dict = {}
            
            # Now, thread the hidden state through all layers for this expert
            x = expert_input
            for j in range(self.configs[i].num_layers):
                block = self.expert_blocks[i][j]
                layer_name = f'expert_{i}_layer_{j}'
                
                layer_cache = expert_cache_dict.get(layer_name)

                updated_layer_cache, x = block(
                    x,
                    positions,
                    layer_cache,
                    mask,
                )
                
                x = drop(x, deterministic=deterministic)
                new_expert_cache_dict[layer_name] = updated_layer_cache
            
            # After all layers, `x` is the final hidden state for this expert
            final_expert_outputs[i] = x
            next_kv_cache_all_experts.append(new_expert_cache_dict)
        
        # Finally, apply the final normalization to the outputs we collected.
        final_normalized_outputs = [
            self.final_norms[i](x, a)[0] if x is not None else None
            for i, (x, a) in enumerate(zip(final_expert_outputs, adarms_cond))
        ]

        return final_normalized_outputs, next_kv_cache_all_experts