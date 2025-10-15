# gemma3.py (Refactored for Pi0 Integration)

"""Gemma3 adaptation for Pi, separating embedding from the transformer backbone.

This version is refactored to support the Pi0 architecture by:
1.  Providing a separate `embed_prefix` method for handling VLM inputs (images, text).
2.  The main `__call__` method now accepts pre-embedded inputs, allowing for the
    injection of continuous action embeddings from the diffusion process.
3.  Implementing `AdaRMSNorm` to allow the action expert to be conditioned on the
    diffusion timestep, a critical feature for flow matching.

We follow this einsum axis naming convention:
  B: batch, T: query length, S: k/v length, N: num query heads, K: num k/v heads,
  G: num query heads per k/v head, H: head dim, D: d_model ("features")
"""

from typing import Sequence, Union, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

# Imports from the official Gemma repository
from gemma.gm.nn import _config
from gemma.gm.nn import _layers as gemma_layers
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn import _modules as gemma_modules
from gemma.gm.vision import _token_utils
from gemma.multimodal import vision as gemma_vision


# Define the type for the KV Cache for clarity
LayerCache = gemma_modules.LayerCache


# --- Step 1: Implement Adaptive RMSNorm and Gated Residual ---
# This is a critical component from the original `gemma.py` needed for Pi0.5/AdaRMS.
# It allows conditioning the normalization layers on an external vector (the time embedding).

class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for injecting conditioning information (e.g., timestep)."""
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        # Compute variance in float32 for stability.
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_x = x * jax.lax.rsqrt(var + 1e-6)

        if cond is None:
            # Standard RMSNorm: learn a single scaling parameter.
            scale = self.param("scale", nn.initializers.zeros_init(), x.shape[-1])
            output = normed_x * (1 + scale)
            gate = None
        else:
            # Adaptive RMSNorm: generate scale, shift, and gate from the condition.
            # The Dense layer projects the condition to the required dimension.
            modulation = nn.Dense(
                features=x.shape[-1] * 3,
                kernel_init=nn.initializers.zeros_init(),
                dtype=dtype,
                name="modulation_dense"
            )(cond)
            scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
            output = normed_x * (1 + scale) + shift

        return output.astype(dtype), gate


def _gated_residual(x, y, gate):
    """Applies a residual connection, gated if a gate is provided."""
    if gate is None:
        return x + y  # Standard residual connection
    return x + y * gate # Gated residual for the action expert


# --- Step 2: MoE-Aware Attention (Largely Unchanged) ---
# This module correctly operates on sequences of pre-embedded inputs.

class Gemma3MoEAttention(nn.Module):
    """
    Wraps a single Gemma 3 Attention module to handle a sequence of tensors.
    It concatenates inputs, runs one efficient attention pass, and splits the outputs.
    """
    config: _config.TransformerConfig

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:
        active_tensors = [x for x in xs if x is not None]
        if not active_tensors:
            return [None] * len(xs), kv_cache

        lengths = [x.shape[1] for x in active_tensors]
        x_concat = jnp.concatenate(active_tensors, axis=1)
        
        cfg = self.config
        
        qkv_einsum = nn.Dense(
            features=(cfg.num_heads + 2 * cfg.num_kv_heads) * cfg.head_dim,
            use_bias=False,
            name="qkv_proj",
        )
        qkv = qkv_einsum(x_concat)
        
        q, k, v = jnp.split(qkv, [cfg.num_heads * cfg.head_dim, (cfg.num_heads + cfg.num_kv_heads) * cfg.head_dim], axis=-1)

        q = einops.rearrange(q, "b t (n h) -> b t n h", n=cfg.num_heads, h=cfg.head_dim)
        k = einops.rearrange(k, "b s (k h) -> b s k h", k=cfg.num_kv_heads, h=cfg.head_dim)
        v = einops.rearrange(v, "b s (k h) -> b s k h", k=cfg.num_kv_heads, h=cfg.head_dim)
        
        # --- THIS IS THE FIX ---
        # Call the function from the correct module: `gemma_layers`
        # --- THIS IS THE FIX ---
        # Call the function from the correct, privately imported module.
        q = _positional_embeddings.apply_rope(
            q,
            positions,
            base_frequency=cfg.local_base_frequency, # Use a real value from config
        )
        k = _positional_embeddings.apply_rope(
            k,
            positions,
            base_frequency=cfg.local_base_frequency, # Use a real value from config
        )
        
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        
        num_q_per_kv = cfg.num_heads // cfg.num_kv_heads
        q_grouped = einops.rearrange(q, "b t (k g) h -> b t k g h", k=cfg.num_kv_heads, g=num_q_per_kv)

        logits = jnp.einsum("btkgh,bskh->bkgts", q_grouped, k, preferred_element_type=jnp.float32)
        
        if cfg.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / cfg.attn_logits_soft_cap) * cfg.attn_logits_soft_cap
        
        logits = logits / jnp.sqrt(cfg.head_dim)

        # --- THIS IS THE BROADCASTING FIX ---
        # The mask [B, 1, Q, K] needs to be broadcastable to logits [B, K_heads, G, Q, K]
        # We add the missing G dimension.
        final_mask = attn_mask[:, :, None, :, :] # Shape [B, 1, 1, Q, K]
        masked_logits = jnp.where(final_mask, logits, gemma_modules.K_MASK)
        
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x_concat.dtype)
        encoded_grouped = jnp.einsum("bkgts,bskh->btkgh", probs, v)
        encoded = einops.rearrange(encoded_grouped, "b t k g h -> b t (k g h)")
        out_proj = nn.Dense(features=cfg.embed_dim, use_bias=False, name="out_proj")
        final_output = out_proj(encoded)
        split_outputs = lax.split(final_output, lengths, axis=1) if lengths else []
        outputs = []
        output_idx = 0
        for x in xs:
            if x is not None:
                outputs.append(split_outputs[output_idx])
                output_idx += 1
            else: outputs.append(None)
        return outputs, (k, v)
        
        # masked_logits = jnp.where(attn_mask, logits, gemma_modules.K_MASK)
        
        # probs = jax.nn.softmax(masked_logits, axis=-1).astype(x_concat.dtype)
        
        # encoded_grouped = jnp.einsum("bkgts,bskh->btkgh", probs, v)
        # # --- THIS IS THE FIX ---
        # # 1. Rearrange the output to combine the head dimensions back into one.
        # # Shape goes from [B, T, K, G, H] -> [B, T, (K*G*H)] which is [B, T, D_model]
        # encoded = einops.rearrange(encoded_grouped, "b t k g h -> b t (k g h)")
        
        # # 2. Apply the final dense projection to the correctly shaped 3D tensor.
        # out_proj = nn.Dense(features=cfg.embed_dim, use_bias=False, name="out_proj")
        # final_output = out_proj(encoded)
        # # --- End of Fix ---
        
        # split_outputs = lax.split(final_output, lengths, axis=1) if lengths else []
        # outputs = []
        # output_idx = 0
        # for x in xs:
        #     if x is not None:
        #         outputs.append(split_outputs[output_idx])
        #         output_idx += 1
        #     else:
        #         outputs.append(None)
                
        # return outputs, (k, v)
    









        # active_tensors = [x for x in xs if x is not None]
        # if not active_tensors:
        #     return [None] * len(xs), kv_cache

        # lengths = [x.shape[1] for x in active_tensors]
        # concatenated_x = jnp.concatenate(active_tensors, axis=1)

        # # We assume the shared attention layer uses the config of the first expert.
        # attn_type = self.config.attention_types[0]
        # rope_base_frequency = (
        #     self.config.local_base_frequency
        #     if attn_type == gemma_modules.AttentionType.LOCAL_SLIDING
        #     else self.config.global_base_frequency
        # )
        
        # # Instantiate the shared Gemma 3 Attention module.
        # shared_attention_layer = gemma_modules.Attention(
        #     num_heads=self.config.num_heads,
        #     num_kv_heads=self.config.num_kv_heads,
        #     features=self.config.embed_dim,
        #     head_dim=self.config.head_dim,
        #     attn_type=attn_type,
        #     query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
        #     sliding_window_size=self.config.sliding_window_size,
        #     attn_logits_soft_cap=self.config.attn_logits_soft_cap,
        #     use_qk_norm=self.config.use_qk_norm,
        #     rope_base_frequency=rope_base_frequency,
        #     name="shared_gemma3_attention"
        # )

        # print(f"DEBUG: concatenated_x shape: {concatenated_x.shape}")
        # print(f"DEBUG: positions shape: {positions.shape}")
        # print(f"DEBUG: attn_mask shape: {attn_mask.shape}")
        # new_kv_cache, encoded = shared_attention_layer(
        #     x=concatenated_x,
        #     segment_pos=positions,
        #     cache=kv_cache,
        #     attn_mask=attn_mask,
        # )

        # # Split the outputs back to match the input sequence.
        # split_outputs = lax.split(encoded, lengths, axis=1) if lengths else []
        # outputs = []
        # output_idx = 0
        # for x in xs:
        #     if x is not None:
        #         outputs.append(split_outputs[output_idx])
        #         output_idx += 1
        #     else:
        #         outputs.append(None)
                
        # return outputs, new_kv_cache


# --- Step 3: MoE-Aware Transformer Block with AdaRMSNorm ---

class Gemma3MoEBlock(nn.Module):
    """
    The main MoE transformer block using Gemma 3 components and AdaRMSNorm.
    """
    configs: Sequence[_config.TransformerConfig] # A list of configs, one for each expert.

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        adarms_cond: Sequence[Union[jax.Array, None]],
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:

        # 1. Pre-Attention Normalization (with optional conditioning)
        normed_for_attn = []
        gates_attn = []
        for i, (x, cond) in enumerate(zip(xs, adarms_cond)):
            if x is not None:
                # Use AdaRMSNorm, which handles both cases (cond is None or not)
                normed_x, gate = AdaRMSNorm(name=f"pre_attention_norm_{i}")(x, cond)
                normed_for_attn.append(normed_x)
                gates_attn.append(gate)
            else:
                normed_for_attn.append(None)
                gates_attn.append(None)

        # 2. Shared Self-Attention
        attn_outputs, new_kv_cache = Gemma3MoEAttention(config=self.configs[0], name="moe_attention")(
            normed_for_attn, positions, attn_mask, kv_cache
        )
        
        # 3. First Gated Residual Connection
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, attn_outputs, gates_attn)]

        # 4. Pre-FFW Normalization (with optional conditioning)
        normed_for_ffw = []
        gates_ffw = []
        for i, (x, cond) in enumerate(zip(xs, adarms_cond)):
            if x is not None:
                normed_x, gate = AdaRMSNorm(name=f"pre_ffw_norm_{i}")(x, cond)
                normed_for_ffw.append(normed_x)
                gates_ffw.append(gate)
            else:
                normed_for_ffw.append(None)
                gates_ffw.append(None)

        # 5. Feed-Forward "Experts"
        mlp_outputs = []
        for i, x_normed in enumerate(normed_for_ffw):
            if x_normed is not None:
                expert_mlp = gemma_modules.FeedForward(
                    features=self.configs[i].embed_dim,
                    hidden_dim=self.configs[i].hidden_dim,
                    transpose_gating_einsum=self.configs[i].transpose_gating_einsum,
                    name=f"expert_mlp_{i}"
                )
                mlp_outputs.append(expert_mlp(x_normed))
            else:
                mlp_outputs.append(None)
        
        # 6. Second Gated Residual Connection
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, mlp_outputs, gates_ffw)]
        
        return xs, new_kv_cache


# --- Step 4: Top-Level Model with Separated Interface ---

class Gemma3MoEModel(nn.Module):
    """
    A multimodal, MoE model for Pi0 with a flexible interface.
    - `embed_prefix`: Handles vision and language embedding.
    - `__call__`: Runs the transformer backbone on pre-embedded inputs.
    """
    config: _config.TransformerConfig
    num_experts: int

    def setup(self):
        # 1. Instantiate Embedder and Vision Encoder for the `embed_prefix` method.
        self.embedder = gemma_modules.Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
            vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
            if self.config.vision_encoder
            else None,
        )
        self.vision_encoder = self.config.vision_encoder

        # 2. Create the MoE Blocks.
        expert_configs = [self.config] * self.num_experts
        self.blocks = [
            Gemma3MoEBlock(configs=expert_configs, name=f"moe_block_{i}")
            for i in range(self.config.num_layers)
        ]
        
        # 3. Create a final AdaRMSNorm for each expert path.
        self.final_norms = [
            AdaRMSNorm(name=f"final_norm_{i}") for i in range(self.num_experts)
        ]
    
    # --- ADD THIS NEW METHOD ---
    def init_all(self, dummy_images, dummy_tokens, dummy_embedded_inputs, dummy_positions, dummy_mask, dummy_adarms_cond):
        """
        A special method that calls all components of the model to ensure
        they are all included in a single, unified initialization state.
        """
        _ = self.embed_prefix(images=dummy_images, tokens=dummy_tokens)
        
        # Pass the dummy adarms_cond to trace the conditional layers
        _ = self.__call__(
            embedded_inputs=dummy_embedded_inputs,
            positions=dummy_positions,
            mask=dummy_mask,
            adarms_cond=dummy_adarms_cond,
        )

    def embed_prefix(
        self, images: jax.Array | None, tokens: jax.Array
    ) -> jax.Array:
        """
        Encapsulates the vision and text embedding logic for the VLM expert.
        Takes raw images and token IDs and returns a single embedding tensor.
        """
        text_embeddings = self.embedder.encode(tokens)

        if images is not None and self.vision_encoder is not None:
            # --- THIS IS THE FINAL FIX ---

            # 1. Manually patchify the raw 4D image tensor. This produces a 3D tensor.
            patches = self.vision_encoder.patchify_images(images)

            # 2. Manually add the "num_frames" dimension (N=1) to match the expected rank 4 input.
            # Shape goes from [B, P, D] -> [B, 1, P, D].
            patches = patches[:, None, :, :]

            # 3. Call the vision encoder with the now-correct 4D tensor and the correct keyword 'patches'.
            vision_embeds = self.vision_encoder(patches=patches, is_training=False)
            
            # --- End of Fix ---

            vision_embeds = self.embedder.encode_vision(vision_embeds)
            
            merged_embeddings = _token_utils.merge_embeddings(
                text_embeddings=text_embeddings,
                vision_embeddings=vision_embeds,
                mask=tokens == gemma_vision.TOKEN_PLACEHOLDER,
            )
            return merged_embeddings
        
        return text_embeddings

    def __call__(
        self,
        embedded_inputs: Sequence[Union[jax.Array, None]],
        positions: jnp.ndarray,
        mask: jnp.ndarray,
        adarms_cond: Sequence[Union[jax.Array, None]] | None = None,
        kv_cache: Sequence[LayerCache | None] | None = None,
    ) -> tuple[Sequence[Union[jax.Array, None]], Sequence[LayerCache | None]]:
        """
        Runs the transformer backbone on a sequence of pre-embedded inputs.
        This allows mixing VLM embeddings with continuous action embeddings.
        """
        if kv_cache is None:
            kv_cache = [None] * self.config.num_layers
        if adarms_cond is None:
            adarms_cond = [None] * self.num_experts
            
        xs = embedded_inputs
        new_kv_caches = []
        
        # Run through MoE Transformer Blocks
        for i, block in enumerate(self.blocks):
            xs, new_cache = block(
                xs,
                positions=positions,
                attn_mask=mask,
                adarms_cond=adarms_cond,
                kv_cache=kv_cache[i]
            )
            new_kv_caches.append(new_cache)

        # Final Normalization per Expert
        outputs = []
        for i, (x, cond) in enumerate(zip(xs, adarms_cond)):
            if x is not None:
                # We only need the normalized output here, not the gate.
                normed_x, _ = self.final_norms[i](x, cond)
                outputs.append(normed_x)
            else:
                outputs.append(None)
        
        return outputs, new_kv_caches