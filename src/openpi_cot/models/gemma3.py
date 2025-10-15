"""Gemma3 adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

import dataclasses
from typing import Literal, TypeAlias, Sequence, Union

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


from gemma.gm.ckpts import _paths
from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
from gemma.gm.vision import _token_utils # We need this for vision merging
from gemma.multimodal import vision as gemma_vision

_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_270M = 18
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62


GEMMA3_ATTENTION_PATTERN = (
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.GLOBAL,
)


# Define the type for the KV Cache for clarity
LayerCache = _modules.LayerCache

# --- Step 2: Create the MoE-Aware Attention Wrapper ---
# This class implements the crucial "concatenate-compute-split" pattern.
class Gemma3MoEAttention(nn.Module):
    """
    Wraps a single Gemma 3 Attention module to handle a sequence of tensors.
    It works by concatenating inputs, running one efficient attention pass,
    and splitting the outputs back.
    """
    config: _config.TransformerConfig

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jnp.array,
        attn_mask: jnp.array,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:
        
        # --- 1. CONCATENATE ---
        active_tensors = [x for x in xs if x is not None]
        if not active_tensors:
            return [None] * len(xs), kv_cache

        lengths = [x.shape[1] for x in active_tensors]
        concatenated_x = jnp.concatenate(active_tensors, axis=1)

        # --- 2. COMPUTE (This section is now corrected) ---
        
        # We assume the shared attention layer uses the config of the first layer
        attn_type = self.config.attention_types[0]
        
        # Gemma 3 uses different RoPE frequencies for local vs global attention.
        # We need to replicate that logic here.
        rope_base_frequency = (
            self.config.local_base_frequency
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency
        )
        
        # Instantiate the shared Gemma 3 Attention module, now passing ALL the
        # necessary parameters from the config object.
        shared_attention_layer = _modules.Attention(
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            features=self.config.embed_dim,
            head_dim=self.config.head_dim,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            
            # --- THESE ARE THE CRITICAL ADDITIONS ---
            sliding_window_size=self.config.sliding_window_size,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            use_qk_norm=self.config.use_qk_norm,
            rope_base_frequency=rope_base_frequency,
            # ----------------------------------------
            
            name="shared_gemma3_attention"
        )
        
        # Call the Gemma 3 attention module. This part remains the same.
        new_kv_cache, encoded = shared_attention_layer(
            x=concatenated_x,
            segment_pos=positions,
            cache=kv_cache,
            attn_mask=attn_mask,
        )

        # --- 3. SPLIT --- (This part remains the same)
        # We must also handle the case where there is only one active tensor,
        # as lax.split requires at least one section.
        if len(lengths) > 0:
            split_outputs = lax.split(encoded, lengths, axis=1)
        else:
            split_outputs = []

        outputs = []
        output_idx = 0
        for x in xs:
            if x is not None:
                outputs.append(split_outputs[output_idx])
                output_idx += 1
            else:
                outputs.append(None)
                
        return outputs, new_kv_cache

# --- Step 3: Create the MoE-Aware Transformer Block ---
# This class contains the shared MoE Attention and a list of MLP "experts".
class Gemma3MoEBlock(nn.Module):
    """
    The main MoE transformer block using Gemma 3 components.
    It follows the logic from `gemma2.py` but uses `_modules` and `_layers`.
    """
    configs: Sequence[_config.TransformerConfig] # A list of configs, one for each expert.

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jnp.array,
        attn_mask: jnp.array,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:
        # --- 1. Shared Self-Attention Part ---
        # MoE Logic: Normalize each tensor in the sequence separately.
        normed_for_attn = [
            _layers.RMSNorm(name=f"pre_attention_norm_{i}")(x) if x is not None else None
            for i, x in enumerate(xs)
        ]

        # Call our MoE-aware attention wrapper.
        # We use the config of the first expert for the shared attention parameters.
        attn_outputs, new_kv_cache = Gemma3MoEAttention(config=self.configs[0], name="moe_attention")(
            normed_for_attn, positions, attn_mask, kv_cache
        )
        
        # MoE Logic: First residual connection for each tensor in the sequence.
        # This matches the `attn_output += x` line in `_modules.Block`.
        xs = [x + y if x is not None else y for x, y in zip(xs, attn_outputs)]
        
        # --- 2. Feed-Forward "Experts" Part ---
        # MoE Logic: Normalize each tensor separately before the MLP.
        normed_for_ffw = [
            _layers.RMSNorm(name=f"pre_ffw_norm_{i}")(x) if x is not None else None
            for i, x in enumerate(xs)
        ]

        # *** CORE MoE LOGIC ***
        # Apply a different MLP (expert) to each tensor in the sequence.
        mlp_outputs = []
        for i, x_normed in enumerate(normed_for_ffw):
            if x_normed is not None:
                # Instantiate and apply the i-th expert MLP using its specific config.
                # This creates separate, trainable weights for each expert.
                expert_mlp = _modules.FeedForward(
                    features=self.configs[i].embed_dim,
                    hidden_dim=self.configs[i].hidden_dim,
                    transpose_gating_einsum=self.configs[i].transpose_gating_einsum,
                    name=f"expert_mlp_{i}"
                )
                mlp_outputs.append(expert_mlp(x_normed))
            else:
                mlp_outputs.append(None)
        
        # MoE Logic: Second residual connection.
        # This matches the `outputs += attn_output` line in `_modules.Block`.
        xs = [x + y if x is not None else y for x, y in zip(xs, mlp_outputs)]
        
        return xs, new_kv_cache

# --- NEW: A Multimodal, MoE-aware Top-Level Model ---
class Gemma3MoEModel(nn.Module):
    """
    A unified, multimodal model that replaces both SigLIP and PaliGemma.
    It handles vision encoding internally and routes tokens to different experts.
    """
    # This now takes a single config, which contains vision info
    config: _config.TransformerConfig 
    # Add a field to know how many experts we have
    num_experts: int

    def setup(self):
        # 1. Instantiate the Embedder and Vision Encoder from Gemma 3
        #    This is the logic from _transformer.py's setup()
        self.embedder = _modules.Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
            vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
            if self.config.vision_encoder
            else None,
        )
        self.vision_encoder = self.config.vision_encoder

        # 2. Create the MoE Blocks
        #    We create a list of configs for the blocks, one for each expert
        expert_configs = [self.config] * self.num_experts
        self.blocks = [
            Gemma3MoEBlock(configs=expert_configs, name=f"moe_block_{i}")
            for i in range(self.config.num_layers)
        ]
        
        # 3. Create a final norm for EACH expert path
        self.final_norms = [
            _layers.RMSNorm(name=f"final_norm_{i}") for i in range(self.num_experts)
        ]

    # This __call__ method now looks more like Pi0's usage pattern
    def __call__(
        self,
        # The model now takes raw inputs, not pre-embedded ones
        prefix_tok: jax.Array,
        suffix_tok: jax.Array,
        images: jax.Array | None,
        mask: jax.Array,
        positions: jax.Array,
        adarms_cond: jax.Array | None,
        kv_cache: Sequence[LayerCache | None] | None = None,
    ):
        # --- Part A: Internal Vision & Text Embedding (from _transformer.py) ---
        
        # Merge prefix and suffix tokens for initial embedding
        full_tokens = jnp.concatenate([prefix_tok, suffix_tok], axis=1)

        # Embed all tokens
        embeddings = self.embedder.encode(full_tokens)

        # If images are present, encode them and merge them into the embeddings
        if images is not None and self.vision_encoder is not None:
            # This logic is borrowed from _transformer.py/_merge_mm_embeddings
            patches = self.vision_encoder.patchify_images(images)
            vision_embeds = self.vision_encoder(patches=patches, is_training=False)
            vision_embeds = self.embedder.encode_vision(vision_embeds)
            
            embeddings = _token_utils.merge_embeddings(
                text_embeddings=embeddings,
                vision_embeddings=vision_embeds,
                mask=full_tokens == gemma_vision.TOKEN_PLACEHOLDER,
            )

        # --- Part B: Route to Experts ---
        prefix_len = prefix_tok.shape[1]
        
        # Split the combined embeddings back into prefix and suffix
        prefix_embeddings = embeddings[:, :prefix_len, :]
        suffix_embeddings = embeddings[:, prefix_len:, :]
        
        # This is the core MoE routing for Pi0.
        # Expert 0 (PaliGemma) gets the prefix.
        # Expert 1 (Action Expert) gets the suffix.
        xs = [prefix_embeddings, suffix_embeddings]

        # --- Part C: Run through MoE Blocks ---
        if kv_cache is None:
            kv_cache = [None] * self.config.num_layers
        
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            xs, new_cache = block(
                xs,
                positions=positions,
                attn_mask=mask,
                kv_cache=kv_cache[i]
            )
            # Apply AdaRMS conditioning only to the action expert (expert 1)
            if adarms_cond is not None and xs[1] is not None:
                # This is a simplification; the real AdaRMS is inside the norm layer.
                # The condition needs to be passed into the block. We'll adjust the block.
                # For now, let's assume the block handles it.
                pass
            new_kv_caches.append(new_cache)

        # --- Part D: Final Normalization per Expert ---
        outputs = [
            norm(x) if x is not None else None
            for norm, x in zip(self.final_norms, xs)
        ]
        
        return outputs, new_kv_caches
    
        
def main():
    # --- Example Usage ---

    # 1. Define the number of experts
    NUM_EXPERTS = 2

    # 2. Get the base configuration for a Gemma 3 model
    # You would get this from the Gemma 3 repository's model definitions.
    # Let's create a simplified one for demonstration.
    base_config = _config.TransformerConfig(
        final_logit_softcap=None,
        num_embed=262_144,
        embed_dim=2560,
        hidden_dim=2560 * 8 // 2,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=_config.make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
        ),
        query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=gemma_vision.SigLiPFromPatches(),
    )

    print(f"Base Config: {base_config}")

    # 3. Create a list of configs, one for each expert.
    # Here you could vary the hidden_dim for each expert if you wanted.
    moe_model = Gemma3MoEModel(config=base_config, num_experts=NUM_EXPERTS)


    print(f"MoE Model: {moe_model}")

    # You can now use moe_model.init(...) and moe_model.apply(...)
    # The input `embedded` would be a list of tensors, one per expert.
    # e.g., [expert0_tokens, expert1_tokens, None, expert3_tokens]

if __name__ == "__main__":
    main()