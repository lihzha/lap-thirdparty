"""Gemma3 adaptation for Pi0, based on official Gemma 3 modules.

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

import openpi.shared.array_typing as at

from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules

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

# Type alias for KV cache
LayerCache = _modules.LayerCache

# Configuration type alias matching old gemma.py interface
Variant = Literal["gemma3_4b", "gemma3_4b_lora"]


def get_config(variant: Variant) -> _config.TransformerConfig:
    """Returns config for specified Gemma3 variant, matching old interface."""
    if variant == "gemma3_4b":
        return _config.TransformerConfig(
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
            vision_encoder=None,
        )
    elif variant == "gemma3_4b_lora":
        # TODO: Add LoRA support if needed
        raise NotImplementedError("LoRA not yet implemented for Gemma3")
    else:
        raise ValueError(f"Unknown variant: {variant}")


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with optional adaptive conditioning (AdaRMS) for timestep injection."""
    
    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array | None = None):
        """
        Args:
            x: Input tensor
            cond: Optional conditioning vector for AdaRMS (e.g., timestep embedding)
        
        Returns:
            Tuple of (normalized_output, gate) where gate is used for gated residuals
        """
        # Standard RMSNorm
        normed = _layers.RMSNorm()(x)
        
        if cond is None:
            # Regular RMSNorm without conditioning
            return normed, None
        
        # Adaptive RMSNorm (AdaRMS)
        # Project conditioning to scale, shift, and gate
        dtype = x.dtype
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype, name="ada_modulation")(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        
        # Apply adaptive modulation
        output = normed * (1 + scale) + shift
        return output, gate


class Gemma3MoEAttention(nn.Module):
    """
    MoE-aware attention that handles multiple token sequences.
    Concatenates inputs, runs shared attention, then splits outputs.
    """
    configs: Sequence[_config.TransformerConfig]

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jax.Array,
        attn_mask: jax.Array,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:
        
        # Filter out None inputs and track their positions
        active_indices = [i for i, x in enumerate(xs) if x is not None]
        active_tensors = [xs[i] for i in active_indices]
        
        if not active_tensors:
            return [None] * len(xs), kv_cache

        # Concatenate active tensors
        lengths = [x.shape[1] for x in active_tensors]
        concatenated_x = jnp.concatenate(active_tensors, axis=1)

        # Use config from first expert for shared attention
        config = self.configs[0]
        attn_type = config.attention_types[0]  # Use first layer's attention type
        
        # Select appropriate RoPE frequency based on attention type
        rope_base_frequency = (
            config.local_base_frequency
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else config.global_base_frequency
        )
        
        # Create shared attention module
        shared_attention = _modules.Attention(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            features=config.embed_dim,
            head_dim=config.head_dim,
            attn_type=attn_type,
            query_pre_attn_scalar=config.query_pre_attn_scalar(),
            sliding_window_size=config.sliding_window_size,
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            use_qk_norm=config.use_qk_norm,
            rope_base_frequency=rope_base_frequency,
            name="shared_attention"
        )
        
        # Run attention
        new_kv_cache, encoded = shared_attention(
            x=concatenated_x,
            segment_pos=positions,
            cache=kv_cache,
            attn_mask=attn_mask,
        )

        # Split outputs back
        split_outputs = jnp.split(encoded, jnp.cumsum(jnp.array(lengths[:-1])), axis=1)
        
        # Reconstruct output list with Nones in original positions
        outputs = [None] * len(xs)
        for idx, active_idx in enumerate(active_indices):
            outputs[active_idx] = split_outputs[idx]
                
        return outputs, new_kv_cache


class Gemma3MoEBlock(nn.Module):
    """
    MoE transformer block with shared attention and separate FFN experts.
    Supports adaptive normalization for timestep conditioning.
    """
    configs: Sequence[_config.TransformerConfig]
    use_adarms: Sequence[bool]  # Whether each expert uses AdaRMS

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jax.Array, None]],
        positions: jax.Array,
        attn_mask: jax.Array,
        kv_cache: LayerCache | None,
        adarms_cond: Sequence[Union[jax.Array, None]],
    ) -> tuple[Sequence[Union[jax.Array, None]], LayerCache | None]:
        
        # Pre-attention normalization (with optional AdaRMS)
        normed_for_attn = []
        attn_gates = []
        for i, (x, use_ada) in enumerate(zip(xs, self.use_adarms)):
            if x is not None:
                cond = adarms_cond[i] if use_ada else None
                normed, gate = AdaptiveRMSNorm(name=f"pre_attention_norm_{i}")(x, cond)
                normed_for_attn.append(normed)
                attn_gates.append(gate)
            else:
                normed_for_attn.append(None)
                attn_gates.append(None)

        # Shared attention
        attn_outputs, new_kv_cache = Gemma3MoEAttention(
            configs=self.configs, 
            name="moe_attention"
        )(normed_for_attn, positions, attn_mask, kv_cache)
        
        # First residual with optional gating
        xs = [
            _gated_residual(x, attn_out, gate)
            for x, attn_out, gate in zip(xs, attn_outputs, attn_gates)
        ]
        
        # Pre-FFN normalization (with optional AdaRMS)
        normed_for_ffw = []
        ffw_gates = []
        for i, (x, use_ada) in enumerate(zip(xs, self.use_adarms)):
            if x is not None:
                cond = adarms_cond[i] if use_ada else None
                normed, gate = AdaptiveRMSNorm(name=f"pre_ffw_norm_{i}")(x, cond)
                normed_for_ffw.append(normed)
                ffw_gates.append(gate)
            else:
                normed_for_ffw.append(None)
                ffw_gates.append(None)

        # Expert FFNs (separate for each expert)
        mlp_outputs = []
        for i, x_normed in enumerate(normed_for_ffw):
            if x_normed is not None:
                expert_mlp = _modules.FeedForward(
                    features=self.configs[i].embed_dim,
                    hidden_dim=self.configs[i].hidden_dim,
                    transpose_gating_einsum=self.configs[i].transpose_gating_einsum,
                    name=_name("mlp", i)
                )
                mlp_outputs.append(expert_mlp(x_normed))
            else:
                mlp_outputs.append(None)
        
        # Second residual with optional gating
        xs = [
            _gated_residual(x, mlp_out, gate)
            for x, mlp_out, gate in zip(xs, mlp_outputs, ffw_gates)
        ]
        
        return xs, new_kv_cache


class Module(nn.Module):
    """
    Main Gemma3 MoE module matching the interface of old gemma.py.
    Supports multiple experts with shared attention and separate FFNs.
    """
    configs: Sequence[_config.TransformerConfig]
    embed_dtype: str
    adarms: bool = False  # Whether to use AdaRMS (for Pi0.5)

    def setup(self):
        # Verify all configs have same depth
        assert all(c.num_layers == self.configs[0].num_layers for c in self.configs)
        
        # Single embedder for all experts (uses first config)
        self.embedder = _modules.Embedder(
            vocab_size=self.configs[0].num_embed,
            embed_dim=self.configs[0].embed_dim,
            vision_proj_dim=None,  # Vision handled separately
        )
        
        # Determine which experts use AdaRMS
        # Convention: only the action expert (index 1) uses AdaRMS in Pi0.5
        use_adarms = [False, self.adarms] if len(self.configs) == 2 else [False] * len(self.configs)
        
        # Create MoE blocks using scan for efficiency
        block_cls = nn.remat(
            Gemma3MoEBlock,
            prevent_cse=False,
            static_argnums=(5,),  # adarms_cond structure is static
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].num_layers,
        )(configs=self.configs, use_adarms=use_adarms)
        
        # Final norms for each expert
        self.final_norms = [
            AdaptiveRMSNorm(name=_name("final_norm", i)) 
            for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """Embed tokens (matching old gemma.py interface)."""
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[Union[at.Float[at.Array, "b _t _d"], None]],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[Union[at.Float[at.Array, "b _d"], None]] | None = None,
        kv_cache: Sequence[LayerCache | None] | None = None,
    ) -> tuple[Sequence[Union[at.Float[at.Array, "b _t _d"], None]], Sequence[LayerCache | None]]:
        """
        Forward pass matching old gemma.py interface.
        
        Args:
            embedded: List of embedded token sequences (one per expert, None for unused)
            positions: Position indices
            mask: Attention mask
            adarms_cond: Optional adaptive conditioning (for Pi0.5)
            kv_cache: Optional KV cache for inference
        
        Returns:
            Tuple of (outputs, new_kv_cache)
        """
        # Cast to target dtype
        embedded = [e.astype(self.embed_dtype) if e is not None else None for e in embedded]
        
        # Reshape mask to match expected format [B, 1, T, S]
        mask = jnp.asarray(mask)[:, None, :, :]
        
        # Default adarms_cond if not provided
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        
        # Initialize KV cache if needed
        if kv_cache is None:
            kv_cache = [None] * self.configs[0].num_layers
        
        # Run through transformer blocks
        xs = embedded
        new_kv_caches = []
        for i in range(self.configs[0].num_layers):
            xs, new_cache = self.layers(
                xs, positions, mask, kv_cache[i], adarms_cond
            )
            new_kv_caches.append(new_cache)
        
        # Final normalization for each expert
        outputs = []
        for i, (x, use_ada) in enumerate(zip(xs, [False, self.adarms])):
            if x is not None:
                cond = adarms_cond[i] if use_ada else None
                normed, _ = self.final_norms[i](x, cond)
                outputs.append(normed)
            else:
                outputs.append(None)
        
        return outputs, new_kv_caches

    def init(self, use_adarms: Sequence[bool]):
        """Initialize all parameters (matching old gemma.py interface)."""
        # Dummy forward pass to initialize parameters
        batch_size = 1
        seq_len = 10
        
        self.embed(jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
        
        dummy_embedded = [
            jnp.zeros((batch_size, seq_len, c.embed_dim)) 
            for c in self.configs
        ]
        dummy_positions = jnp.arange(seq_len)[None, :]
        dummy_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=bool)
        dummy_adarms = [
            jnp.zeros((batch_size, c.embed_dim)) if use_ada else None
            for c, use_ada in zip(self.configs, use_adarms)
        ]
        
        self(dummy_embedded, dummy_positions, dummy_mask, dummy_adarms)


def _name(name: str, i: int) -> str:
    """
    Name layers to match checkpoint loading convention.
    First expert has no suffix, subsequent experts have _1, _2, etc.
    """
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    """Apply residual connection with optional gating."""
    if x is None or y is None:
        return x if y is None else y
    if gate is None:
        return x + y
    return x + y * gate



# Unit Tests
# Test Gemma3 module initialization
config = get_config("gemma3_4b")
model = Module(configs=[config, config], embed_dtype="bfloat16")
model.init(use_adarms=[False, True])

# Test forward pass
embedded = [jnp.zeros((1, 10, 2560)), jnp.zeros((1, 5, 2560))]
positions = jnp.arange(15)[None, :]
mask = jnp.ones((1, 15, 15), dtype=bool)
outputs, _ = model(embedded, positions, mask)