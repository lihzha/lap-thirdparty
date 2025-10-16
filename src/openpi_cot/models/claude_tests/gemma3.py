
"""Gemma3 core modules (copied from official Gemma3 repo with minimal changes).

This file contains the essential Attention, FeedForward, and RMSNorm layers
from the official Gemma3 codebase, adapted for use in Pi0 MoE architecture.

Original sources:
- _modules.py: Attention, FeedForward, Embedder, Block
- _layers.py: Einsum, RMSNorm
- _positional_embeddings.py: apply_rope
"""
import dataclasses
import enum
from typing import Literal, Sequence, Union, TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron import kd

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


# ============================================================================
# Constants
# ============================================================================
K_MASK = -2.3819763e38 
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

LayerCache = dict[str, jax.Array]

# ============================================================================
# Configuration & Factory
# ============================================================================

Variant = Literal["gemma3_1b", "gemma3_4b", "gemma3_12b"]

@dataclasses.dataclass
class Config:
    embed_dim: int # width
    hidden_dim: int # mlp_dim
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    num_layers: int # depth
    sliding_window_size: int
    use_qk_norm: bool
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    transpose_gating_einsum: bool

def get_config(variant: Variant) -> Config:
    """Returns config dict for specified Gemma3 variant."""
    if variant == "gemma3_1b":
        return Config(
            embed_dim=1152,
            hidden_dim=6 * 1152,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            vocab_size=262_144,
            num_layers=26,
            sliding_window_size=512,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    elif variant == "gemma3_4b":
        return Config(
            embed_dim=2560,
            hidden_dim=2560 * 8 // 2,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=262_144,
            num_layers=34,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    elif variant == "gemma3_12b":
        return Config(
            embed_dim=3840,
            hidden_dim=8 * 3840 // 2,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            vocab_size=262_144,
            num_layers=48,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

# ============================================================================
# Positional Embeddings (from _positional_embeddings.py)
# ============================================================================

def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE.
    f(x, m)_i     = x_i * cos(m * θ_j) - x_{i+1} * sin(m * θ_j)
    f(x, m)_{i+1} = x_i * sin(m * θ_j) + x_{i+1} * cos(m * θ_j)

    Let B denote batch size, L denote sequence length, N denote number of heads,
    and H denote head dimension. Note that H must be divisible by 2.

    Args:
        inputs: Array of shape [B, L, N, H].
        positions:  Array of shape [B, L].
        base_frequency: Base frequency used to compute rotations.
        scale_factor: The scale factor used for positional interpolation, allowing
        an expansion of sequence length beyond the pre-trained context length.

    Returns:
        Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim //2) / head_dim
    timescale = base_frequency ** fraction
    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)

# ============================================================================
# Base Layers (from _layers.py)
# ============================================================================

class Einsum(nn.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""
    shape: tuple[int, ...]
    weight_name: str = 'w'
    initializer: nn.initializers.Initializer = nn.initializers.normal()
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            self.weight_name,
            self.initializer,
            self.shape,
            self.dtype if self.dtype is not None else None,
        )
        return jnp.einsum(eqn, x, w)

@at.typecheck
class RMSNorm(nn.Module):
    """RMSNorm layer."""
    @nn.compact
    def __call__(self, x, cond):
        scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        if cond is None:

            # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
            # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
            # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
            scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs, None
    
        modulation = nn.Dense(x.shape[-1]*3, kernel_init=nn.initializers.zeros, dtype=x.dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(x.dtype), gate

# ============================================================================
# Attention & FeedForward (from _modules.py)
# ============================================================================

def _create_sliding_mask(
        segment_pos: jnp.ndarray,
        end_index: int,
        cache_len: int,
        sliding_window_size: int,
):
    """Creates mask for sliding window attention.""" 
    total_tokens = end_index + segment_pos.shape[1] # cached + processing tokens

    def _reconstruct_rotated_cache_positions():
        cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
        cache_positions = (
            jnp.zeros_like(cache_positions)
            # kv were placed at index (possition_id % cache_len) in the cache
            .at[cache_positions % cache_len].set(cache_positions)
        )
        return cache_positions
    
    # Reconstruct position_ids for cached kv.
    cache_positions = jax.lax.cond(
        total_tokens <= cache_len,
        lambda: jnp.arange(cache_len),
        _reconstruct_rotated_cache_positions,
    )
    cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
    segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]
    sliding_mask = cache_positions > segment_pos - sliding_window_size
    sliding_mask *= cache_positions < segment_pos + sliding_window_size
    return sliding_mask

class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2

@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""
    vocab_size: int
    embed_dim: int
    vision_proj_dim: int | None = None

    def setup(self):
        # Embedding matrix of shape [vocab_size, embed_dim].
        self.input_embedding_table = self.param(
            'input_embedding',
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )
        # The original Gemma3 code has a possible multi-modal projection layer
        # which we are omitting here and adding directly in the Pi0 model
    
    def encode(self, x: jax.Array) -> jax.Array:
        """Encodes the input tokens.

        Args:
        x: Input tokens of shape [seq_len] or [batch_size, seq_len], where
            each token is an integer in [0, vocab_size).

        Returns:
        Encoded tokens of shape [seq_len, embed_dim] or [batch_size, seq_len,
        embed_dim].
        """
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x
    
    def decode(self, x: jax.Array) -> jax.Array:
        """Decodes the input vectors.

        Args:
        x: Array of shape [seq_len, embed_dim] or [batch_size, seq_len,
        embed_dim].

        Returns:
        Array of shape [seq_len, vocab_size] or [batch_size, seq_len, vocab_size].
        """
        logits = jnp.dot(x, self.input_embedding_table.T)
        return logits
    
    def encode_vision(self, x: jax.Array) -> jax.Array:
        pass  # Vision encoder not implemented here.

class Attention_VLM(nn.Module):
    """Attention module.
    Original implementation for Gemma3. Needs wrapped to handle MoE architecture
    """
    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int
    attn_type: AttentionType
    query_pre_attn_scalar: float
    rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
    rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
    attn_logits_soft_cap: float | None = None
    sliding_window_size: int | None = None
    use_qk_norm: bool = False

    @property
    def use_qkv_einsum(self):
        return self.num_kv_heads == self.num_heads
    
    @property
    def use_gqa(self):
        return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1
    
    def setup(self):
        self.attn_vec_einsum = Einsum(
            shape=(self.num_heads, self.head_dim, self.features),
        )
        if self.use_qkv_einsum:
            self.qkv_einsum = Einsum(
                shape=(3, self.num_heads, self.features, self.head_dim),
            )
        else:
            self.q_einsum = Einsum(
                shape=(self.num_heads, self.features, self.head_dim),
            )
            self.kv_einsum = Einsum(
                shape=(2, self.num_kv_heads, self.features, self.head_dim),
            )
        if self.use_qk_norm:
            self._query_norm = RMSNorm()
            self._key_norm = RMSNorm()
        self.attention_weights = kd.nn.Identity()

    def __call__(
            self,
            x: jax.Array,
            segment_pos: jax.Array,
            cache: LayerCache | None,
            attn_mask: jax.Array,
    ) -> tuple[LayerCache | None, jax.Array]:
        """Applies multi-head attention to the inputs.

        Args:
            x: Input sequence of shape [batch_size, seq_len, embed_dim].
            segment_pos: Input absolute positions of shape [batch_size, seq_len].
            cache: KV cache or None.
            attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

        Returns:
            cache: Updated attention KV cache.
            outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
        """
        if self.use_qkv_einsum:
            # [batch_size, seq_len, num_heads, head_dim]
            query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
        else:
            query_proj = self.q_einsum('BTD,NDH->BTNH', x)
            key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

        if self.use_qk_norm:
            query_proj = self._query_norm(query_proj)
            key_proj = self._key_norm(key_proj)

        query_proj = apply_rope(
            query_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )
        query_scaled = query_proj * self.query_pre_attn_scalar

        key_proj = apply_rope(
            key_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )

        # Cache is left aligned.
        # Save the KV values to the cache.
        if cache is not None:
            end_index = cache['end_index'][0]
            cache_size = cache['v'].shape[1]
            slice_indices = (0, end_index % cache_size, 0, 0)

            # [batch_size, cache_size, num_heads, head_dim]
            value_proj = jax.lax.dynamic_update_slice(
                cache['v'],
                value_proj,
                slice_indices,
            )

            # [batch_size, cache_size, num_heads, head_dim]
            key_proj = jax.lax.dynamic_update_slice(
                cache['k'], key_proj, slice_indices
            )

        if self.use_gqa:
            b, t, kg, h = query_scaled.shape
            query_scaled = query_scaled.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
            b, t, k, g, s = logits.shape
            logits = logits.reshape((b, t, k * g, s))
        else:
            # [batch_size, seq_len, num_heads, cache_size]
            # If cache is None, then cache_size = seq_len.
            logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

        if self.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / self.attn_logits_soft_cap)
            logits = logits * self.attn_logits_soft_cap

        if self.attn_type == AttentionType.LOCAL_SLIDING:
            if self.sliding_window_size is None:
                raise ValueError('Sliding_window_size must be set if Local Sliding attention')
            sliding_mask = _create_sliding_mask(
                segment_pos,
                end_index=cache['end_index'][0] if cache is not None else 0,
                # Derive cache length from attn_mask shape in case cache is None
                cache_len=attn_mask.shape[-1],
                sliding_window_size=self.sliding_window_size,
            )
            # [batch_size, seq_len, cache_size]
            attn_mask *= sliding_mask
        # [batch_size, seq_len, num_heads, cache_size]
        padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

        # Multi-head attention matrices.
        # [batch_size, seq_len, num_heads, cache_size]
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
        probs = self.attention_weights(probs)

        if self.use_gqa:
            # Reshape matrices to enable einsums over groups.
            b, t, kg, h = probs.shape
            probs = probs.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
            b, t, k, g, h = encoded.shape
            encoded = encoded.reshape((b, t, k * g, h))
        else:
            # [batch_size, seq_len, num_heads, head_dim]
            encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

        # [batch_size, seq_len, features]
        attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

        if cache is not None:
            seq_len = x.shape[1]
            new_cache = {
                # [batch_size, cache_size, num_heads, head_dim]
                'v': value_proj,
                # [batch_size, cache_size, num_heads, head_dim]
                'k': key_proj,
                # [batch_size]
                'end_index': cache['end_index'] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, attn_output
    
    @classmethod
    def init_cache(
        cls,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> LayerCache:
        del cls  # not used
        return {
            'v': jnp.zeros(
                (batch_size, cache_size, num_heads, head_dim), dtype=dtype
            ),
            'k': jnp.zeros(
                (batch_size, cache_size, num_heads, head_dim), dtype=dtype
            ),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
    
@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int  # features = embed_dim
    hidden_dim: int
    transpose_gating_einsum: bool

    @nn.compact
    def __call__(self, x):
        """Applies the feed forward module.

        Args:
            x: Input sequence of shape [batch_size, seq_len, features].

        Returns:
            Output sequence of shape [batch_size, seq_len, features].
        """
        # Some versions use an alternate parameter ordering that
        # transposes hidden_dim and features.
        if self.transpose_gating_einsum:
            eq = '...F,NHF->...NH'
            gating = Einsum(
                shape=(2, self.hidden_dim, self.features),
                weight_name='gating_einsum',
            )
        else:
            eq = '...F,NFH->...NH'
            gating = Einsum(
                shape=(2, self.features, self.hidden_dim),
                weight_name='gating_einsum',
            )

        # Use the same scope for backwards compatibility with existing checkpoints
        # created before using `_layers.Einsum` here.
        nn.share_scope(self, gating)

        # [batch_size, seq_len, 2, hidden_dim]
        gate = gating(eq, x)
        # [batch_size, seq_len, hidden_dim]
        activations = nn.gelu(gate[..., 0, :]) * gate[..., 1, :]

        # Project back from hidden_dim to features.
        linear = Einsum(
            shape=(self.hidden_dim, self.features),
            weight_name='linear',
        )
        nn.share_scope(self, linear)

        # [batch_size, seq_len, features]
        outputs = linear('...H,HF->...F', activations)

        return outputs

class Block_VLM(nn.Module):
    """Transformer block.
    Original implementation for Gemma3. Needs wrapped to handle MoE architecture
    """

    num_heads: int
    num_kv_heads: int
    embed_dim: int
    head_dim: int
    hidden_dim: int
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    attn_type: AttentionType
    query_pre_attn_scalar: float
    transpose_gating_einsum: bool
    rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
    rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
    attn_logits_soft_cap: float | None = None
    sliding_window_size: int | None = None
    use_qk_norm: bool = False

    def setup(self):
        self.pre_attention_norm = RMSNorm()

        self.attn = Attention(
            num_heads=self.num_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            attn_type=self.attn_type,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            rope_base_frequency=self.rope_base_frequency,
            rope_scale_factor=self.rope_scale_factor,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            sliding_window_size=self.sliding_window_size,
            use_qk_norm=self.use_qk_norm,
        )

        self.post_attention_norm = None
        if self.use_post_attn_norm:
            self.post_attention_norm = RMSNorm()

        self.pre_ffw_norm = RMSNorm()

        self.mlp = FeedForward(
            features=self.embed_dim,
            hidden_dim=self.hidden_dim,
            transpose_gating_einsum=self.transpose_gating_einsum,
        )

        self.post_ffw_norm = None
        if self.use_post_ffw_norm:
            self.post_ffw_norm = RMSNorm()

    def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
        """Applies the block to the inputs.

        Args:
            x: Input sequence of shape [batch_size, seq_len, embed_dim].
            segment_pos: Input absolute positions of shape [batch_size, seq_len].
            cache: KV cache or None.
            attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

        Returns:
            cache: Updated attention KV cache.
            outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
        """
        inputs_normalized = self.pre_attention_norm(x)

        # attn_output.shape = [batch_size, seq_len, embed_dim]
        # cache["k"].shape = [batch_size, cache_size, num_heads, head_dim]
        # cache["v"].shape = [batch_size, cache_size, num_heads, head_dim]
        # cache["end_index"].shape = [batch_size]
        cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )

        if self.post_attention_norm is not None:
            attn_output = self.post_attention_norm(attn_output)

        attn_output += x

        outputs = self.pre_ffw_norm(attn_output)

        outputs = self.mlp(outputs)

        if self.post_ffw_norm is not None:
            outputs = self.post_ffw_norm(outputs)

        outputs += attn_output

        return cache, outputs


# ============================================================================
# MoE Wrappers for Pi0 Model
# ============================================================================

class Attention(nn.Module):
    """Shared attention for MoE pattern.

    Takes multiple token sequences, concatenates them, runs attention,
    then splits the output back.
    """
    config: dict  # From gemma3.get_config()

    @nn.compact
    def __call__(
        self, 
        xs: Sequence[Union[jnp.ndarray, None]],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jnp.ndarray, None]], LayerCache | None]:
        """Apply shared attention to multiple sequences.

        Args:
        xs: List of token sequences, one per expert (None for unused experts)
        positions: Absolute position indices
        attn_mask: Attention mask
        kv_cache: KV cache from previous step (or None for prefill)

        Returns:
        (outputs, new_cache) where outputs is list of same shape as xs
        """
        
        # Filter active sequences
        active_indices = [i for i, x in enumerate(xs) if x is not None]
        active_xs = [xs[i] for i in active_indices]

        if not active_xs:
            return [None] * len(xs), kv_cache
        
        # Concatenate active sequences
        lengths = [x.shape[1] for x in active_xs]
        concat_x = jnp.concatenate(active_xs, axis=1)

        # Run shared attention
        attn = Attention_VLM(
            num_heads = self.config["num_heads"],
            num_kv_heads = self.config["num_kv_heads"],
            features = self.config["embed_dim"],
            head_dim = self.config["head_dim"],
            attn_type = AttentionType.GLOBAL, # or LOCAL_SLIDING based on config
            query_pre_attn_scalar = self.config["heads_dim"] ** -0.5,
            rope_base_frequency = DEFAULT_ROPE_BASE_FREQUENCY,
            sliding_window_size = self.config["sliding_window_size"],
            use_qk_norm = self.config["use_qk_norm"],
        )
        new_cache, output = attn(concat_x, positions, kv_cache, attn_mask)
        
        # Split output back
        split_outputs = jnp.split(output, jnp.cumsum(jnp.array(lengths[:-1])), axis=1)

        # Reconstruct with Nones in original positions
        result = [None] * len(xs)
        for idx, active_idx in enumerate(active_indices):
            result[active_idx] = split_outputs[idx]
        return result, new_cache

class Block(nn.Module):
    """MoE transformer block: shared attention + expert FFNs."""
    config: dict
    expert_configs: Sequence[dict]  # One config per expert (can be same as config)

    @nn.compact
    def __call__(
        self,
        xs: Sequence[Union[jnp.ndarray, None]],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[Union[jnp.ndarray, None]], LayerCache | None]:
        """Apply MoE block.

        Args:
            xs: List of expert token sequences
            positions: Position indices
            attn_mask: Attention mask
            kv_cache: KV cache

        Returns:
        (outputs, new_cache)
        """
        normed_xs = [
            RMSNorm(name=f"pre_attn_norm_{i}")(x) if x is not None else None
            for i, x in enumerate(xs)
        ]

        # Shared attention
        attn_outs, new_cache = Attention(config = self.config, name="shared_attn")(
            normed_xs, positions, attn_mask, kv_cache
        )

        # Residual connection
        xs = [
            x + attn_out if x is not None else attn_out
            for x, attn_out in zip(xs, attn_outs)
        ]

        # Pre-FFN norm
        normed_xs = [
            RMSNorm(name=f"pre_ffn_norm_{i}")(x) if x is not None else None
            for i, x in enumerate(xs)
        ]

        # Expert FFNs (different network per expert)
        ffn_outs = []
        for i, (x, cfg) in enumerate(zip(normed_xs, self.expert_configs)):
            if x is not None:
                ffn = FeedForward(
                    features = cfg["embed_dim"],
                    hidden_dim = cfg["hidden_dim"],
                    transpose_gating_einsum = cfg["transpose_gating_einsum"],
                    name = f"expert_ffn_{i}",
                )
                ffn_outs.append(ffn(x))
            else:
                ffn_outs.append(None)

        # Residual connection after FFN
        xs = [
            x + ffn_out if x is not None else ffn_out
            for x, ffn_out in zip(xs, ffn_outs)
        ]

        return xs, new_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]

class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens.
    Assumes first config is VLM, second config is action expert
    """

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # assert all experts have the same number of layers / depth
        assert all(config.num_layers == self.configs[0].num_layers for config in self.configs)
        self.embedder = Embedder(
            vocab_size = self.configs[0].vocab_size,
            embed_dim = self.configs[0].embed_dim,
            vision_proj_dim = None,  # No vision encoder here,
            name="embedder",
        )

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].num_layers,
        )(
            configs = self.configs,
            dropout = self.dropout,
            dropout_bdims = self.dropout_bdims,
            post_norms = self.configs[0].post_norms
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i), param_dtype=self.configs[i].param_dtype) for i in range(len(self.configs))]

    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """Embed input tokens."""
        x = self.embedder.encode(tokens)
        return x.astype(self.embed_dtype)
    
    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache
    
    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.embed_dim)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.embed_dim)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )

    
def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"
