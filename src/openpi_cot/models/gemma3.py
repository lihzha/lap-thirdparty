"""Gemma3 core modules (copied from official Gemma3 repo with minimal changes).

This file contains the essential Attention, FeedForward, and RMSNorm layers
from the official Gemma3 codebase, adapted for use in Pi0 MoE architecture.

Original sources:
- _modules.py: Attention, FeedForward, Embedder, Block
- _layers.py: Einsum, RMSNorm
- _positional_embeddings.py: apply_rope
"""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Literal, TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from openpi_cot.models.gemma_common import Embedder as CommonEmbedder
from openpi_cot.models.gemma_common import RMSNorm as CommonRMSNorm
from openpi_cot.models.gemma_common import _gated_residual
from openpi_cot.models.gemma_common import _name

# ============================================================================
# Constants
# ============================================================================
K_MASK = -2.3819763e38
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0
GEMMA3_VOCAB_SIZE = 262_144

LayerCache = dict[str, jax.Array]


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


# ============================================================================
# Configuration & Factory
# ============================================================================

Variant = Literal["gemma3_300m", "gemma3_1b", "gemma3_4b", "gemma3_12b"]


@dataclasses.dataclass
class Config:
    width: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    sliding_window_size: int
    use_qk_norm: bool
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    transpose_gating_einsum: bool
    vocab_size: int = GEMMA3_VOCAB_SIZE
    lora_configs: dict = dataclasses.field(default_factory=dict)
    attn_logits_soft_cap: float | None = None
    attn_type: AttentionType = AttentionType.GLOBAL


def get_config(variant: Variant) -> Config:
    """Returns config dict for specified Gemma3 variant."""
    if variant == "gemma3_1b":
        return Config(
            width=1152,
            hidden_dim=6 * 1152,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=26,
            sliding_window_size=512,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
            attn_logits_soft_cap=None,
            attn_type=AttentionType.GLOBAL,
        )
    if variant == "gemma3_4b":
        return Config(
            width=2560,
            hidden_dim=2560 * 8 // 2,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=34,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
            attn_logits_soft_cap=None,
            attn_type=AttentionType.GLOBAL,
        )
    if variant == "gemma3_12b":
        return Config(
            width=3840,
            hidden_dim=8 * 3840 // 2,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=48,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
            attn_logits_soft_cap=None,
            attn_type=AttentionType.GLOBAL,
        )
    if variant == "gemma3_300m":
        return Config(
            width=768,
            hidden_dim=768 * 4,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=34,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
            attn_logits_soft_cap=None,
            attn_type=AttentionType.GLOBAL,
        )
    if variant == "gemma3_4b_lora":
        return Config(
            width=2560,
            hidden_dim=2560 * 8 // 2,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=34,
            sliding_window_size=1024,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
            attn_logits_soft_cap=None,
            attn_type=AttentionType.GLOBAL,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


# ============================================================================
# Wrapper Classes with Gemma3 Defaults
# ============================================================================


@at.typecheck
class Embedder(CommonEmbedder):
    """Embedder with Gemma3 default parameter dtype.

    Inherits from common implementation. param_dtype can be overridden per instance.
    """

    param_dtype: str | None = "bfloat16"


@at.typecheck
class RMSNorm(CommonRMSNorm):
    """RMSNorm with Gemma3 default parameter dtype.

    Inherits from common implementation. param_dtype can be overridden per instance.
    """

    param_dtype: str | None = "bfloat16"


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
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction
    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class Einsum(nn.Module):
    """Einsum layer for parameterized tensor multiplication.

    For Gemma3, parameters default to bfloat16 to ensure proper sharding.
    """

    shape: tuple[int, ...]
    weight_name: str = "w"
    initializer: nn.initializers.Initializer = nn.initializers.normal()
    dtype: jnp.dtype | None = None
    lora_config: "lora.LoRAConfig" = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        if self.lora_config is not None:
            # Use LoRA if configured
            w = lora.Einsum(
                shape=self.shape,
                weight_name=self.weight_name,
                init_fn=self.initializer,
                lora_config=self.lora_config,
            )(eqn, x)
            return w
        # Default to bfloat16 for Gemma3 to avoid sharding mismatches
        # (float32 params > 4MB get sharded, but bfloat16 params < 4MB stay replicated)
        param_dtype = self.dtype if self.dtype is not None else jnp.bfloat16
        w = self.param(
            self.weight_name,
            self.initializer,
            self.shape,
            param_dtype,
        )
        return jnp.einsum(eqn, x, w)


# ============================================================================
# Attention & FeedForward (from _modules.py)
# ============================================================================


@at.typecheck
class Attention(nn.Module):
    configs: Sequence[Config]

    @property
    def use_qkv_einsum(self):
        return self.num_kv_heads == self.num_heads

    @property
    def use_gqa(self):
        return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

    @nn.compact
    def __call__(
        self,
        xs: Sequence[jnp.ndarray | None],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        kv_cache: LayerCache | None,
    ) -> tuple[Sequence[jnp.ndarray | None], LayerCache | None]:
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        # 1) Compute Projections for all experts
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:  # Non GQA
                qkv_einsum = Einsum(
                    name=_name("qkv_einsum", i),
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    lora_config=config.lora_configs.get("attn"),
                    dtype=dtype,
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:  # GQA
                q_einsum = Einsum(
                    name=_name("q_einsum", i),
                    shape=(config.num_heads, config.width, config.head_dim),
                    lora_config=config.lora_configs.get("attn"),
                    dtype=dtype,
                )

                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = Einsum(
                    name=_name("kv_einsum", i),
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    lora_config=config.lora_configs.get("attn"),
                    dtype=dtype,
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)

                # Apply RMSNorm to Q and K if configured
                if config.use_qk_norm:
                    rmsnorm_q = RMSNorm(name=_name("q_rmsnorm", i))
                    rmsnorm_k = RMSNorm(name=_name("k_rmsnorm", i))
                    q, _ = rmsnorm_q(q, None)
                    k, _ = rmsnorm_k(k, None)
                qkvs.append((q, k, v))

        # concatenate all experts along the sequence dimension
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 2) Apply RoPE and scale queries
        q = apply_rope(
            q, positions=positions, base_frequency=DEFAULT_ROPE_BASE_FREQUENCY, scale_factor=DEFAULT_ROPE_SCALE_FACTOR
        )
        q *= self.configs[0].head_dim ** -0.5

        k = apply_rope(
            k, positions=positions, base_frequency=DEFAULT_ROPE_BASE_FREQUENCY, scale_factor=DEFAULT_ROPE_SCALE_FACTOR
        )

        assert q.dtype == k.dtype == v.dtype == dtype, "Mismatched dtypes in attention inputs"

        # 3) Save KV Values to the Cache
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        # 4) Compute Attention Scores
        if config.num_kv_heads == config.num_heads:  # Non GQA
            B, T, N, H = q.shape
            logits = jnp.einsum("BTNH,BSNH->BTNS", q, k, preferred_element_type=jnp.float32)  # B T N S
        else:  # GQA
            B, T, N, H = q.shape
            K = self.configs[0].num_kv_heads
            G = int(N // K)
            q = q.reshape(B, T, K, G, H)
            # logits = jnp.einsum('BTKGH,BSKH->BKGTS', q, k, preferred_element_type=jnp.float32) # B T K G S
            logits = jnp.einsum("BTKGH,BSKH->BTKGS", q, k)
            _B, _T, _K, _G, _S = logits.shape
            # logits = logits.reshape((B, T, K*G, S))
            logits = logits.reshape((_B, _T, _K * _G, _S))

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )
        # Assuming Global Attention Pattern since MoE experts share attn_type
        # Thus, no need for sliding window mask here

        # masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, K_MASK)
        # masked_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
        # Goal of attn_mask is to produce a mask of shape (B, T, S) and apply the same mask across all N heads
        # Hence we can expand dims to form (B, 1, T, S) and broadcast across N heads
        broadcastable_mask = jnp.expand_dims(attn_mask.squeeze(axis=1), axis=2)
        masked_logits = jnp.where(broadcastable_mask, logits, K_MASK)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 5) Compute Attention Output
        if config.num_kv_heads == config.num_heads:  # Non GQA
            encoded = jnp.einsum("BTNS,BSNH->BTNH", probs, v)
        else:  # GQA
            B, T, N, S = probs.shape
            K = self.configs[0].num_kv_heads
            G = int(N // K)
            probs_reshaped = probs.reshape(B, T, K, G, S)

            encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs_reshaped, v)
            # encoded = jnp.einsum('BTKGH,BSKH->BTKGS', probs, v)
            _B, _T, _K, _G, _H = encoded.shape
            encoded = encoded.reshape((_B, _T, _K * _G, _H))

        # Expert-specific output projections

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = Einsum(
                    name=_name("attn_vec_einsum", i),
                    shape=(config.num_heads, config.head_dim, config.width),
                    lora_config=config.lora_configs.get("attn"),
                    dtype=dtype,
                )

                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end, :, :]))

                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    transpose_gating_einsum: bool = False
    lora_config: "lora.LoRAConfig" = None

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        if self.transpose_gating_einsum:
            eq = "...F,NHF->...NH"
            gating = Einsum(
                shape=(2, self.hidden_dim, self.features),
                weight_name="gating_einsum",
                lora_config=self.lora_config,
            )
        else:
            eq = "...F,NFH->...NH"
            gating = Einsum(
                shape=(2, self.features, self.hidden_dim),
                weight_name="gating_einsum",
                lora_config=self.lora_config,
            )

        gate = gating(eq, x)
        activations = nn.gelu(gate[..., 0, :]) * gate[..., 1, :]

        linear = Einsum(
            shape=(self.hidden_dim, self.features),
            weight_name="linear",
            lora_config=self.lora_config,
        )
        outputs = linear("...H,HF->...F", activations)
        return outputs


@at.typecheck
class Block(nn.Module):
    """MoE transformer block: shared attention + expert FFNs."""

    configs: Sequence[Config]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    use_post_attn_norm: bool = True  # new in gemma2 and gemma3
    use_post_ffw_norm: bool = True  # new in gemma2 and gemma3

    @nn.compact
    def __call__(
        self,
        xs: Sequence[jax.Array | None],
        kv_cache: LayerCache | None,
        positions: jax.Array,
        attn_mask: jax.Array,
        adarms_cond: Sequence[jax.Array | None],
        deterministic: bool = True,
    ) -> tuple[Sequence[jax.Array | None], LayerCache | None]:
        """Apply MoE block with adaptive conditioning."""
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # Pre-attention normalization with optional AdaRMS
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)

        # Shared attention
        post_attn, kv_cache = Attention(configs=self.configs, name="attn")(pre_attn, positions, attn_mask, kv_cache)

        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)

        # Apply post attention to first expert only if configured
        if self.use_post_attn_norm:
            post_attn_normed = []
            for i, x in enumerate(post_attn):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_attention_norm")(x, None)  # noqa: PLW2901
                post_attn_normed.append(x)
            post_attn = post_attn_normed

        # First residual with gating
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # Pre-FFN normalization with optional AdaRMS
        pre_ffw = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
            pre_ffw.append(x)
            gates.append(gate if x is not None else None)

        pre_ffw = sharding.activation_sharding_constraint(pre_ffw)

        # Expert FFNs (separate for each expert)
        ffn_outs = []
        for i, (x, config) in enumerate(zip(pre_ffw, self.configs, strict=True)):
            if x is not None:
                ffn = FeedForward(
                    features=config.width,
                    hidden_dim=config.hidden_dim,
                    transpose_gating_einsum=config.transpose_gating_einsum,
                    lora_config=config.lora_configs.get("ffn"),
                    name=_name("mlp", i),
                )
                ffn_outs.append(ffn(x))
            else:
                ffn_outs.append(None)

        ffn_outs = sharding.activation_sharding_constraint(ffn_outs)
        ffn_outs = jax.tree.map(lambda x: drop(x, deterministic), ffn_outs)

        # Apply post_ffw_norm only to the first expert if post_norms is enabled
        if self.use_post_ffw_norm:
            out_normed = []
            for i, x in enumerate(ffn_outs):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_ffw_norm")(x, None)  # noqa: PLW2901
                out_normed.append(x)
            out = out_normed

        # Second residual with gating
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model supporting mixture of experts for Pi0."""

    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.num_layers == self.configs[0].num_layers for config in self.configs)

        self.embedder = Embedder(
            vocab_size=self.configs[0].vocab_size,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # deterministic is static
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,  # kv_cache
                nn.broadcast,  # positions
                nn.broadcast,  # attn_mask
                nn.broadcast,  # adarms_cond
                nn.broadcast,  # deterministic
            ),
            length=self.configs[0].num_layers,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            use_post_attn_norm=self.configs[0].use_post_attn_norm,
            use_post_ffw_norm=self.configs[0].use_post_ffw_norm,
        )

        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
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
        """Initialize all parameters."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )
