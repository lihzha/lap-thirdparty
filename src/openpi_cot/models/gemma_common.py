"""Shared utilities and base classes for Gemma2 and Gemma3 models.

This module consolidates common code between gemma2.py and gemma3.py to reduce
duplication and improve maintainability.
"""

import re

import flax.linen as nn
import jax.numpy as jnp
import openpi.models.lora as lora
import openpi.shared.array_typing as at


# ============================================================================
# Base Layer Implementations (unified from both gemma2 and gemma3)
# ============================================================================


class Einsum(nn.Module):
    """Einsum layer with LoRA support and explicit parameter dtype.

    This is a unified implementation that combines features from both gemma2 and gemma3.
    Supports both compact-style and setup-style initialization.
    """

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: lora.LoRAConfig | None = None
    # Parameter dtype (for models that need explicit dtype handling)
    param_dtype: str | None = None
    # Weight name (for compatibility with gemma3 style)
    weight_name: str = "w"

    def setup(self):
        pdtype = jnp.dtype(self.param_dtype) if self.param_dtype else None
        self.w = self.param(self.weight_name, self.init_fn, self.shape, pdtype)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a, pdtype)
            self.w_b = self.param("lora_b", config.init_fn, shape_b, pdtype)

    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora_result = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora_result = jnp.einsum(eqn_b, lora_result, self.w_b.astype(dtype))
            result = result + lora_result * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


@at.typecheck
class RMSNorm(nn.Module):
    """RMSNorm with optional adaptive conditioning and explicit parameter dtype.

    Unified implementation supporting both standard and adaptive (AdaRMS) normalization.
    """

    param_dtype: str | None = None

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        pdtype = jnp.dtype(self.param_dtype) if self.param_dtype else None

        if cond is None:
            # Standard RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],), pdtype)
            scale_dtype = scale.astype(jnp.float32) if self.param_dtype else scale
            normed_inputs = normed_inputs * (1 + scale_dtype)
            return normed_inputs.astype(dtype), None

        # Adaptive RMSNorm (AdaRMS) for timestep/conditioning injection
        modulation = nn.Dense(
            x.shape[-1] * 3,
            kernel_init=nn.initializers.zeros,
            dtype=dtype,
            param_dtype=pdtype,
            name="ada_modulation",
        )(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module with LoRA support and explicit parameter dtype.

    Unified implementation based on gemma2's cleaner approach.
    """

    features: int
    hidden_dim: int
    lora_config: lora.LoRAConfig | None = None
    param_dtype: str | None = None

    def setup(self):
        pdtype = jnp.dtype(self.param_dtype) if self.param_dtype else None
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
            pdtype,
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
            pdtype,
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            self.w_gating_lora = (
                self.param(
                    "gating_einsum_lora_a",
                    self.lora_config.init_fn,
                    (2, self.features, self.lora_config.rank),
                    pdtype,
                ),
                self.param(
                    "gating_einsum_lora_b",
                    self.lora_config.init_fn,
                    (2, self.lora_config.rank, self.hidden_dim),
                    pdtype,
                ),
            )
            self.w_linear_lora = (
                self.param(
                    "linear_lora_a",
                    self.lora_config.init_fn,
                    (self.hidden_dim, self.lora_config.rank),
                    pdtype,
                ),
                self.param(
                    "linear_lora_b",
                    self.lora_config.init_fn,
                    (self.lora_config.rank, self.features),
                    pdtype,
                ),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))


@at.typecheck
class Embedder(nn.Module):
    """Embedder module with optional explicit parameter dtype.

    Supports both standard embedding with default dtype and explicit dtype specification.
    """

    vocab_size: int
    embed_dim: int
    param_dtype: str | None = None

    def setup(self):
        pdtype = jnp.dtype(self.param_dtype) if self.param_dtype else None
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
            pdtype,
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


# ============================================================================
# Helper functions
# ============================================================================


def _name(base_name: str, expert_idx: int) -> str:
    """Generate layer names for multi-expert setup.

    The first expert's weights have no suffix (for seamless loading from checkpoints).
    Subsequent experts get numeric suffixes and are initialized from scratch.

    Example: _name("attn", 0) -> "attn", _name("attn", 1) -> "attn_1"
    """
    if expert_idx == 0:
        return base_name
    return f"{base_name}_{expert_idx}"


def _gated_residual(x, y, gate):
    """Apply gated residual connection: x + y * gate (or x + y if no gate)."""
    assert (x is None) == (y is None)
    if x is None:
        return None
    dtype = x.dtype
    result = (x + y) if gate is None else (x + y * gate)
    return result.astype(dtype)
